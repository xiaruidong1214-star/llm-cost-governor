# 拆分说明：这个仓库从哪里来

## 一句话

本仓库是 **LLM 调用成本治理**部分，从一个**混装仓库**中独立出来的结果。

## 背景

我原先把两套完全无关的系统放在同一个仓库里：

- `app/` 目录 —— FastAPI + Celery + Redis + SQLite 的**代码审查系统**
- 根目录的 `tracker.py` / `tasks.py` / `report.py` / `config.py` —— **本仓库的前身**

它们共用同一个仓库，却**各自使用互不相干的 Redis 配置**（审查系统走 `settings.REDIS_URL`，
成本治理手搓 `REDIS_HOST`）。更糟的是，同一份代码被推到了两个仓库
（`try` 与 `llm-tracker-current`），读起来像"两个项目"，实际只是一个项目推了两遍。

## 现在怎么分

| 原仓库中的东西 | 现在归属 |
|---|---|
| `app/`（FastAPI、Celery、AST 分析、缓存、SQLite、GitHub 抓取） | [`code-review-bot`](https://github.com/xiaruidong1214-star/code-review-bot) |
| `tracker.py` / `tasks.py` / `report.py` / `config.py` / `mock_llm.py` | **本仓库** `llm-cost-governor` |

两个仓库都**独立可运行**，各自有自己的 `pyproject.toml`、依赖、配置、测试、CI 与 Dockerfile，
**不存在跨仓库依赖**。

## 拆分时顺带修掉的真实缺陷

重构不是搬文件。原实现在成本治理这块有 6 处实质问题，全部已修复并补了回归测试：

| # | 原实现的问题 | 现在的做法 |
|---|---|---|
| 1 | 价格表注释写"元/**1K** tokens"，实际官方按**百万 tokens** 计费，且分**高峰/空闲**与**缓存命中/未命中**（命中价低至未命中的 1/50）。**总成本量级与口径全错。** | 价格表以「元 / 百万 tokens」为单位（`PRICE_UNIT` 是**常量**，可被测试断言），含 `effective_date`、`peak`/`off_peak`、`cache_hit`/`cache_miss` 四档，按北京时间工作日 9:00-12:00、14:00-18:00 判高峰 |
| 2 | 未知模型**静默回退默认价**，把"配置漏写"掩盖成"成本偏低" | 显式抛 `UnknownModelError`，第一次调用就暴露问题 |
| 3 | 分位数实现 `idx = int(n*0.95); anomalies = cost > costs[idx]`：因为 `costs` 已升序且 `idx` 落在末位，`p95` **就是最大值**，`anomalies` **在数学上恒为空**。变量名还叫 `P99_KEY` 却用 0.95 | 真正的**线性插值**分位数（与 `statistics.quantiles` 逐点一致），并实现基于**中位数 + MAD** 的离群检测；小样本（<30）返回 `InsufficientSample` 而不是硬算 |
| 4 | 「无效重试」用 `LRANGE llm:trace 0 -1` **拉全表**再在 Python 端逐条 `json.loads`，每次调用都是 O(全部历史)；"窗口"实际由写入时的 `LTRIM 10000` 决定而非时间。且 `threshold=8`（次数）被口头说成"60 秒" | **ZSET 滑动窗口**：`ZREMRANGEBYSCORE` 清理 + `ZCARD` 计数 + `ZADD` 记录，三步在**一个 Lua 脚本内原子完成**；`window_seconds` 与 `max_calls` 是**两个独立参数** |
| 5 | Celery `max_retries=3` 导致**同一次逻辑调用的成本被重复累加**；且没有 `trace_id`，无法把重试归并到同一逻辑请求 | 引入 `trace_id`，用 **Lua 幂等入账**（同 trace_id 只入账一次）。测试断言"同一 trace_id 上报 3 次，总成本只加 1 次" |
| 6 | 所有 Redis key **无 TTL**，Hash field 是 `session:module` 永久增长；模块成本从 Hash 聚合、总成本从 List 求和，两份数据一个会过期一个不会，**总账与分账永远对不平** | 所有 key 带 TTL + **按天分桶**；同一笔金额**按每个维度轴各写一份**，使「分账之和 == 总账」成为**结构事实**而非事后配平；提供 `reconcile()` 与账单对账 |

另外，装饰器层面的问题也一并修掉：

| # | 原实现的问题 | 现在的做法 |
|---|---|---|
| 7 | `def wrapper` 里 `result = func(*args, **kwargs)` 写在 `try` **之外**：① 被装饰的 async 函数会返回协程对象，下一行 `result.get("usage")` 直接 `AttributeError` 且被吞掉只 `print` 一行；② **函数抛异常时埋点代码根本不执行**，失败调用全部漏记——而失败恰恰是成本治理最该抓的部分 | 装饰器**同时支持同步与异步**，用 `perf_counter` 计时，**失败调用必记账**（记一条 `status="error"` 后原样抛出），日志用 structlog 而非 `print` |
| 8 | usage 解析假定返回 dict，拿不到就静默记 0 | 健壮解析：支持 dict、支持带 `.usage` 的对象（OpenAI SDK 风格）；拿不到时标 `estimated=True` 并在 README 声明**不可用于对账** |

## 旧仓库

原来的混装仓库已重命名并归档保留，作为这段历史的凭证，不再更新：

- `xiaruidong1214-star/legacy-archive-mixed-repo`（原 `try`）
- `xiaruidong1214-star/legacy-archive-llm-tracker`（原 `llm-tracker-current`）

## 参考

- 详细架构、设计取舍与**已知限制**见 [README.md](README.md)
- 姊妹项目：[code-review-bot](https://github.com/xiaruidong1214-star/code-review-bot)
