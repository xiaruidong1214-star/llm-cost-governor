"""测试夹具：FakeRedis 替身、fakeredis 与**真实 Redis** 三种后端。

为什么要有自己的 ``FakeRedis`` 替身（而不是只用 fakeredis）：

* ``storage.py`` 刻意只依赖**很小的 Redis 命令表面**，替身就是这个契约的可执行版本 ——
  一旦有人在存储层用了替身不认识的命令，测试会以 ``NotImplementedError`` 立刻失败，
  从而防止命令表面悄悄膨胀；
* 替身里出现的 ``eval`` **不是真正的 Lua 解释器**（见 ``FakeRedis.eval`` 的注释），
  它只按脚本文本分派到对应的 Python 实现，用来验证「幂等」「窗口清理」等语义。

**关于"是否发真实网络请求"**（这句话以前写错过，特此写明）：

* 默认（未设置 ``CG_TEST_REDIS_URL``）**不发任何真实连接**，全部走内存替身；
* 设置 ``CG_TEST_REDIS_URL`` 后，``any_redis`` 会额外加入 ``real`` 参数，
  **真的连上去、真的执行 Lua**；
* 设置 ``CG_REQUIRE_REAL_REDIS=1`` 时，「拿不到真实 Redis」不再是跳过而是**失败**。
  这是防止「Redis 没起来 → 用例静默 skip → 整体仍显示绿灯」的关键开关，
  CI 里会打开它。

为什么真实 Redis 这一档不能省：替身按脚本标记分派到等价 Python 实现，
**发现不了 Lua 下标错位、KEYS/ARGV 布局错误、key 折叠**这类问题 ——
这些在开发中真实发生过 4 次（见 README 第 6 节）。
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest

from costgovernor.settings import Settings
from costgovernor.testing import InMemoryRedis, decode_value

# ======================================================================================
# FakeRedis 替身
# ======================================================================================
_REAL_REDIS_URL = os.environ.get("CG_TEST_REDIS_URL")
#: 为 1 时，「真实 Redis 不可用」直接判失败，而不是跳过。
_REQUIRE_REAL_REDIS = os.environ.get("CG_REQUIRE_REAL_REDIS", "").strip().lower() in {"1", "true", "yes"}


class FakeRedis(InMemoryRedis):
    """测试专用替身：在 :class:`~costgovernor.testing.InMemoryRedis` 之上加测试便利。

    脚本语义（``eval``）、时钟推进（``advance``）、``ttl`` 查询都由
    ``InMemoryRedis`` 提供，这里**不重复实现**，避免「演示替身」与「测试替身」
    两套语义各自漂移（这个坑在开发过程中真的踩到过）。

    仍然要强调：``eval`` **不是 Lua 解释器**，它按脚本文本分派到等价的 Python 实现。
    """

    def __init__(self) -> None:
        super().__init__()


#: 兼容旧名字
decode = decode_value

# ======================================================================================
# 可选：fakeredis 后端
# ======================================================================================
def _fakeredis_or_none() -> Any | None:
    """返回一个支持 ``EVAL`` 的 fakeredis 客户端；不支持则返回 ``None``。

    当前环境未安装 ``lupa``，因此 fakeredis 没有 EVAL 能力（会抛
    ``ResponseError: unknown command 'eval'``）。这种情况下本套件自动跳过
    fakeredis 参数，只用自带的 ``FakeRedis`` 替身验证脚本语义 ——
    这是**已知的验证缺口**，README「已知限制」里已如实记录。
    """
    try:
        import fakeredis
    except ImportError:  # pragma: no cover
        return None
    try:
        client = fakeredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
        client.eval("return 1", 0)  # 探测 Lua 支持（依赖 lupa）
        return client
    except Exception:  # noqa: BLE001 - 缺少 lupa 时不使用该后端
        return None


_FAKEREDIS_CLIENT = _fakeredis_or_none()


# ======================================================================================
# 固件
# ======================================================================================
@pytest.fixture
def fake_redis() -> FakeRedis:
    """纯 Python 替身（默认后端，零外部依赖）。"""
    return FakeRedis()


@pytest.fixture
def settings() -> Settings:
    """测试配置：TTL 缩短、样本量下调，方便断言。"""
    return Settings(
        redis_url="redis://localhost:6379/0",
        key_prefix="llm",
        ttl_seconds=86400,
        ttl_index_seconds=86400 * 3,
        ttl_dedup_seconds=86400 * 7,
        ttl_window_seconds=3600,
        min_sample_size=30,
        mad_threshold=3.5,
        retry_window_seconds=60,
        retry_max_calls=8,
        reconcile_tolerance=0.01,
    )


@pytest.fixture
def small_sample_settings(settings: Settings) -> Settings:
    """把最小样本量降到 5，便于构造小样本 MAD 用例。"""
    return settings.model_copy(update={"min_sample_size": 5})


@pytest.fixture(params=["fake", "fakeredis", "real"])
def any_redis(request: pytest.FixtureRequest) -> Iterator[Any]:
    """参数化三种后端：自家替身 + fakeredis + **真实 Redis**。

    三种后端跑同一批断言，可以避免「只在替身上正确」的实现。

    ``real`` 分支是有意保留的：替身里的 ``eval`` 不是 Lua 解释器，
    它只能验证**语义契约**，无法验证 Lua **语法**是否合法、KEYS/ARGV 下标是否越界
    （越界在真实 Redis 上会直接报错，而在替身里可能被静默容忍）。
    因此在有真实 Redis 的环境里（本地 Docker、CI 的 service 容器）
    必须让这批用例真的跑一次。

    连不上真实 Redis 时：默认**跳过并说明原因**（本地无 Redis 仍可跑全量）；
    但若设置了 ``CG_REQUIRE_REAL_REDIS=1``，则改为**失败** ——
    这样 CI 不会因为「Redis service 没起来」而拿到一个假绿灯。
    """
    if request.param == "fake":
        yield FakeRedis()
        return

    if request.param == "fakeredis":
        if _FAKEREDIS_CLIENT is None:
            pytest.skip("fakeredis 及其 Lua 支持（lupa）不可用")
        client = fakeredis_client()
        try:
            yield client
        finally:
            try:
                client.flushall()
            except Exception:  # noqa: BLE001 - 清理失败不影响断言结果
                pass
        return

    # ---- real：真实 Redis ----
    def _unavailable(reason: str) -> None:
        """按 CG_REQUIRE_REAL_REDIS 决定「跳过」还是「失败」。"""
        if _REQUIRE_REAL_REDIS:
            pytest.fail(
                f"要求真实 Redis（CG_REQUIRE_REAL_REDIS=1）但不可用：{reason}。"
                "替身无法发现 Lua 下标/key 布局错误，因此这不能当作通过。"
            )
        pytest.skip(reason)

    if not _REAL_REDIS_URL:
        _unavailable("未设置 CG_TEST_REDIS_URL")
    try:
        import redis as _redis
    except ImportError:  # pragma: no cover - 依赖缺失
        _unavailable("未安装 redis 客户端库")

    client = _redis.Redis.from_url(_REAL_REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001
        _unavailable(f"真实 Redis 不可达（{_REAL_REDIS_URL}）：{type(exc).__name__}")
    client.flushdb()
    try:
        yield client
    finally:
        try:
            client.flushdb()
            client.close()
        except Exception:  # noqa: BLE001 - 清理失败不影响断言结果
            pass


def fakeredis_client() -> Any:
    """新建一个干净的 fakeredis 客户端。"""
    import fakeredis

    return fakeredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)


def pytest_report_header(config: pytest.Config) -> list[str]:
    """在测试头部打印后端信息，便于判断跳过了哪些用例。"""
    real_state = "已设置（真实 Redis 用例会执行）" if _REAL_REDIS_URL else "未设置（真实 Redis 用例将跳过）"
    enforce = "开启（不可用即失败）" if _REQUIRE_REAL_REDIS else "关闭（不可用则跳过）"
    return [
        f"costgovernor 测试后端：FakeRedis 替身（内置）+ "
        f"fakeredis={'可用' if _FAKEREDIS_CLIENT is not None else '不可用'}",
        f"CG_TEST_REDIS_URL：{real_state}",
        f"CG_REQUIRE_REAL_REDIS：{enforce}",
        "提示：替身只能验证 Lua 的语义契约，无法验证 Lua 语法与下标；"
        "真实 Redis 那一档不可省（见 README 第 6 节记录过的 4 个真实缺陷）。",
    ]


# ======================================================================================
# 测试数据工厂
# ======================================================================================
#: 高峰时刻：北京时间 2026-10-08（周四）10:00
PEAK_MOMENT = datetime(2026, 10, 8, 10, 0, 0)
#: 空闲时刻：北京时间 2026-10-08（周四）20:00
OFF_PEAK_MOMENT = datetime(2026, 10, 8, 20, 0, 0)
#: 周末时刻：北京时间 2026-10-10（周六）10:00
WEEKEND_MOMENT = datetime(2026, 10, 10, 10, 0, 0)
#: 测试用固定日期
FIXED_DAY = date(2026, 10, 8)


@dataclass(frozen=True)
class FakeLLMResponse:
    """模拟 SDK 的响应对象（OpenAI 风格），用于测试 usage 解析。"""

    text: str = "你好"
    usage: Any = None


@dataclass(frozen=True)
class FakeUsage:
    """模拟 SDK 的 usage 对象。"""

    prompt_tokens: int = 10
    completion_tokens: int = 5


@pytest.fixture
def entries_factory() -> Any:
    """构造 :class:`~costgovernor.analytics.LedgerEntry` 的工厂。"""

    def build(
        costs: Iterable[float],
        *,
        dimension: str = "deepseek-flash",
        start: datetime = PEAK_MOMENT,
        step_seconds: float = 1.0,
        day: date = FIXED_DAY,
    ) -> list[Any]:
        from datetime import timedelta

        from costgovernor.analytics import LedgerEntry

        result = []
        for index, cost in enumerate(costs):
            result.append(
                LedgerEntry(
                    dimension=dimension,
                    cost=Decimal(str(cost)),
                    day=day,
                    at=start + timedelta(seconds=index * step_seconds),
                    trace_id=f"trace-{dimension}-{index:04d}",
                )
            )
        return result

    return build
