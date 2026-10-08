"""README 里所有代码示例的验证脚本（手动运行，不进测试套件）。

用法::

    python tools/readme_check.py

任何一段示例与实现不一致都会在这里暴露，避免 README 变成「文档谎言」。
"""

from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal


def main() -> int:
    # ---------- 示例 1：纯函数算成本（不需要 Redis） ----------
    from costgovernor.cost import compute_cost

    at = datetime(2026, 10, 8, 10, 0, 0)  # 北京时间周四 10:00 → 高峰时段
    breakdown = compute_cost(
        "deepseek-flash",
        input_tokens=1_000_000,
        cache_hit_tokens=800_000,
        output_tokens=200_000,
        at=at,
    )
    assert breakdown.total == Decimal("2.032"), breakdown.total
    assert breakdown.cache_savings == Decimal("1.568"), breakdown.cache_savings
    assert breakdown.tier == "peak"
    assert breakdown.cost_cache_hit_input == Decimal("0.032")
    assert breakdown.cost_cache_miss_input == Decimal("0.4")
    assert breakdown.cost_output == Decimal("1.6")
    print("示例 1 OK：total=", breakdown.total, " 节省=", breakdown.cache_savings)

    # ---------- 示例 2：分位数（与 statistics.quantiles 一致） ----------
    from statistics import quantiles

    from costgovernor.cost import percentile

    costs = [0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34, 0.55, 0.89]
    assert math.isclose(percentile(costs, 0.5), 0.105)
    assert math.isclose(percentile(costs, 0.95), 0.737)
    assert math.isclose(percentile(costs, 0.95), quantiles(costs, n=20, method="inclusive")[18])
    assert percentile(costs, 0.95) != max(costs)  # 旧实现会取到最大值
    print("示例 2 OK：p50=", percentile(costs, 0.5), " p95=", percentile(costs, 0.95))

    # ---------- 示例 3：MAD 离群检测 ----------
    from costgovernor.cost import InsufficientSample, OutlierReport, detect_outliers_by_mad

    normal = [1.0 + (index % 7) * 0.01 for index in range(40)]
    result = detect_outliers_by_mad(normal, threshold=3.5, min_sample_size=30)
    assert isinstance(result, OutlierReport)
    assert result.outlier_count == 0

    spiked = normal + [100.0]
    result = detect_outliers_by_mad(spiked, threshold=3.5, min_sample_size=30)
    assert isinstance(result, OutlierReport)
    assert result.outlier_count == 1
    assert result.outliers[0].direction == "high"

    small = detect_outliers_by_mad([1.0, 2.0, 3.0], min_sample_size=30)
    assert isinstance(small, InsufficientSample)
    print("示例 3 OK：离群", result.outlier_count, "个 /", small)

    # ---------- 示例 4：内存替身 + 幂等入账 ----------
    from costgovernor.settings import Settings
    from costgovernor.storage import IdempotentLedger, SlidingWindowCounter
    from costgovernor.testing import InMemoryRedis

    redis = InMemoryRedis()
    settings = Settings(ttl_seconds=172_800, ttl_dedup_seconds=604_800)
    ledger = IdempotentLedger(redis, settings)
    for _ in range(3):  # 模拟 Celery 重试 3 次
        ledger.add_entry("trace-abc", "0.42", dimension="deepseek-flash", at=at)
    day = at.date()
    assert ledger.bucket_total(day) == Decimal("0.42"), ledger.bucket_total(day)
    assert ledger.dimension_totals(day, axis="model") == {"deepseek-flash": Decimal("0.42")}

    counter = SlidingWindowCounter(redis, settings, scope="demo")
    now = at.timestamp()
    for index in range(10):
        status = counter.record_and_count(f"trace-{index}", now=now + index, window_seconds=60, max_calls=8)
    assert status.count_after == 10 and status.exceeded is True
    print("示例 4 OK：总账=", ledger.bucket_total(day), " 窗口计数=", status.count_after)

    # ---------- 示例 5：装饰器（同步 + 异步 + 失败必记账） ----------
    import asyncio

    from costgovernor.tracker import LLMCallRecord, track_llm

    records: list[LLMCallRecord] = []

    @track_llm(model="deepseek-flash", module="demo", recorder=records.append, now=lambda: at)
    def ask(prompt: str) -> dict[str, object]:
        return {"text": "hi", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}

    ask("你好")
    assert records[-1].status == "ok" and records[-1].cost == Decimal("2.0")

    @track_llm(model="deepseek-flash", module="demo", recorder=records.append, now=lambda: at)
    def failing(prompt: str) -> None:
        raise TimeoutError("上游超时")

    try:
        failing("你好")
    except TimeoutError:
        pass
    assert records[-1].status == "error" and records[-1].error_type == "TimeoutError"

    @track_llm(model="deepseek-flash", module="demo", recorder=records.append, now=lambda: at)
    async def ask_async(prompt: str) -> dict[str, object]:
        return {"usage": {"prompt_tokens": 1_000_000}}

    asyncio.run(ask_async("你好"))
    assert records[-1].status == "ok" and records[-1].cost == Decimal("2.0")
    print("示例 5 OK：记录数=", len(records), " 最后一条=", records[-1].status)

    # ---------- 示例 6：对账 ----------
    from costgovernor.cost import reconcile

    result = reconcile("2.016", "2.10", tolerance=Decimal("0.01"))
    assert result.difference == Decimal("-0.084")
    assert result.mismatch is True
    print("示例 6 OK：", result.summary())

    print("\n全部示例通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
