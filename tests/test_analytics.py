"""报表与检测测试。

重点覆盖旧版缺陷 2/3/5 在报表层的表现：

* 分位数报表要真的能报出异常（旧版异常集合恒为空）；
* **窗口秒数与次数阈值是两个独立参数**（旧版把它们说成一回事）；
* 报表聚合与「总账 == 分账之和」；
* 与账单对账能发现差异。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from costgovernor.analytics import (
    LedgerAnalytics,
    LedgerEntry,
    RetryBurst,
    WindowAnalytics,
    build_entries_from_costs,
    detect_retry_bursts,
    discover_days,
    parse_bucket_key,
    window_call_counts,
)
from costgovernor.cost import InsufficientSample, OutlierReport
from costgovernor.storage import IdempotentLedger, SlidingWindowCounter

DAY = date(2026, 10, 8)
OTHER_DAY = date(2026, 10, 7)
DAY_DATETIME = datetime(2026, 10, 8, 10, 0, 0)


def _q(value: str) -> Decimal:
    return Decimal(value).quantize(Decimal("0.000001"))


def _fill(ledger: IdempotentLedger, amounts: dict[str, str], *, day: datetime = DAY_DATETIME) -> None:
    for index, (model, amount) in enumerate(amounts.items()):
        ledger.add_entry(f"trace-{day.date()}-{index}-{model}", amount, dimension=model, at=day)


# ======================================================================================
# 滑动窗口：窗口与阈值是两个独立参数
# ======================================================================================
def test_window_seconds_and_max_calls_are_independent(entries_factory) -> None:
    """缺陷 3 的回归：窗口（秒）与阈值（次）必须能独立调，且互不串味。

    同一组数据（8 次调用，间隔 10 秒）：

    * ``window=30, max_calls=5`` —— 只看 30 秒，命中 4 次（0/10/20/30s）→ 不超阈值；
    * ``window=100, max_calls=5`` —— 看 100 秒，命中 8 次 → 超阈值；
    * ``window=100, max_calls=10`` —— 只把阈值抬高，就不再超阈值。

    如果实现把「窗口」当成「次数」（或反之），这三条断言会互相矛盾而失败。
    """
    entries = entries_factory([1.0] * 8, dimension="deepseek-flash", step_seconds=10.0)

    tight = detect_retry_bursts(entries, window_seconds=30, max_calls=5)
    assert tight == []  # 30 秒窗口内最多 4 次（含边界：0,10,20,30）

    wide = detect_retry_bursts(entries, window_seconds=100, max_calls=5)
    assert len(wide) == 1
    burst = wide[0]
    assert isinstance(burst, RetryBurst)
    assert burst.window_seconds == 100
    assert burst.max_calls == 5
    assert burst.count == 8  # 100 秒内能装下全部 8 次
    assert burst.span_seconds == 70.0

    relaxed = detect_retry_bursts(entries, window_seconds=100, max_calls=10)
    assert relaxed == []  # 只动阈值，结论就变了 → 阈值确实是独立的旋钮


def test_window_boundary_is_inclusive_on_both_ends(entries_factory) -> None:
    """窗口是闭区间：正好相差 ``window_seconds`` 的两次调用算在同一个窗口内。"""
    entries = entries_factory([1.0, 2.0], step_seconds=30.0)
    # 两点相隔 30 秒，window=30 时算两笔；max_calls=1 → 超阈值
    bursts = detect_retry_bursts(entries, window_seconds=30, max_calls=1)
    assert len(bursts) == 1
    assert bursts[0].count == 2


def test_bursts_are_coalesced_into_one_per_storm(entries_factory) -> None:
    """一次持续超阈值的突发只报一条，而不是每个超阈值的端点各报一条。

    12 次调用（间隔 1 秒）+ ``window=60, max_calls=8``：
    第 9..12 次调用都「落在超阈值的窗口里」，但它们属于**同一次**突发。
    """
    entries = entries_factory([0.1] * 12, step_seconds=1.0)
    bursts = detect_retry_bursts(entries, window_seconds=60, max_calls=8)
    assert len(bursts) == 1
    assert bursts[0].count == 12
    assert bursts[0].trace_ids[0].endswith("0000")
    assert bursts[0].trace_ids[-1].endswith("0011")


def test_retry_detection_groups_by_dimension(entries_factory) -> None:
    """不同维度各自独立计数，不会互相污染。"""
    entries = entries_factory([1.0] * 4, dimension="model-a", step_seconds=1.0)
    entries += entries_factory([1.0] * 4, dimension="model-b", step_seconds=1.0)
    bursts = detect_retry_bursts(entries, window_seconds=60, max_calls=5)
    assert bursts == []  # 每个维度只有 4 次
    bursts = detect_retry_bursts(entries, window_seconds=60, max_calls=3)
    assert {burst.dimension for burst in bursts} == {"model-a", "model-b"}


def test_retry_detection_validates_parameters() -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        detect_retry_bursts([], window_seconds=0, max_calls=5)
    with pytest.raises(ValueError, match="max_calls"):
        detect_retry_bursts([], window_seconds=60, max_calls=0)


def test_retry_detection_skips_entries_without_timestamp(entries_factory) -> None:
    entries = [entry for entry in entries_factory([1.0] * 10, step_seconds=1.0)]
    entries.extend(LedgerEntry(dimension="model-a", cost=Decimal("1"), day=DAY, at=None) for _ in range(10))
    bursts = detect_retry_bursts(entries, window_seconds=60, max_calls=5)
    assert len(bursts) == 1
    assert bursts[0].count == 10  # 只有带时间戳的 10 条参与


def test_window_call_counts_matches_redis_semantics() -> None:
    """纯函数计数与 ZSET 的 ``ZREMRANGEBYSCORE -inf cutoff`` 语义保持一致。"""
    base = 1000.0
    stamps = [base - 60, base - 59.5, base - 30, base - 1, base]
    # 右端闭、左端开：(end-60, end]
    assert window_call_counts(stamps, end=base, window_seconds=60) == 4
    assert window_call_counts(stamps, end=base, window_seconds=30) == 2
    assert window_call_counts([], end=base, window_seconds=60) == 0
    with pytest.raises(ValueError):
        window_call_counts(stamps, end=base, window_seconds=0)


def test_window_analytics_reads_from_zset(fake_redis, settings) -> None:
    counter = SlidingWindowCounter(fake_redis, settings, scope="llm")
    base = DAY_DATETIME.timestamp()
    for index in range(10):
        counter.record_and_count(f"trace-{index}", now=base + index, window_seconds=60, max_calls=8)
    analytics = WindowAnalytics(counter, settings)
    status = analytics.status(now=base + 9, window_seconds=60, max_calls=8)
    assert status["count"] == 10
    assert status["exceeded"] is True
    assert status["window_seconds"] == 60
    assert status["max_calls"] == 8

    # 窗口变短 → 计数下降（同一份 ZSET，只是清理范围不同）
    short = analytics.status(now=base + 9, window_seconds=5, max_calls=8)
    # 清理用的是 ZREMRANGEBYSCORE -inf (now-window)，两端都是闭区间，
    # 因此恰好落在 now-window 上的成员也会被清掉：(base+4, base+9] 共 5 条。
    assert short["count"] == 5
    assert short["exceeded"] is False


# ======================================================================================
# 报表聚合
# ======================================================================================
def test_daily_report_aggregates_by_axis(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    ledger.add_entry("t1", "1.0", dimension="deepseek-flash", at=DAY_DATETIME, dimensions={"module": "report"})
    ledger.add_entry("t2", "2.0", dimension="deepseek-v4-pro", at=DAY_DATETIME, dimensions={"module": "report"})
    ledger.add_entry("t3", "3.0", dimension="deepseek-flash", at=DAY_DATETIME, dimensions={"module": "chat"})

    analytics = LedgerAnalytics(ledger, settings)
    report = analytics.daily_report(DAY)
    assert report.total == _q("6.0")
    assert report.by_axis["model"] == {"deepseek-flash": _q("4.0"), "deepseek-v4-pro": _q("2.0")}
    assert report.by_axis["module"] == {"report": _q("3.0"), "chat": _q("3.0")}
    assert report.by_dimension == report.by_axis["model"]
    # 分位数报表按「分组金额」计算：样本是 [4.0, 2.0]
    assert report.quantiles["count"] == 2
    assert report.quantiles["p50"] == pytest.approx(3.0)
    assert "report" in report.render()
    assert report.to_dict()["total"] == "6.0"


def test_daily_report_on_empty_day_explains_itself(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    report = LedgerAnalytics(ledger, settings).daily_report(date(2030, 1, 1))
    assert report.total == 0
    assert report.count == 0
    assert "TTL" in report.note  # 明确说明「为什么没有数据」


def test_overview_sums_across_days(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    _fill(ledger, {"deepseek-flash": "1.0", "deepseek-v4-pro": "2.0"})
    _fill(
        ledger,
        {"deepseek-flash": "0.5"},
        day=datetime(2026, 10, 7, 10, 0, 0),
    )
    analytics = LedgerAnalytics(ledger, settings)
    overview = analytics.overview()
    assert len(overview["days"]) == 2
    assert overview["total"] == _q("3.5")
    assert overview["by_axis"]["model"]["deepseek-flash"] == _q("1.5")
    assert overview["by_axis"]["model"]["deepseek-v4-pro"] == _q("2.0")


def test_discover_days_and_parse_bucket_key(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    _fill(ledger, {"deepseek-flash": "1.0"})
    _fill(ledger, {"deepseek-flash": "1.0"}, day=datetime(2026, 10, 7, 10))
    analytics = LedgerAnalytics(ledger, settings)
    assert analytics.discover_days() == [OTHER_DAY, DAY]
    assert parse_bucket_key("llm:cost:2026-10-08") == DAY
    assert parse_bucket_key("llm:cost:2026-10-08:model") == DAY
    assert parse_bucket_key("llm:days:cost") is None
    assert parse_bucket_key("other:cost:2026-10-08") is None
    assert parse_bucket_key("llm:cost:not-a-date") is None
    assert discover_days(fake_redis, prefix="llm", kind="cost") == [OTHER_DAY, DAY]


def test_token_totals_are_reported(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    ledger.add_entry(
        "t1",
        "1.0",
        dimension="deepseek-flash",
        at=DAY_DATETIME,
        tokens={"cache_hit": 100, "cache_miss": 200, "output": 50},
    )
    report = LedgerAnalytics(ledger, settings).daily_report(DAY)
    assert report.tokens["cache_hit_tokens"] == _q("100")
    assert report.tokens["output_tokens"] == _q("50")


# ======================================================================================
# 离群检测（报表口径）
# ======================================================================================
def test_outliers_returns_insufficient_sample(fake_redis, settings) -> None:
    """只有 2 个模型 → 样本不足，明确返回而不是硬算。"""
    ledger = IdempotentLedger(fake_redis, settings)
    _fill(ledger, {"deepseek-flash": "1.0", "deepseek-v4-pro": "2.0"})
    result = LedgerAnalytics(ledger, settings).outliers(DAY)
    assert isinstance(result, InsufficientSample)
    assert result.sample_size == 2
    assert result.required == settings.min_sample_size


def test_outliers_detects_spike_among_groups(fake_redis, settings) -> None:
    """40 个正常分组 + 1 个明显偏高的分组 → modified z-score 抓出它。"""
    ledger = IdempotentLedger(fake_redis, settings)
    for index in range(40):
        ledger.add_entry(f"normal-{index}", "1.0", dimension=f"model-{index:02d}", at=DAY_DATETIME)
    ledger.add_entry("spike", "100", dimension="model-spike", at=DAY_DATETIME)

    result = LedgerAnalytics(ledger, settings).outliers(DAY)
    assert isinstance(result, OutlierReport)
    assert result.sample_size == 41
    assert result.outlier_count == 1
    assert result.outliers[0].direction == "high"


def test_outliers_from_costs_uses_call_level_data(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    analytics = LedgerAnalytics(ledger, settings)
    costs = [0.5 + (index % 5) * 0.01 for index in range(60)] + [9.99]
    result = analytics.outliers_from_costs(costs)
    assert isinstance(result, OutlierReport)
    assert result.outlier_count == 1
    assert result.outliers[0].value == pytest.approx(9.99)


# ======================================================================================
# 总账 vs 分账
# ======================================================================================
def test_split_check_is_balanced_after_many_writes(fake_redis, settings) -> None:
    """缺陷 5 的报表层断言：分账之和 == 总账。"""
    ledger = IdempotentLedger(fake_redis, settings)
    for index in range(25):
        ledger.add_entry(
            f"trace-{index}",
            f"{0.1 * (index + 1):.2f}",
            dimension=f"model-{index % 5}",
            at=DAY_DATETIME,
            dimensions={"module": f"mod-{index % 3}"},
        )
    analytics = LedgerAnalytics(ledger, settings)
    check = analytics.split_check()
    assert check.balanced is True
    assert check.difference == 0
    assert check.total == analytics.grand_total()
    assert check.total == analytics.grand_total([DAY])
    # 每个维度轴自己也得平
    assert analytics.split_check(axis="module").balanced is True


def test_split_check_detects_drift(fake_redis, settings) -> None:
    """人为制造漂移（模拟旧版「总账与分账各自漂移」），必须被判定为不平。"""
    ledger = IdempotentLedger(fake_redis, settings)
    _fill(ledger, {"deepseek-flash": "1.0", "deepseek-v4-pro": "2.0"})
    # 直接改总账，模拟「两份数据源不一致」
    fake_redis.hincrbyfloat("llm:cost:2026-10-08", "deepseek-flash", "0.25")

    check = LedgerAnalytics(ledger, settings).split_check()
    assert check.balanced is False
    assert check.difference == _q("0.25")


def test_totals_by_axis(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    _fill(ledger, {"deepseek-flash": "1.0"}, day=datetime(2026, 10, 8, 10))
    _fill(ledger, {"deepseek-flash": "2.0"}, day=datetime(2026, 10, 7, 10))
    analytics = LedgerAnalytics(ledger, settings)
    assert analytics.totals() == {"deepseek-flash": _q("3.0")}
    assert analytics.grand_total() == _q("3.0")


# ======================================================================================
# 对账
# ======================================================================================
def test_reconcile_matches_within_tolerance(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    _fill(ledger, {"deepseek-flash": "1.0", "deepseek-v4-pro": "2.0"})
    result = LedgerAnalytics(ledger, settings).reconcile("3.0")
    assert result.our_total == _q("3.0")
    assert result.mismatch is False
    assert "✅" in result.summary()


def test_reconcile_detects_difference(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    _fill(ledger, {"deepseek-flash": "1.0", "deepseek-v4-pro": "2.0"})
    result = LedgerAnalytics(ledger, settings).reconcile("3.5")
    assert result.difference == _q("-0.5")
    assert result.ratio == pytest.approx(Decimal("0.5") / Decimal("3.5"))
    assert result.ratio_percent > Decimal("14")
    assert result.mismatch is True
    assert "❌" in result.summary()
    assert result.to_dict()["mismatch"] is True


def test_reconcile_uses_custom_tolerance(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    _fill(ledger, {"deepseek-flash": "1.0"})
    analytics = LedgerAnalytics(ledger, settings)
    assert analytics.reconcile("1.2").mismatch is True  # 20% 差异
    assert analytics.reconcile("1.2", tolerance="0.25").mismatch is False
    assert analytics.reconcile("1.2", tolerance=Decimal("0.5")).mismatch is False


def test_reconcile_with_no_data(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    result = LedgerAnalytics(ledger, settings).reconcile("0")
    assert result.our_total == 0
    assert result.mismatch is False


# ======================================================================================
# 重试突发的数据来源
# ======================================================================================
def test_retry_bursts_returns_empty_without_timestamped_data(fake_redis, settings) -> None:
    """分账 Hash 没有时间戳 → 明确返回空并记警告，而不是硬滤出一个假窗口。"""
    ledger = IdempotentLedger(fake_redis, settings)
    _fill(ledger, {"deepseek-flash": "1.0"})
    analytics = LedgerAnalytics(ledger, settings)
    assert analytics.retry_bursts() == []
    assert analytics.load_entries() == []


def test_retry_bursts_accepts_explicit_entries(fake_redis, settings, entries_factory) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    analytics = LedgerAnalytics(ledger, settings)
    entries = entries_factory([0.1] * 12, step_seconds=1.0)
    bursts = analytics.retry_bursts(entries=entries, window_seconds=60, max_calls=8)
    assert len(bursts) == 1
    assert bursts[0].count == 12
    assert bursts[0].window_seconds == 60
    assert bursts[0].max_calls == 8
    assert len(bursts[0].trace_ids) == 12


def test_window_counts_requires_counter(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    analytics = LedgerAnalytics(ledger, settings)
    with pytest.raises(RuntimeError, match="SlidingWindowCounter"):
        analytics.window_counts()


def test_window_counts_with_counter(fake_redis, settings) -> None:
    ledger = IdempotentLedger(fake_redis, settings)
    counter = SlidingWindowCounter(fake_redis, settings, scope="llm")
    base = DAY_DATETIME.timestamp()
    for index in range(3):
        counter.record_and_count(f"t{index}", now=base + index, window_seconds=60, max_calls=8)
    analytics = LedgerAnalytics(ledger, settings, windows=counter)
    assert analytics.window_counts(now=base + 3, window_seconds=60) == 3


def test_build_entries_from_costs_helper() -> None:
    entries = build_entries_from_costs([0.1, 0.2], start=DAY_DATETIME, step_seconds=5)
    assert [entry.cost for entry in entries] == [Decimal("0.1"), Decimal("0.2")]
    assert entries[1].at - entries[0].at == timedelta(seconds=5)
    assert build_entries_from_costs([0.1])[0].at is None


# ======================================================================================
# 可选异步路径（Celery）
# ======================================================================================
def test_build_celery_app_returns_none_without_broker() -> None:
    """未配置 broker 时返回 None，且**不在导入期连接任何东西**。"""
    from costgovernor.settings import Settings
    from costgovernor.tasks import build_celery_app

    assert build_celery_app(Settings(celery_broker_url=None)) is None


def test_build_celery_app_builds_with_broker() -> None:
    """配了 broker 也只构造对象，不发起连接（Celery 是惰性连接）。"""
    pytest.importorskip("celery")
    from costgovernor.settings import Settings
    from costgovernor.tasks import build_celery_app, make_celery_task

    app = build_celery_app(Settings(celery_broker_url="redis://127.0.0.1:6379/1"))
    assert app is not None
    assert app.conf.task_default_queue == "costgov"
    task = make_celery_task(app)
    assert task is not None
    assert make_celery_task(None) is None


def test_compute_and_record_is_idempotent_across_retries(fake_redis, settings) -> None:
    """旧版缺陷 4 的端到端回归：任务重试复用同一 trace_id，账本只累加一次。

    这里直接调用 :func:`compute_and_record`（不经过 Celery），
    因为它在没有 broker / worker 的环境里也能跑，且与任务体共用同一份逻辑。
    """
    from costgovernor.bootstrap import Container
    from costgovernor.tasks import compute_and_record

    container = Container(settings=settings, client=fake_redis)
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    results = [
        compute_and_record("trace-celery", "deepseek-flash", usage, at=DAY_DATETIME, container=container)
        for _ in range(4)  # 1 次执行 + 3 次重试
    ]
    assert [item["duplicate"] for item in results] == [False, True, True, True]
    assert container.ledger.bucket_total(DAY) == _q("2.0")  # 只累加一次（1M × 2 元/M）


def test_record_llm_usage_retries_with_same_trace_id(fake_redis, settings, monkeypatch) -> None:
    """任务体在失败时调用 ``self.retry``，并且**参数里带的是同一个 trace_id**。"""
    from costgovernor import tasks as tasks_module

    attempts: list[tuple[str, str]] = []

    def fake_compute(trace_id: str, model: str, usage: Mapping[str, int], **kwargs: Any) -> dict[str, Any]:
        attempts.append((trace_id, model))
        if len(attempts) == 1:
            raise ConnectionError("broker 抖动")
        return {"trace_id": trace_id, "duplicate": False}

    monkeypatch.setattr(tasks_module, "compute_and_record", fake_compute)

    class FakeSelf:
        max_retries = 3

        class request:  # noqa: N801 - 模拟 Celery 的 request 属性
            retries = 0

        def retry(self, exc: BaseException, args: tuple[Any, ...]) -> None:
            # 关键断言：重试参数里复用了同一个 trace_id
            assert args[0] == "trace-retry-same"
            self.request.retries += 1
            tasks_module.record_llm_usage(self, *args)

    tasks_module.record_llm_usage(FakeSelf(), "trace-retry-same", "deepseek-flash", {"input_tokens": 1})
    assert attempts == [
        ("trace-retry-same", "deepseek-flash"),
        ("trace-retry-same", "deepseek-flash"),
    ]
