"""纯函数算术测试：分位数、MAD 离群、成本计算、对账。

这些是不需要 Redis 的最强测试：旧版的「P95 恒等于最大值、异常集合恒为空」
就是在这一层被抓住的。
"""

from __future__ import annotations

import statistics
from datetime import datetime
from decimal import Decimal

import pytest

from costgovernor.cost import (
    DEFAULT_MAD_CONSTANT,
    DEFAULT_MAD_THRESHOLD,
    InsufficientSample,
    OutlierReport,
    check_ledger_split,
    compute_cost,
    detect_outliers_by_mad,
    median,
    median_absolute_deviation,
    modified_zscore,
    percentile,
    percentile_anomalies,
    quantile_summary,
    reconcile,
    sum_cost_breakdowns,
)
from costgovernor.pricing import PRICE_UNIT, UnknownModelError

PEAK = datetime(2026, 10, 8, 10, 0, 0)  # 周四 10:00（高峰）
OFF_PEAK = datetime(2026, 10, 8, 20, 0, 0)  # 周四 20:00（空闲）


# ======================================================================================
# 分位数
# ======================================================================================
def test_percentile_matches_statistics_quantiles() -> None:
    """与标准库 ``statistics.quantiles(method="inclusive")`` 对照。

    ``statistics.quantiles`` 只支持把数据等分成 n 份，因此分别用 n=4/10/20
    覆盖 25/50/75、10..90、5..95 这些分位点。
    """
    data = [float(value) for value in range(1, 101)]  # 1..100
    for divisions in (4, 10, 20):
        reference = statistics.quantiles(data, n=divisions, method="inclusive")
        for index, expected in enumerate(reference, start=1):
            q = index / divisions
            assert percentile(data, q) == pytest.approx(expected), f"q={q}"


def test_percentile_matches_numpy_linear() -> None:
    """与 ``numpy.quantile(..., method="linear")`` 对照（默认插值方法）。"""
    numpy = pytest.importorskip("numpy")
    samples = [
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 0.25, 0.75, 1.0, 3.0],
        list(range(50)),
        [12.5] * 9 + [1000.0],
    ]
    quantiles = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    for data in samples:
        for q in quantiles:
            assert percentile(data, q) == pytest.approx(
                float(numpy.quantile(data, q, method="linear"))
            ), f"data={data} q={q}"


def test_percentile_interpolates_not_indexes() -> None:
    """缺陷 2 的核心回归：不能用 ``int(n*q)`` 取下标。

    旧版 ``idx = int(len(costs) * 0.95); p95 = costs[idx]`` 在升序数据上
    取到的就是最大值，于是 ``cost > p95`` 恒为空。
    """
    data = [1.0, 2.0, 3.0, 4.0]
    old_style = sorted(data)[int(len(data) * 0.95)]  # = data[3] = 4.0（最大值）
    assert old_style == 4.0
    assert percentile(data, 0.95) == pytest.approx(3.85)
    assert percentile(data, 0.95) != old_style  # 新实现确实不同


def test_percentile_median_equals_median() -> None:
    for data in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [5.0], [1.0, 9.0]):
        assert percentile(data, 0.5) == pytest.approx(median(data))


def test_percentile_empty_raises() -> None:
    with pytest.raises(ValueError, match="至少一个样本"):
        percentile([], 0.95)


def test_percentile_rejects_out_of_range_q() -> None:
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 1.5)
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], -0.1)


def test_percentile_of_single_value() -> None:
    assert percentile([7.5], 0.0) == 7.5
    assert percentile([7.5], 1.0) == 7.5


def test_percentile_accepts_unsorted_input() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)


def test_quantile_summary_shape() -> None:
    summary = quantile_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["max"] == 4.0
    assert quantile_summary([])["p95"] is None


def test_percentile_anomalies_is_only_top_slice() -> None:
    """``percentile_anomalies`` 只是「取最高的那一小撮」，文档里写清楚了。"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    threshold, flagged = percentile_anomalies(data, q=0.8)
    assert threshold == pytest.approx(4.2)
    assert flagged == [4]  # 5.0 严格大于 4.2


# ======================================================================================
# MAD 离群检测
# ======================================================================================
def _mad_dataset() -> list[float]:
    """35 个 100 附近的小抖动 + 1 个明显离群点。"""
    base = [100.0 + (index % 7) * 0.5 for index in range(35)]  # 100.0 .. 103.0
    return base + [1000.0]


def test_median_and_mad_are_robust() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert median(data) == 3.0
    # median(|x-3|) = median([2,1,0,1,2]) = 1
    assert median_absolute_deviation(data) == 1.0
    # MAD 刻意不做 1.4826 缩放（缩放职责在 0.6745 系数上）
    assert median_absolute_deviation(data) != pytest.approx(1.4826)


def test_modified_zscore_formula() -> None:
    assert modified_zscore(10.0, 5.0, 2.0) == pytest.approx(DEFAULT_MAD_CONSTANT * 5.0 / 2.0)
    assert modified_zscore(5.0, 5.0, 0.0) == 0.0
    assert modified_zscore(6.0, 5.0, 0.0) == float("inf")
    assert modified_zscore(4.0, 5.0, 0.0) == float("-inf")


def test_detect_outliers_finds_the_spike() -> None:
    data = _mad_dataset()
    report = detect_outliers_by_mad(data, threshold=3.5, min_sample_size=30)
    assert isinstance(report, OutlierReport)
    assert report.sample_size == len(data)
    assert report.outlier_count == 1
    assert report.outliers[0].value == pytest.approx(1000.0)
    assert report.outliers[0].direction == "high"
    assert abs(report.outliers[0].modified_z) > 3.5
    assert report.method == "median+mad(modified z-score)"
    assert report.to_dict()["outlier_count"] == 1


def test_detect_outliers_returns_insufficient_sample() -> None:
    """缺陷 2 的另一半：样本不足必须明确返回「样本不足」，不能硬算。"""
    data = _mad_dataset()[:10]
    result = detect_outliers_by_mad(data, min_sample_size=30)
    assert isinstance(result, InsufficientSample)
    assert result.enough is False
    assert result.sample_size == 10
    assert result.required == 30
    assert "样本不足" in str(result)


def test_detect_outliers_boundary_at_min_sample_size() -> None:
    data = _mad_dataset()[:30]
    result = detect_outliers_by_mad(data, min_sample_size=30)
    assert isinstance(result, OutlierReport)
    # 降一级就变成样本不足
    assert isinstance(detect_outliers_by_mad(data[:29], min_sample_size=30), InsufficientSample)


def test_detect_outliers_no_false_positive_on_uniform_data() -> None:
    data = [100.0, 101.0, 99.0, 100.5, 99.5] * 8  # 40 个点，最大偏离很小
    report = detect_outliers_by_mad(data, threshold=3.5, min_sample_size=30)
    assert isinstance(report, OutlierReport)
    assert report.outlier_count == 0


def test_detect_outliers_handles_zero_mad_with_note() -> None:
    """MAD=0（超半数样本相同）时退化处理，并留下可读说明。"""
    data = [100.0] * 35 + [500.0, 1.0]
    report = detect_outliers_by_mad(data, min_sample_size=30)
    assert isinstance(report, OutlierReport)
    assert report.mad == 0.0
    assert report.outlier_count == 2
    assert "MAD 为 0" in report.note


def test_outlier_indices_can_be_offset() -> None:
    data = _mad_dataset()
    report = detect_outliers_by_mad(data, min_sample_size=30, index_offset=1000)
    assert isinstance(report, OutlierReport)
    assert report.outliers[0].index == 1000 + len(data) - 1


# ======================================================================================
# 成本计算
# ======================================================================================
def test_compute_cost_peak_uses_peak_rates() -> None:
    breakdown = compute_cost(
        "deepseek-flash",
        input_tokens=1_000_000,
        output_tokens=500_000,
        at=PEAK,
    )
    assert breakdown.tier == "peak"
    # 1M × 2 元/M = 2 元（缓存未命中），0.5M × 8 元/M = 4 元
    assert breakdown.cost_cache_miss_input == Decimal("2.0")
    assert breakdown.cost_output == Decimal("4.0")
    assert breakdown.total == Decimal("6.0")
    assert breakdown.unit == PRICE_UNIT


def test_compute_cost_off_peak_is_half() -> None:
    peak = compute_cost("deepseek-flash", input_tokens=1_000_000, output_tokens=500_000, at=PEAK)
    off_peak = compute_cost(
        "deepseek-flash", input_tokens=1_000_000, output_tokens=500_000, at=OFF_PEAK
    )
    assert off_peak.total == peak.total / 2
    assert off_peak.tier == "off_peak"


def test_compute_cost_three_decades_apart_from_old_price_table() -> None:
    """量级回归：旧版把 2 元/百万当成 0.001 元/1K，同一笔调用差 1000 倍。"""
    breakdown = compute_cost("deepseek-flash", input_tokens=1_000_000, at=PEAK)
    assert breakdown.total == Decimal("2.0")
    assert breakdown.total != Decimal("0.002")


def test_compute_cost_splits_cache_hit_and_miss() -> None:
    breakdown = compute_cost(
        "deepseek-v4-pro",
        input_tokens=1_000_000,
        cache_hit_tokens=750_000,
        cache_miss_tokens=250_000,
        output_tokens=100_000,
        at=PEAK,
    )
    assert breakdown.cache_hit_tokens == 750_000
    assert breakdown.cache_miss_tokens == 250_000
    # 0.75M × 0.30 = 0.225 元；0.25M × 9.0 = 2.25 元；0.1M × 27.0 = 2.7 元
    assert breakdown.cost_cache_hit_input == Decimal("0.225")
    assert breakdown.cost_cache_miss_input == Decimal("2.25")
    assert breakdown.cost_output == Decimal("2.7")
    assert breakdown.total == Decimal("5.175")


def test_compute_cost_cache_savings() -> None:
    breakdown = compute_cost(
        "deepseek-flash",
        input_tokens=1_000_000,
        cache_hit_tokens=1_000_000,
        at=PEAK,
    )
    # 全部命中：1M × 0.04 = 0.04 元；若按未命中则 2 元，节省 1.96 元
    assert breakdown.total == Decimal("0.04")
    assert breakdown.cache_savings == Decimal("1.96")


def test_compute_cost_infers_split_from_cache_hit_flag() -> None:
    flag_style = compute_cost("deepseek-flash", input_tokens=1_000, cache_hit=True, at=PEAK)
    explicit_style = compute_cost(
        "deepseek-flash", input_tokens=1_000, cache_hit_tokens=1_000, at=PEAK
    )
    assert flag_style.total == explicit_style.total
    assert flag_style.cache_hit_tokens == 1_000


def test_compute_cost_missing_cache_info_assumes_miss() -> None:
    """拿不到缓存信息时按未命中计价（偏保守，金额偏高而不是偏低）。"""
    breakdown = compute_cost("deepseek-flash", input_tokens=1_000, at=PEAK)
    assert breakdown.cache_miss_tokens == 1_000
    assert breakdown.cache_hit_tokens == 0
    assert "缓存命中" in breakdown.notes[0]


def test_compute_cost_notes_conflicting_split() -> None:
    breakdown = compute_cost(
        "deepseek-flash",
        input_tokens=1_000,
        cache_hit_tokens=100,
        cache_miss_tokens=500,  # 100+500 != 1000
        at=PEAK,
    )
    assert any("!= input_tokens" in note for note in breakdown.notes)


def test_compute_cost_marks_estimated() -> None:
    breakdown = compute_cost("deepseek-flash", input_tokens=100, at=PEAK, estimated=True)
    assert breakdown.estimated is True
    assert "估算" in breakdown.explanation()
    assert breakdown.to_dict()["estimated"] is True


def test_compute_cost_explanation_is_readable() -> None:
    text = compute_cost("deepseek-v4-pro", input_tokens=1_000_000, output_tokens=1_000_000, at=PEAK).explanation()
    assert "百万" in text or "unit" in text or "元" in text
    assert "27.0" in text


def test_compute_cost_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelError):
        compute_cost("llama-3", input_tokens=10, at=PEAK)


def test_compute_cost_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError):
        compute_cost("deepseek-flash", input_tokens=-1, at=PEAK)


def test_compute_cost_zero_tokens_is_free() -> None:
    breakdown = compute_cost("deepseek-flash", at=PEAK)
    assert breakdown.total == Decimal(0)
    assert breakdown.total_tokens == 0


def test_sum_cost_breakdowns() -> None:
    items = [
        compute_cost("deepseek-flash", input_tokens=1_000_000, at=PEAK),
        compute_cost("deepseek-flash", input_tokens=1_000_000, at=OFF_PEAK),
        compute_cost("deepseek-v4-pro", output_tokens=1_000_000, at=PEAK, estimated=True),
    ]
    summary = sum_cost_breakdowns(items)
    assert summary["count"] == 3
    assert summary["estimated_count"] == 1
    assert summary["total"] == Decimal("2.0") + Decimal("1.0") + Decimal("27.0")
    assert summary["by_model"]["deepseek-flash"] == Decimal("3.0")


# ======================================================================================
# 对账
# ======================================================================================
def test_reconcile_balanced() -> None:
    result = reconcile("10.00", "10.00")
    assert result.mismatch is False
    assert result.difference == Decimal("0.00")
    assert result.ratio == 0
    assert "✅" in result.summary()


def test_reconcile_detects_small_drift() -> None:
    result = reconcile("10.5", "10.0")
    assert result.difference == Decimal("0.5")
    assert result.ratio == Decimal("0.05")
    assert result.ratio_percent == Decimal("5.0000")
    assert result.mismatch is True  # 5% > 1%


def test_reconcile_within_tolerance() -> None:
    result = reconcile("10.05", "10.0", tolerance=Decimal("0.01"))
    assert result.ratio == Decimal("0.005")
    assert result.mismatch is False


def test_reconcile_negative_difference_is_abs_in_ratio() -> None:
    result = reconcile("9.0", "10.0")
    assert result.difference == Decimal("-1.0")
    assert result.abs_difference == Decimal("1.0")
    assert result.ratio == Decimal("0.1")
    assert result.mismatch is True


def test_reconcile_zero_bill_with_local_spend_is_mismatch() -> None:
    result = reconcile("1.0", "0")
    assert result.ratio == Decimal("Infinity")
    assert result.mismatch is True
    assert "账单为 0" in result.note


def test_reconcile_zero_and_zero_is_fine() -> None:
    result = reconcile("0", "0")
    assert result.mismatch is False
    assert result.note


def test_reconcile_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError):
        reconcile("-1", "10")
    with pytest.raises(ValueError):
        reconcile("1", "-10")
    with pytest.raises(ValueError):
        reconcile("1", "1", tolerance="-0.1")


def test_reconcile_accepts_decimal_float_and_str() -> None:
    for value in ("10.5", 10.5, Decimal("10.5")):
        assert reconcile(value, "10.0").difference == Decimal("0.5")


# ======================================================================================
# 分账 vs 总账
# ======================================================================================
def test_check_ledger_split_balanced() -> None:
    check = check_ledger_split("3.0", {"deepseek-flash": "2.0", "deepseek-v4-pro": "1.0"})
    assert check.balanced is True
    assert check.difference == 0
    assert "✅" in check.summary()


def test_check_ledger_split_detects_imbalance() -> None:
    """旧版分账来自 Hash、总账来自 List，两者永不保证相等；这里把它变成可断言的事实。"""
    check = check_ledger_split("3.0", {"deepseek-flash": "2.0", "deepseek-v4-pro": "0.5"})
    assert check.balanced is False
    assert check.difference == Decimal("0.5")


def test_default_thresholds_are_documented_values() -> None:
    assert DEFAULT_MAD_THRESHOLD == 3.5
    assert DEFAULT_MAD_CONSTANT == 0.6745
