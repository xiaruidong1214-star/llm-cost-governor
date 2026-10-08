"""价格表多生效区间测试。

**为什么要有这批用例**：早期实现每个模型只有一个 ``effective_date``，
于是官方一调价，**历史区间**就会被用新价重算——这正是本项目声称要解决的
"调价后历史被重新解释"问题的残留同类问题。改成"区间列表"之后，
1 月的账单必须永远按 1 月的价算。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from costgovernor.pricing import (
    MODEL_PRICING,
    PRICE_UNIT,
    ModelPricing,
    PeakWindow,
    PriceBand,
    PricingProfile,
    UnknownModelError,
    single_band_pricing,
)

# 一个用于测试的两区间模型：1 月价便宜，7 月起涨价一倍
_CHEAP = dict(
    input_cache_hit_peak=Decimal("0.01"),
    input_cache_hit_off_peak=Decimal("0.005"),
    input_cache_miss_peak=Decimal("1"),
    input_cache_miss_off_peak=Decimal("0.5"),
    output_peak=Decimal("4"),
    output_off_peak=Decimal("2"),
)
_EXPENSIVE = dict(
    input_cache_hit_peak=Decimal("0.02"),
    input_cache_hit_off_peak=Decimal("0.01"),
    input_cache_miss_peak=Decimal("2"),
    input_cache_miss_off_peak=Decimal("1"),
    output_peak=Decimal("8"),
    output_off_peak=Decimal("4"),
)


def _two_band_model() -> ModelPricing:
    return ModelPricing(
        model="two-band",
        bands=(
            PriceBand(effective_from="2026-01-01", until="2026-06-30", **_CHEAP),
            PriceBand(effective_from="2026-07-01", **_EXPENSIVE),
        ),
    )


@pytest.fixture
def profile() -> PricingProfile:
    return PricingProfile(
        models={"two-band": _two_band_model()},
        timezone=__import__("datetime").timezone.utc,
        peak_windows=(PeakWindow(__import__("datetime").time(9), __import__("datetime").time(12)),),
    )


# ---------------------------------------------------------------- 区间选择


def test_historical_date_uses_the_historical_band(profile) -> None:
    """核心用例：6 月的调用必须用便宜价，不能被 7 月的涨价影响。"""
    june = datetime(2026, 6, 15, 10, 0)
    assert profile.rate_for("two-band", "input", "cache_miss", june) == Decimal("1")


def test_new_date_uses_the_new_band(profile) -> None:
    july = datetime(2026, 7, 15, 10, 0)
    assert profile.rate_for("two-band", "input", "cache_miss", july) == Decimal("2")


def test_boundary_days_are_inclusive(profile) -> None:
    """区间是左闭右闭：6-30 仍属旧价，7-01 起用新价。"""
    assert profile.rate_for("two-band", "input", "cache_miss", datetime(2026, 6, 30, 10, 0)) == Decimal("1")
    assert profile.rate_for("two-band", "input", "cache_miss", datetime(2026, 7, 1, 10, 0)) == Decimal("2")


def test_band_for_returns_the_covering_band(profile) -> None:
    band = profile.band_for_time("two-band", datetime(2026, 3, 1, 10, 0))
    assert band.effective_from == "2026-01-01"
    assert band.until == "2026-06-30"


def test_date_before_first_band_raises(profile) -> None:
    """早于最早区间时宁可报错，也不要拿不属于那个时期的价格算账。"""
    from costgovernor.pricing import PricingNotEffectiveError

    with pytest.raises(PricingNotEffectiveError):
        profile.band_for_time("two-band", datetime(2025, 12, 31, 10, 0))


def test_unknown_model_still_raises(profile) -> None:
    with pytest.raises(UnknownModelError):
        profile.band_for_time("nope", datetime(2026, 7, 1, 10, 0))


# ---------------------------------------------------------------- 结构约束


def test_bands_must_be_sorted() -> None:
    with pytest.raises(ValueError, match="升序"):
        ModelPricing(
            model="bad",
            bands=(
                PriceBand(effective_from="2026-07-01", **_EXPENSIVE),
                PriceBand(effective_from="2026-01-01", until="2026-06-30", **_CHEAP),
            ),
        )


def test_empty_bands_rejected() -> None:
    with pytest.raises(ValueError, match="至少需要一个价格区间"):
        ModelPricing(model="empty", bands=())


def test_band_covers_boundaries() -> None:
    band = PriceBand(effective_from="2026-01-01", until="2026-06-30", **_CHEAP)
    assert band.covers("2026-01-01") is True
    assert band.covers("2026-06-30") is True
    assert band.covers("2025-12-31") is False
    assert band.covers("2026-07-01") is False


def test_open_ended_band_covers_everything_after() -> None:
    band = PriceBand(effective_from="2026-07-01", **_EXPENSIVE)
    assert band.covers("2099-12-31") is True
    assert band.covers("2026-06-30") is False


# ---------------------------------------------------------------- 兼容性


def test_effective_date_property_is_the_earliest(profile) -> None:
    assert profile.models["two-band"].effective_date == "2026-01-01"


def test_latest_is_the_last_band(profile) -> None:
    latest = profile.models["two-band"].latest
    assert latest.effective_from == "2026-07-01"
    assert latest.until is None


def test_single_band_helper_matches_direct_construction() -> None:
    via_helper = single_band_pricing(
        model="x",
        effective_date="2026-01-01",
        **_CHEAP,
    )
    via_direct = ModelPricing(model="x", bands=(PriceBand(effective_from="2026-01-01", **_CHEAP),))
    assert via_helper.bands == via_direct.bands
    assert via_helper.rate("output", "cache_miss", "peak") == via_direct.rate(
        "output", "cache_miss", "peak"
    )


def test_official_table_is_well_formed() -> None:
    """官方表本身必须满足结构约束（每个模型至少一个区间、按时间升序）。"""
    for name, pricing in MODEL_PRICING.items():
        assert pricing.bands, f"{name} 没有价格区间"
        starts = [b.effective_from for b in pricing.bands]
        assert starts == sorted(starts), f"{name} 的区间未按升序排列"
        # 相邻区间不应重叠
        for earlier, later in zip(pricing.bands, pricing.bands[1:], strict=False):
            if earlier.until is not None:
                assert earlier.until < later.effective_from, f"{name} 区间重叠"


def test_price_unit_is_per_million_tokens() -> None:
    """单位是「元 / 百万 tokens」——旧版写成 1K 导致总量级全错。

    这里同时断言两个可检验的事实：换算常量就是 1e6，且单位文本里有「百万」。
    """
    from costgovernor.pricing import TOKENS_PER_PRICE_UNIT

    assert TOKENS_PER_PRICE_UNIT == 1_000_000
    assert "百万" in PRICE_UNIT
