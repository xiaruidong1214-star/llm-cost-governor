"""价格表回归测试。

重点是锁住三件旧版做错的事：

1. 单位是**百万 tokens**（不是 1K）；
2. 高峰/空闲、缓存命中/未命中是**两档独立**的价；
3. 未知模型必须 **raise**，不能静默回退默认价。
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from costgovernor.pricing import (
    DEFAULT_PRICING,
    MODEL_PRICING,
    PRICE_CHECKED_AT,
    PRICE_SOURCE_URL,
    PRICE_UNIT,
    TOKENS_PER_PRICE_UNIT,
    PeakWindow,
    PricingNotEffectiveError,
    PricingProfile,
    UnknownModelError,
    default_known_models,
    is_peak_time,
    parse_peak_windows,
    price_for,
    rates_for,
    resolve_model_name,
)

# 2026-10-08 是周四
THURSDAY_10AM = datetime(2026, 10, 8, 10, 0, 0)
THURSDAY_9AM = datetime(2026, 10, 8, 9, 0, 0)
THURSDAY_1159 = datetime(2026, 10, 8, 11, 59, 59)
THURSDAY_NOON = datetime(2026, 10, 8, 12, 0, 0)
THURSDAY_2PM = datetime(2026, 10, 8, 14, 0, 0)
THURSDAY_6PM = datetime(2026, 10, 8, 18, 0, 0)
THURSDAY_8PM = datetime(2026, 10, 8, 20, 0, 0)
SATURDAY_10AM = datetime(2026, 10, 10, 10, 0, 0)
SUNDAY_10AM = datetime(2026, 10, 11, 10, 0, 0)


# ======================================================================================
# 单位
# ======================================================================================
def test_price_unit_is_per_million_tokens() -> None:
    """缺陷 1 的回归：单位必须是「百万 tokens」。

    旧版 ``MODEL_PRICING = {"deepseek-chat": {"input": 0.001, "output": 0.002}}``
    注释写「元/1K tokens」，而官方按百万 tokens 计费 —— 量级差 1000 倍。
    """
    assert PRICE_UNIT == "元 / 百万 tokens"
    assert TOKENS_PER_PRICE_UNIT == 1_000_000
    assert "百万" in PRICE_UNIT
    assert "1K" not in PRICE_UNIT
    assert "千" not in PRICE_UNIT


def test_unit_is_carried_into_every_price_lookup() -> None:
    """单价必须是「百万 tokens」口径下的数字，而不是 1K 口径。"""
    # 官方：deepseek-flash 高峰缓存未命中输入 2 元/百万 tokens
    assert price_for("deepseek-flash", THURSDAY_10AM, kind="input") == Decimal("2.0")
    # 若误用 1K 口径，这里会是 0.002；明确断言绝非该值
    assert price_for("deepseek-flash", THURSDAY_10AM, kind="input") != Decimal("0.002")


def test_one_million_tokens_costs_one_rate() -> None:
    """100 万 tokens 的输入成本应当**恰好等于**标价本身（这是单位正确的直接推论）。"""
    rate = price_for("deepseek-v4-pro", THURSDAY_10AM, kind="output")
    cost = Decimal(TOKENS_PER_PRICE_UNIT) * rate / TOKENS_PER_PRICE_UNIT
    assert cost == rate == Decimal("27.0")


# ======================================================================================
# 官方价格数值
# ======================================================================================
@pytest.mark.parametrize(
    ("model", "tier", "kind", "cache_tier", "expected"),
    [
        # deepseek-flash
        ("deepseek-flash", "peak", "input", "cache_miss", "2.0"),
        ("deepseek-flash", "off_peak", "input", "cache_miss", "1.0"),
        ("deepseek-flash", "peak", "output", "cache_miss", "8.0"),
        ("deepseek-flash", "off_peak", "output", "cache_miss", "4.0"),
        ("deepseek-flash", "peak", "input", "cache_hit", "0.04"),
        ("deepseek-flash", "off_peak", "input", "cache_hit", "0.02"),
        # deepseek-v4-pro
        ("deepseek-v4-pro", "peak", "input", "cache_miss", "9.0"),
        ("deepseek-v4-pro", "off_peak", "input", "cache_miss", "4.5"),
        ("deepseek-v4-pro", "peak", "output", "cache_miss", "27.0"),
        ("deepseek-v4-pro", "off_peak", "output", "cache_miss", "13.5"),
        ("deepseek-v4-pro", "peak", "input", "cache_hit", "0.30"),
        ("deepseek-v4-pro", "off_peak", "input", "cache_hit", "0.15"),
    ],
)
def test_official_rates(
    model: str, tier: str, kind: str, cache_tier: str, expected: str
) -> None:
    """与 https://api-docs.deepseek.com/zh-cn/quick_start/pricing/ 逐项核对。"""
    pricing = MODEL_PRICING[model]
    assert pricing.rate(kind, cache_tier, tier) == Decimal(expected)


def test_off_peak_is_half_of_peak() -> None:
    """官方明确说明：空闲时段价格是高峰时段价格的一半。"""
    for model, pricing in MODEL_PRICING.items():
        for kind, cache_tier in (("input", "cache_miss"), ("input", "cache_hit"), ("output", "cache_miss")):
            peak = pricing.rate(kind, cache_tier, "peak")
            off_peak = pricing.rate(kind, cache_tier, "off_peak")
            assert peak == off_peak * 2, f"{model} {kind} {cache_tier} 不满足高峰=空闲×2"


def test_cache_hit_is_much_cheaper_than_cache_miss() -> None:
    """缺陷 1 的另一半：缓存命中价远低于未命中价（官方低至 1/25~1/50）。"""
    for model, pricing in MODEL_PRICING.items():
        for tier in ("peak", "off_peak"):
            hit = pricing.rate("input", "cache_hit", tier)
            miss = pricing.rate("input", "cache_miss", tier)
            assert hit < miss
            assert miss / hit >= 25, f"{model} {tier} 的缓存折扣不足 1/25"
    # deepseek-flash 空闲：1 / 0.02 = 50
    assert MODEL_PRICING["deepseek-flash"].rate("input", "cache_miss", "off_peak") / MODEL_PRICING[
        "deepseek-flash"
    ].rate("input", "cache_hit", "off_peak") == Decimal("50")


# ======================================================================================
# 时段判定
# ======================================================================================
@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (THURSDAY_9AM, True),
        (THURSDAY_10AM, True),
        (THURSDAY_1159, True),
        (THURSDAY_NOON, False),  # 左闭右开：12:00:00 整属于空闲
        (THURSDAY_2PM, True),
        (THURSDAY_6PM, False),  # 18:00:00 整属于空闲
        (THURSDAY_8PM, False),
        (SATURDAY_10AM, False),
        (SUNDAY_10AM, False),
    ],
)
def test_peak_window_boundaries(moment: datetime, expected: bool) -> None:
    """高峰时段：周一至周五 9:00-12:00、14:00-18:00（左闭右开）。"""
    assert is_peak_time(moment) is expected
    assert DEFAULT_PRICING.tier_for(moment) == ("peak" if expected else "off_peak")


def test_peak_and_off_peak_prices_differ_in_practice() -> None:
    """同一模型同一 token 种类，高峰与空闲的实际取价必须不同。"""
    assert price_for("deepseek-flash", THURSDAY_10AM) == Decimal("2.0")
    assert price_for("deepseek-flash", THURSDAY_8PM) == Decimal("1.0")


def test_holidays_are_off_peak() -> None:
    """节假日全天按空闲时段计价（官方规则）。默认不内置日历，需要显式配置。"""
    profile = PricingProfile(
        models=MODEL_PRICING,
        timezone=DEFAULT_PRICING.timezone,
        peak_windows=DEFAULT_PRICING.peak_windows,
        holidays=frozenset({"2026-10-08"}),
    )
    assert profile.is_peak(THURSDAY_10AM) is False
    assert DEFAULT_PRICING.is_peak(THURSDAY_10AM) is True  # 默认无日历，仍视为高峰


def test_naive_datetime_is_treated_as_beijing_time() -> None:
    """naive 时间按北京时间解释；带时区的时间会被换算过去。"""
    from datetime import timedelta

    # 北京时间周四 10:00 == UTC 02:00
    utc_moment = datetime(2026, 10, 8, 2, 0, 0, tzinfo=UTC)
    assert DEFAULT_PRICING.is_peak(utc_moment) is True
    # 北京时间周四 20:00 == UTC 12:00
    utc_night = datetime(2026, 10, 8, 12, 0, 0, tzinfo=UTC)
    assert DEFAULT_PRICING.is_peak(utc_night) is False
    # UTC+8 固定偏移的换算结果与 Asia/Shanghai 一致（中国无夏令时）
    assert DEFAULT_PRICING.localize(utc_moment).utcoffset() == timedelta(hours=8)


def test_parse_peak_windows_rejects_bad_input() -> None:
    assert parse_peak_windows("09:00-12:00")[0] == PeakWindow(
        start=datetime(2026, 1, 1, 9).time(), end=datetime(2026, 1, 1, 12).time()
    )
    with pytest.raises(ValueError):
        parse_peak_windows("9:00-12:00")  # 必须是两位小时
    with pytest.raises(ValueError):
        parse_peak_windows("12:00-09:00")  # 起点晚于终点
    with pytest.raises(ValueError):
        parse_peak_windows("")
    with pytest.raises(ValueError):
        parse_peak_windows("09:00-25:00")


# ======================================================================================
# 未知模型 / 生效日期
# ======================================================================================
def test_unknown_model_raises_instead_of_falling_back() -> None:
    """缺陷 1 的第三面：未知模型必须显式报错。

    旧版静默走默认价，把「配置漏写」掩盖成「成本偏低」。
    """
    with pytest.raises(UnknownModelError) as excinfo:
        price_for("gpt-4o-mini", THURSDAY_10AM)
    assert "gpt-4o-mini" in str(excinfo.value)
    assert "deepseek-flash" in str(excinfo.value)  # 错误信息里给出已知模型
    assert "pricing.py" in str(excinfo.value)


def test_unknown_model_also_raises_via_profile() -> None:
    with pytest.raises(UnknownModelError):
        DEFAULT_PRICING.pricing_for("no-such-model", THURSDAY_10AM)


def test_pricing_not_effective_raises() -> None:
    """生效日期之前的调用要报错，而不是拿新价格去解释历史数据。"""
    from dataclasses import replace

    future = replace(
        MODEL_PRICING["deepseek-flash"],
        model="future-model",
        effective_date="2026-12-01",
    )
    profile = PricingProfile(
        models={"future-model": future},
        timezone=DEFAULT_PRICING.timezone,
        peak_windows=DEFAULT_PRICING.peak_windows,
    )
    with pytest.raises(PricingNotEffectiveError) as excinfo:
        profile.pricing_for("future-model", THURSDAY_10AM)
    assert "2026-12-01" in str(excinfo.value)
    # 生效之后正常
    assert profile.pricing_for("future-model", datetime(2027, 1, 4, 10)).model == "future-model"


def test_every_model_carries_effective_date_and_provenance() -> None:
    """价格表必须带生效日期与来源信息，否则一次调价就会让历史数据被重新解释。"""
    for model, pricing in MODEL_PRICING.items():
        assert pricing.effective_date, model
        assert pricing.checked_at == PRICE_CHECKED_AT
        assert pricing.source_url == PRICE_SOURCE_URL
        assert pricing.currency == "CNY"
        datetime.fromisoformat(pricing.effective_date)  # 格式必须合法


# ======================================================================================
# 别名与便捷函数
# ======================================================================================
def test_legacy_model_name_is_resolved_but_priced_like_flash() -> None:
    assert resolve_model_name("deepseek-chat") == "deepseek-flash"
    assert price_for("deepseek-chat", THURSDAY_10AM, kind="output") == Decimal("8.0")
    assert "deepseek-flash" in default_known_models()


def test_default_profile_has_both_official_models() -> None:
    known = default_known_models()
    assert "deepseek-flash" in known
    assert "deepseek-v4-pro" in known


def test_rates_for_returns_three_rates() -> None:
    rates = rates_for("deepseek-v4-pro", THURSDAY_8PM)
    assert rates == {
        "input_cache_hit": Decimal("0.15"),
        "input_cache_miss": Decimal("4.5"),
        "output": Decimal("13.5"),
    }


def test_output_tokens_have_no_cache_tier() -> None:
    """输出 token 不存在「缓存命中价」；显式报错好过静默按某个价算。"""
    with pytest.raises(ValueError):
        MODEL_PRICING["deepseek-flash"].rate("output", "cache_hit", "peak")
    with pytest.raises(ValueError):
        MODEL_PRICING["deepseek-flash"].rate("input", "cache_something", "peak")
    with pytest.raises(ValueError):
        MODEL_PRICING["deepseek-flash"].rate("total", "cache_miss", "peak")
