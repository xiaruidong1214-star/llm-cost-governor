"""价格表与时段判定。

**单位口径**：本模块所有单价均为「元 / 百万 tokens」。
旧版把 ``0.001`` 注释成「元/1K tokens」，但官方实际按**百万 tokens** 计费，
且区分**高峰/空闲时段**与**缓存命中/未命中**，导致成本量级与口径全错。
这里把口径固化成 :data:`PRICE_UNIT`，并用测试锁住。

官方价格来源：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
核对日期：2026-10-08（价格会随官方调整而变化，参见 README「已知限制」）。

官方原文（节选）：
    「下表所列模型价格以“百万 tokens”为单位。」
    「空闲时段价格为高峰时段价格的一半。北京时间周一至周五（不含中国法定节假日）
      9:00 - 12:00、14:00 - 18:00 为高峰时段；其余时段，包括周末及中国法定节假日
      全天均为空闲时段。」
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal
from typing import Literal

try:  # pragma: no cover - 平台差异
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

__all__ = [
    "PRICE_UNIT",
    "PRICE_SOURCE_URL",
    "PRICE_CHECKED_AT",
    "TOKENS_PER_PRICE_UNIT",
    "MODEL_ALIASES",
    "default_pricing_profile",
    "default_known_models",
    "resolve_model_name",
    "CacheTier",
    "MODEL_PRICING",
    "PricingProfile",
    "PeakWindow",
    "UnknownModelError",
    "PricingNotEffectiveError",
    "parse_peak_windows",
    "is_peak_time",
    "price_for",
    "rates_for",
    "unit_price",
    "DEFAULT_PRICING",
]

#: 价格单位：所有 ``ModelPricing`` 中的数字都是「元 / 百万 tokens」。
#: 这不是注释里的说明，而是可被测试断言的常量：旧版正是把单位写错才导致总成本量级全错。
PRICE_UNIT = "元 / 百万 tokens"

#: 每百万 tokens 的换算因子。计算成本时必须除以它。
TOKENS_PER_PRICE_UNIT = 1_000_000

PRICE_SOURCE_URL = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
PRICE_CHECKED_AT = "2026-10-08"

#: 缓存命中/未命中两档
CacheTier = Literal["cache_hit", "cache_miss"]

_DAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

_WINDOW_RE = re.compile(r"^\s*(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})\s*$")


class UnknownModelError(KeyError):
    """未知模型。

    旧版对未知模型静默回退到默认价，这会把「配置漏写」掩盖成「成本偏低」。
    这里显式 raise，让问题在第一次调用时就暴露出来。
    """

    def __init__(self, model: str, known: Iterable[str] | None = None) -> None:
        known_list = sorted(known) if known is not None else sorted(MODEL_PRICING)
        super().__init__(model)
        self.model = model
        self.known = known_list
        self.message = (
            f"未收录的模型 {model!r}；已知模型：{known_list}。"
            f"请先在 pricing.py 中按 {PRICE_SOURCE_URL} 补充价格，"
            "而不是依赖默认价（静默回退会掩盖口径错误）。"
        )

    def __str__(self) -> str:  # pragma: no cover - 仅用于报错可读性
        return self.message


class PricingNotEffectiveError(ValueError):
    """在价格生效日期之前调用计费。

    旧版没有 ``effective_date`` 概念，价格一改，历史数据立刻被重新解释成错的。
    """

    def __init__(self, model: str, at: datetime, effective_date: str) -> None:
        super().__init__(model, at, effective_date)
        self.model = model
        self.at = at
        self.effective_date = effective_date
        self.message = (
            f"模型 {model!r} 的价格自 {effective_date} 起生效，"
            f"但计费时刻为 {at.isoformat()}。请更换价格档或修正计费时刻。"
        )

    def __str__(self) -> str:  # pragma: no cover
        return self.message


@dataclass(frozen=True, slots=True)
class PeakWindow:
    """高峰时段，半开区间 ``[start, end)``。

    官方写的是「9:00 - 12:00」，边界语义官方未明确（是否含 12:00 本身）。
    本库选择**左闭右开**：12:00:00 整已属空闲。该选择被测试锁定，避免实现漂移。
    """

    start: time
    end: time

    def contains(self, moment: time) -> bool:
        """判断某个「当天时刻」是否落在窗口内（左闭右开）。"""
        return self.start <= moment < self.end

    def __str__(self) -> str:
        return f"{self.start.strftime('%H:%M')}-{self.end.strftime('%H:%M')}"


def parse_peak_windows(spec: str | Sequence[str]) -> tuple[PeakWindow, ...]:
    """解析 ``"09:00-12:00,14:00-18:00"`` 为 :class:`PeakWindow` 元组。

    非法输入抛 ``ValueError``，不做静默兜底。
    """
    if isinstance(spec, str):
        raw_items = [item for item in spec.split(",") if item.strip()]
    else:
        raw_items = []
        for item in spec:
            raw_items.extend(part for part in str(item).split(",") if part.strip())

    windows: list[PeakWindow] = []
    for raw in raw_items:
        match = _WINDOW_RE.match(raw)
        if match is None:
            raise ValueError(f"无法解析的高峰时段 {raw!r}，期望格式 'HH:MM-HH:MM'")
        start_h, start_m, end_h, end_m = (int(group) for group in match.groups())
        if start_h > 23 or end_h > 23 or start_m > 59 or end_m > 59:
            raise ValueError(f"非法时间 {raw!r}")
        window = PeakWindow(time(hour=start_h, minute=start_m), time(hour=end_h, minute=end_m))
        if window.start >= window.end:
            raise ValueError(f"高峰时段 {raw!r} 的起点不早于终点")
        windows.append(window)
    if not windows:
        raise ValueError("高峰时段不能为空")
    return tuple(windows)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """单个模型的价格档。

    字段命名规则：``<token 种类>_<缓存档>_<时段>``，单位见 :data:`PRICE_UNIT`。
    token 种类只有两种：``input`` / ``output``；
    缓存档两种：``cache_hit`` / ``cache_miss``（命中价可能是未命中的 1/25）；
    时段两种：``peak`` / ``off_peak``。
    """

    model: str
    effective_date: str
    input_cache_hit_peak: Decimal
    input_cache_hit_off_peak: Decimal
    input_cache_miss_peak: Decimal
    input_cache_miss_off_peak: Decimal
    output_peak: Decimal
    output_off_peak: Decimal
    currency: str = "CNY"
    source_url: str = PRICE_SOURCE_URL
    checked_at: str = PRICE_CHECKED_AT

    # -------- 查询 --------
    def rate(self, kind: str, cache_tier: str, tier: str) -> Decimal:
        """返回单价（元 / 百万 tokens）。"""
        if kind == "input":
            if cache_tier == "cache_hit":
                return self.input_cache_hit_peak if tier == "peak" else self.input_cache_hit_off_peak
            if cache_tier == "cache_miss":
                return (
                    self.input_cache_miss_peak if tier == "peak" else self.input_cache_miss_off_peak
                )
            raise ValueError(f"未知的缓存档 {cache_tier!r}，期望 'cache_hit' 或 'cache_miss'")
        if kind == "output":
            if cache_tier != "cache_miss":
                # 输出 token 不存在缓存命中价；显式报错好过静默按某个价算。
                raise ValueError(f"输出 token 不支持缓存档 {cache_tier!r}")
            return self.output_peak if tier == "peak" else self.output_off_peak
        raise ValueError(f"未知的 token 种类 {kind!r}，期望 'input' 或 'output'")

    def as_flat_dict(self, tier: str) -> dict[str, str]:
        """给 CLI/报表用的扁平展示字典。"""
        return {
            "model": self.model,
            "effective_date": self.effective_date,
            "tier": tier,
            "cache_hit_input": str(self.rate("input", "cache_hit", tier)),
            "cache_miss_input": str(self.rate("input", "cache_miss", tier)),
            "output": str(self.rate("output", "cache_miss", tier)),
            "unit": PRICE_UNIT,
        }


@dataclass(frozen=True, slots=True)
class PricingProfile:
    """一套价格表 + 时段规则。

    把「价格表」和「时段规则」装在一起，测试里可以构造任意 profile，
    不必污染全局的 :data:`MODEL_PRICING`。
    """

    models: Mapping[str, ModelPricing]
    timezone: tzinfo
    peak_windows: tuple[PeakWindow, ...]
    holidays: frozenset[str] = frozenset()
    weekday_numbers: tuple[int, ...] = (0, 1, 2, 3, 4)  # 周一=0 ... 周五=4

    # -------- 时段判定 --------
    def localize(self, at: datetime) -> datetime:
        """把 ``at`` 转换到本 profile 的时区。

        naive 时间按「已经是本地时间」处理（很多测试和日志都这么写），
        aware 时间则做真正的时区换算。
        """
        if at.tzinfo is None:
            return at.replace(tzinfo=self.timezone)
        return at.astimezone(self.timezone)

    def is_peak(self, at: datetime) -> bool:
        """是否为高峰时段。

        规则：本地时区的**周一至周五**、**非法定节假日**、且落在任一高峰窗口内。
        """
        local = self.localize(at)
        if local.isoweekday() - 1 not in self.weekday_numbers:  # isoweekday: 周一=1
            return False
        if local.strftime("%Y-%m-%d") in self.holidays:
            return False
        moment = local.time()
        return any(window.contains(moment) for window in self.peak_windows)

    def tier_for(self, at: datetime) -> str:
        """返回 ``"peak"`` 或 ``"off_peak"``。"""
        return "peak" if self.is_peak(at) else "off_peak"

    def describe_tier(self, at: datetime) -> str:
        """人类可读的时段说明，用于报表与日志。"""
        local = self.localize(at)
        tier = self.tier_for(at)
        label = "高峰" if tier == "peak" else "空闲"
        day = _DAY_NAMES[local.isoweekday() - 1]
        windows = "、".join(str(window) for window in self.peak_windows) or "（未配置）"
        return f"{label}（{local.strftime('%Y-%m-%d %H:%M:%S')} {day}，高峰窗口 {windows}）"

    # -------- 价格查询 --------
    def pricing_for(self, model: str, at: datetime) -> ModelPricing:
        """取某模型在某时刻适用的价格档，并校验生效日期。"""
        try:
            pricing = self.models[model]
        except KeyError:
            raise UnknownModelError(model, self.models) from None

        local = self.localize(at)
        if local.strftime("%Y-%m-%d") < pricing.effective_date:
            raise PricingNotEffectiveError(model, local, pricing.effective_date)
        return pricing

    def rate_for(self, model: str, kind: str, cache_tier: str, at: datetime) -> Decimal:
        """某个模型、某个 token 种类/缓存档、某个时刻的单价（元 / 百万 tokens）。"""
        return self.pricing_for(model, at).rate(kind, cache_tier, self.tier_for(at))

    def known_models(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))


def default_pricing_profile() -> PricingProfile:
    """默认 profile：官方价格表 + 北京时间 + 官方高峰窗口，无内置节假日日历。"""
    return PricingProfile(
        models=MODEL_PRICING,
        timezone=_beijing_tzinfo(),
        peak_windows=parse_peak_windows("09:00-12:00,14:00-18:00"),
        holidays=frozenset(),
    )


def _beijing_tzinfo() -> tzinfo:
    """返回 Asia/Shanghai 时区对象；没有 tzdata 时退化为固定 UTC+8。

    中国自 1991 年起不再使用夏令时，UTC+8 偏移是恒定的，
    因此退化实现不会影响高峰时段判定的正确性。
    """
    if ZoneInfo is not None:
        try:
            return ZoneInfo("Asia/Shanghai")
        except Exception:  # noqa: BLE001 - 缺少 tzdata 时退化，不掩盖其他分支
            pass
    return timezone(timedelta(hours=8))


# --------------------------------------------------------------------------------------
# 官方价格表（元 / 百万 tokens），核对时间 2026-10-08
# --------------------------------------------------------------------------------------
MODEL_PRICING: dict[str, ModelPricing] = {
    # deepseek-flash：输入未命中 空闲 1 元 / 高峰 2 元；输出 空闲 4 元 / 高峰 8 元；
    # 缓存命中 空闲 0.02 元 / 高峰 0.04 元（恰好是未命中价的 1/50）。
    "deepseek-flash": ModelPricing(
        model="deepseek-flash",
        effective_date="2026-01-01",
        input_cache_hit_off_peak=Decimal("0.02"),
        input_cache_hit_peak=Decimal("0.04"),
        input_cache_miss_off_peak=Decimal("1.0"),
        input_cache_miss_peak=Decimal("2.0"),
        output_off_peak=Decimal("4.0"),
        output_peak=Decimal("8.0"),
    ),
    # deepseek-v4-pro：输入未命中 空闲 4.5 元 / 高峰 9.0 元；输出 空闲 13.5 元 / 高峰 27.0 元；
    # 缓存命中 空闲 0.15 元 / 高峰 0.30 元（未命中价的 1/30）。
    "deepseek-v4-pro": ModelPricing(
        model="deepseek-v4-pro",
        effective_date="2026-01-01",
        input_cache_hit_off_peak=Decimal("0.15"),
        input_cache_hit_peak=Decimal("0.30"),
        input_cache_miss_off_peak=Decimal("4.5"),
        input_cache_miss_peak=Decimal("9.0"),
        output_off_peak=Decimal("13.5"),
        output_peak=Decimal("27.0"),
    ),
    # 旧模型名仍可调用，按 Flash 价格计费（官方脚注 1）。
    "deepseek-v4-flash": ModelPricing(
        model="deepseek-v4-flash",
        effective_date="2026-01-01",
        input_cache_hit_off_peak=Decimal("0.02"),
        input_cache_hit_peak=Decimal("0.04"),
        input_cache_miss_off_peak=Decimal("1.0"),
        input_cache_miss_peak=Decimal("2.0"),
        output_off_peak=Decimal("4.0"),
        output_peak=Decimal("8.0"),
    ),
}

#: 别名 → 官方模型名。旧模型名会被归一到新名字，避免报表里出现同名不同价的重复项。
MODEL_ALIASES: dict[str, str] = {
    "deepseek-v4-flash": "deepseek-v4-flash",  # 自身保留，按 Flash 计价
    "deepseek-chat": "deepseek-flash",  # 旧版价格表用的名字
}

#: 默认 profile（模块级单例）。需要自定义时用 :func:`default_pricing_profile` 构造新的。
DEFAULT_PRICING: PricingProfile = default_pricing_profile()


def default_known_models() -> tuple[str, ...]:
    """已收录的模型名（不含别名）。"""
    return DEFAULT_PRICING.known_models()


def resolve_model_name(model: str, profile: PricingProfile | None = None) -> str:
    """把别名归一成官方模型名；未知模型原样返回（交由 :func:`price_for` 报错）。"""
    active = profile or DEFAULT_PRICING
    if model in active.models:
        return model
    return MODEL_ALIASES.get(model, model)


# --------------------------------------------------------------------------------------
# 模块级便捷函数
# --------------------------------------------------------------------------------------
def is_peak_time(at: datetime, profile: PricingProfile | None = None) -> bool:
    """是否为高峰时段（默认 profile）。"""
    return (profile or DEFAULT_PRICING).is_peak(at)


def rates_for(
    model: str,
    at: datetime,
    *,
    profile: PricingProfile | None = None,
) -> dict[str, Decimal]:
    """返回该模型在该时刻的四个单价（元 / 百万 tokens）。"""
    active = profile or DEFAULT_PRICING
    resolved = resolve_model_name(model, active)
    return {
        "input_cache_hit": active.rate_for(resolved, "input", "cache_hit", at),
        "input_cache_miss": active.rate_for(resolved, "input", "cache_miss", at),
        "output": active.rate_for(resolved, "output", "cache_miss", at),
    }


def price_for(
    model: str,
    at: datetime,
    *,
    kind: str = "input",
    cache_tier: str = "cache_miss",
    profile: PricingProfile | None = None,
) -> Decimal:
    """返回单价（元 / 百万 tokens）。

    :param model: 模型名（支持别名）。
    :param at: 计费时刻；naive 视为本地时间（默认北京时间）。
    :param kind: ``"input"`` 或 ``"output"``。
    :param cache_tier: ``"cache_hit"`` 或 ``"cache_miss"``。
    :raises UnknownModelError: 模型未收录。
    :raises PricingNotEffectiveError: 计费时刻早于价格生效日期。
    """
    active = profile or DEFAULT_PRICING
    resolved = resolve_model_name(model, active)
    return active.rate_for(resolved, kind, cache_tier, at)


def unit_price(model: str, at: datetime, **kwargs: object) -> Decimal:
    """``price_for`` 的别名，语义更贴近「单价」这个词。"""
    return price_for(model, at, **kwargs)  # type: ignore[arg-type]
