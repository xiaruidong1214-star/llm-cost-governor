"""纯函数算术层：成本计算、分位数、MAD 离群检测、对账。

本模块**不导入 redis、不读写任何存储**，所有函数都可以用普通断言测试。
这是刻意的设计：成本与统计的错误最容易由纯函数测试抓出来，
旧版把分位数写错（``idx = int(len(costs) * 0.95)`` 落在末位 → 异常集合恒为空）
就是因为算术和 IO 搅在一起、没法单独验证。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from .pricing import (
    DEFAULT_PRICING,
    PRICE_UNIT,
    TOKENS_PER_PRICE_UNIT,
    PricingProfile,
    resolve_model_name,
)

__all__ = [
    "TOKENS_PER_PRICE_UNIT",
    "PRICE_UNIT",
    "CostBreakdown",
    "Outlier",
    "OutlierReport",
    "InsufficientSample",
    "LedgerSplitCheck",
    "ReconcileResult",
    "DEFAULT_MAD_CONSTANT",
    "DEFAULT_MAD_THRESHOLD",
    "percentile",
    "quantile_summary",
    "median",
    "median_absolute_deviation",
    "modified_zscore",
    "detect_outliers_by_mad",
    "percentile_anomalies",
    "compute_cost",
    "sum_cost_breakdowns",
    "reconcile",
    "check_ledger_split",
]

#: modified z-score 的 0.6745 系数（等价于 0.75 分位数在正态分布下的取值）。
DEFAULT_MAD_CONSTANT = 0.6745
#: Iglewicz & Hoaglin 建议的离群阈值。
DEFAULT_MAD_THRESHOLD = 3.5

_HUNDRED = Decimal(100)


# ======================================================================================
# 分位数
# ======================================================================================
def percentile(values: Sequence[float], q: float) -> float:
    """线性插值分位数（与 ``numpy.quantile(..., method="linear")`` 一致）。

    实现即 numpy/``statistics`` 默认的 ``linear``（旧称 ``type 7``）方法：

    * ``h = (n - 1) * q``
    * ``lo = floor(h)``，``hi = ceil(h)``
    * 结果 ``= x[lo] + (h - lo) * (x[hi] - x[lo])``（``x`` 为升序）

    与旧版的区别：旧版写的是 ``costs[int(len(costs) * q)]``，
    当 ``q=0.95`` 且 ``n`` 不大时下标直接落到末位，取到的就是**最大值**，
    于是 ``[r for r in recs if r["cost"] > p95]`` 在数学上恒为空集，
    异常检测等于没做；而且那个 ``int(n*q)`` 取法本质上只是「按固定比例取 top 5%」，
    并不是分位数。

    :param values: 样本序列（无需预先排序，本函数自己排）。
    :param q: 分位点，``0 <= q <= 1``。
    :raises ValueError: ``values`` 为空，或 ``q`` 超出 ``[0, 1]``。
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"分位点 q 必须在 [0, 1] 内，收到 {q!r}")

    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        raise ValueError("percentile() 需要至少一个样本，收到空序列")

    if count == 1:
        return ordered[0]

    position = (count - 1) * q
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (position - lower_index) * (upper_value - lower_value)


def quantile_summary(
    values: Sequence[float],
    quantiles: Iterable[float] = (0.5, 0.9, 0.95, 0.99),
) -> dict[str, Any]:
    """一次性输出多个分位数，供报表使用。"""
    ordered = sorted(float(value) for value in values)
    summary: dict[str, Any] = {"count": len(ordered), "total": math.fsum(ordered) if ordered else 0.0}
    if ordered:
        summary["min"] = ordered[0]
        summary["max"] = ordered[-1]
        summary["mean"] = statistics.fmean(ordered)
        summary["median"] = percentile(ordered, 0.5)
    for q in quantiles:
        summary[f"p{q * 100:g}"] = percentile(ordered, q) if ordered else None
    return summary


def median(values: Sequence[float]) -> float:
    """中位数（偶数个样本时取中间两个的均值）。

    :raises ValueError: 空序列。
    """
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("median() 需要至少一个样本，收到空序列")
    return statistics.median(ordered)


# ======================================================================================
# MAD 离群检测
# ======================================================================================
@dataclass(frozen=True, slots=True)
class Outlier:
    """一条离群样本。"""

    index: int
    value: float
    modified_z: float
    direction: str  # "high" | "low"


@dataclass(frozen=True, slots=True)
class InsufficientSample:
    """样本不足。

    旧版无论样本多小都会硬算一个「异常」出来；这里显式返回一个可判定的结果对象，
    让上层能区分「确实没有异常」和「根本没法判断」。
    """

    sample_size: int
    required: int
    reason: str

    @property
    def enough(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"样本不足：{self.sample_size} < {self.required}（{self.reason}）"


@dataclass(frozen=True, slots=True)
class OutlierReport:
    """MAD 离群检测结果。"""

    sample_size: int
    median: float
    mad: float
    threshold: float
    method: str
    outliers: tuple[Outlier, ...] = ()
    note: str = ""

    @property
    def enough(self) -> bool:
        return True

    @property
    def outlier_count(self) -> int:
        return len(self.outliers)

    @property
    def outlier_indices(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.outliers)

    def to_dict(self) -> dict[str, Any]:
        """可 JSON 序列化的结构。"""
        return {
            "sample_size": self.sample_size,
            "median": self.median,
            "mad": self.mad,
            "threshold": self.threshold,
            "method": self.method,
            "outlier_count": self.outlier_count,
            "outliers": [
                {
                    "index": item.index,
                    "value": item.value,
                    "modified_z": item.modified_z,
                    "direction": item.direction,
                }
                for item in self.outliers
            ],
            "note": self.note,
        }


def median_absolute_deviation(values: Sequence[float]) -> float:
    """MAD = median(|x - median(x)|)。

    刻意**不做** ``* 1.4826`` 的正态一致性缩放：本库使用 modified z-score
    ``0.6745 * (x - median) / MAD``，0.6745 已经承担了缩放职责，
    再乘一次 1.4826 会把阈值整体缩小到约 1/2.2，导致误报。

    :raises ValueError: 空序列。
    """
    central = median(values)
    return median([abs(float(value) - central) for value in values])


def modified_zscore(value: float, central: float, mad: float, constant: float = DEFAULT_MAD_CONSTANT) -> float:
    """``0.6745 * (x - median) / MAD``。

    ``MAD == 0`` 时（超过一半样本完全相同）返回 ``inf``/``0``：
    偏离中位数的点视为无穷偏离，等于中位数的点视为 0。
    """
    if mad == 0:
        if value == central:
            return 0.0
        return math.inf if value > central else -math.inf
    return constant * (value - central) / mad


def detect_outliers_by_mad(
    values: Sequence[float],
    *,
    threshold: float = DEFAULT_MAD_THRESHOLD,
    min_sample_size: int = 30,
    constant: float = DEFAULT_MAD_CONSTANT,
    index_offset: int = 0,
) -> OutlierReport | InsufficientSample:
    """基于「中位数 + MAD」的稳健离群检测。

    * 样本量 ``< min_sample_size`` 时返回 :class:`InsufficientSample`（不硬算）；
    * 否则返回 :class:`OutlierReport`，其中 ``outliers`` 按 ``|modified_z|`` 降序。
    * ``MAD == 0`` 且存在偏离中位数的样本时，退化为「非中位数即离群」的保守策略，
      并在 ``note`` 里说明，避免零方差把真实异常全部放过。

    :param index_offset: 报表里样本的起始下标（例如样本来自第 100 条记录时传 100）。
    """
    samples = [float(value) for value in values]
    count = len(samples)
    if count < min_sample_size:
        return InsufficientSample(
            sample_size=count,
            required=min_sample_size,
            reason=(
                "MAD 是基于中位数的稳健统计量，小样本下方差估计不可靠；"
                "样本不足时不输出离群点，避免误导"
            ),
        )

    central = median(samples)
    mad = median_absolute_deviation(samples)
    note = ""
    if mad == 0:
        note = (
            "MAD 为 0（超过半数样本取值相同），已退化为「非中位数即离群」的保守策略；"
            "此时 modified z-score 无定义，不等号方向可能偏保守。"
        )

    found: list[Outlier] = []
    for offset, value in enumerate(samples):
        zscore = modified_zscore(value, central, mad, constant)
        if abs(zscore) > threshold:
            found.append(
                Outlier(
                    index=offset + index_offset,
                    value=value,
                    modified_z=zscore,
                    direction="high" if zscore > 0 else "low",
                )
            )

    found.sort(key=lambda item: (-abs(item.modified_z), item.index))
    return OutlierReport(
        sample_size=count,
        median=central,
        mad=mad,
        threshold=threshold,
        method="median+mad(modified z-score)",
        outliers=tuple(found),
        note=note,
    )


def percentile_anomalies(
    values: Sequence[float],
    *,
    q: float = 0.95,
) -> tuple[float, list[int]]:
    """基于分位数的**参考性**高位标记。

    这里刻意与 :func:`detect_outliers_by_mad` 分开，并在文档里写清楚它的性质：
    它是「取最高的那一小撮」，**不是**统计学意义上的异常检测。
    保留它只是为了报表里给运营一个「top N%」视角，判定异常请用 MAD。
    旧版把这两件事混为一谈，还把 ``int(n * 0.95)`` 当成分位数，才出现了恒空集合。

    :return: ``(分位数值, 严格大于该值的下标列表)``
    """
    threshold_value = percentile(values, q)
    flagged = [index for index, value in enumerate(values) if float(value) > threshold_value]
    return threshold_value, flagged


# ======================================================================================
# 成本计算
# ======================================================================================
@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """一次（或一批）调用的成本明细。

    ``total`` 的单位是元；所有单价字段的单位是 :data:`PRICE_UNIT`（元/百万 tokens）。
    """

    model: str
    total: Decimal
    currency: str
    unit: str
    tier: str
    at: datetime
    pricing_effective_date: str
    cache_hit_tokens: int
    cache_miss_tokens: int
    output_tokens: int
    estimated: bool
    price_cache_hit_input: Decimal
    price_cache_miss_input: Decimal
    price_output: Decimal
    cost_cache_hit_input: Decimal
    cost_cache_miss_input: Decimal
    cost_output: Decimal
    cache_savings: Decimal
    notes: tuple[str, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.cache_hit_tokens + self.cache_miss_tokens + self.output_tokens

    def explanation(self) -> str:
        """人类可读的计算过程，方便在 code review / 账单核对时逐项对账。"""
        lines = [
            f"模型 {self.model}，时段 {self.tier}（{self.at.isoformat()}）",
            f"计价单位：{self.unit}（价格生效日期 {self.pricing_effective_date}）",
            (
                f"输入·缓存命中 {self.cache_hit_tokens} tokens × {self.price_cache_hit_input} "
                f"{self.unit} = {self.cost_cache_hit_input} 元"
            ),
            (
                f"输入·缓存未命中 {self.cache_miss_tokens} tokens × {self.price_cache_miss_input} "
                f"{self.unit} = {self.cost_cache_miss_input} 元"
            ),
            (
                f"输出 {self.output_tokens} tokens × {self.price_output} "
                f"{self.unit} = {self.cost_output} 元"
            ),
            f"缓存带来的节省：{self.cache_savings} 元（相对全部按未命中计价）",
            f"合计：{self.total} {self.currency}",
        ]
        if self.estimated:
            lines.append("注意：本次 token 数为估算值（未能从响应中解析出 usage）。")
        lines.extend(self.notes)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """可 JSON 序列化的结构（Decimal 转 str，避免浮点丢精度）。"""
        return {
            "model": self.model,
            "total": str(self.total),
            "currency": self.currency,
            "unit": self.unit,
            "tier": self.tier,
            "at": self.at.isoformat(),
            "pricing_effective_date": self.pricing_effective_date,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "output_tokens": self.output_tokens,
            "estimated": self.estimated,
            "price_cache_hit_input": str(self.price_cache_hit_input),
            "price_cache_miss_input": str(self.price_cache_miss_input),
            "price_output": str(self.price_output),
            "cost_cache_hit_input": str(self.cost_cache_hit_input),
            "cost_cache_miss_input": str(self.cost_cache_miss_input),
            "cost_output": str(self.cost_output),
            "cache_savings": str(self.cache_savings),
            "notes": list(self.notes),
        }


def _as_int(value: Any, *, field_name: str) -> int:
    """把可能是 str/float/None 的 token 数安全转成 int。"""
    if value is None:
        return 0
    if isinstance(value, bool):  # bool 是 int 的子类，单独拦掉以免 True -> 1 的意外
        raise TypeError(f"{field_name} 不能是布尔值")
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal, str)):
        try:
            return int(Decimal(str(value)))
        except Exception as exc:  # noqa: BLE001 - 转成带上下文的错误信息
            raise ValueError(f"{field_name} 无法解析为整数：{value!r}") from exc
    raise TypeError(f"{field_name} 类型不支持：{type(value).__name__}")


def compute_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int | None = None,
    cache_hit: bool = False,
    at: datetime,
    estimated: bool = False,
    profile: PricingProfile | None = None,
    notes: Sequence[str] = (),
) -> CostBreakdown:
    """纯函数成本计算。

    计价公式::

        成本(元) = Σ tokens × 单价(元/百万 tokens) / 1_000_000

    输入 token 的拆分规则（三种用法，按优先级）：

    1. 显式给 ``cache_hit_tokens`` / ``cache_miss_tokens``：直接采用；
    2. 只给 ``input_tokens`` + ``cache_hit=True``：全部输入算缓存命中；
    3. 只给 ``input_tokens``：全部输入算缓存未命中（保守，金额偏高而不是偏低）。

    当 ``cache_miss_tokens`` 显式给出且与 ``input_tokens`` 不一致时，以显式值为准，
    并在 ``notes`` 里记录该不一致（不静默吞掉矛盾数据）。

    :raises UnknownModelError: 模型未收录（静默回退默认价会掩盖配置漏写）。
    :raises PricingNotEffectiveError: 计费时刻早于价格生效日期。
    :raises ValueError: token 数为负数，或缓存命中/未命中之和与 input_tokens 矛盾。
    """
    active = profile or DEFAULT_PRICING
    resolved = resolve_model_name(model, active)

    input_total = _as_int(input_tokens, field_name="input_tokens")
    output_total = _as_int(output_tokens, field_name="output_tokens")
    hit = _as_int(cache_hit_tokens, field_name="cache_hit_tokens")
    extra_notes: list[str] = list(notes)

    if cache_miss_tokens is None:
        if hit:
            miss = input_total - hit
        elif cache_hit:
            hit, miss = input_total, 0
        else:
            hit, miss = 0, input_total
            if input_total:
                extra_notes.append(
                    "未提供缓存命中信息，全部输入 token 按缓存未命中计价（偏保守，金额偏高而不是偏低）"
                )
    else:
        miss = _as_int(cache_miss_tokens, field_name="cache_miss_tokens")
        if input_total and hit + miss != input_total:
            extra_notes.append(
                f"cache_hit_tokens({hit}) + cache_miss_tokens({miss}) != "
                f"input_tokens({input_total})，已按显式的命中/未命中拆分计价"
            )
            if hit == 0 and cache_hit:
                extra_notes.append("cache_hit=True 与显式拆分同时给出，显式拆分优先")

    for name, value in (
        ("input_tokens", input_total),
        ("output_tokens", output_total),
        ("cache_hit_tokens", hit),
        ("cache_miss_tokens", miss),
    ):
        if value < 0:
            raise ValueError(f"{name} 不能为负数：{value}")

    if hit + miss != input_total and not extra_notes:
        # 走到这里说明用户只给了 cache_hit_tokens 但 input_tokens 为 0 之类的矛盾输入
        extra_notes.append(
            f"输入 token 汇总不一致：命中 {hit} + 未命中 {miss} != input_tokens {input_total}"
        )

    pricing = active.pricing_for(resolved, at)
    tier = active.tier_for(at)

    price_hit = pricing.rate("input", "cache_hit", tier)
    price_miss = pricing.rate("input", "cache_miss", tier)
    price_out = pricing.rate("output", "cache_miss", tier)

    cost_hit = Decimal(hit) * price_hit / TOKENS_PER_PRICE_UNIT
    cost_miss = Decimal(miss) * price_miss / TOKENS_PER_PRICE_UNIT
    cost_out = Decimal(output_total) * price_out / TOKENS_PER_PRICE_UNIT
    total = cost_hit + cost_miss + cost_out

    # 缓存节省额：命中部分相对「按未命中计价」少花掉的钱
    cache_savings = Decimal(hit) * (price_miss - price_hit) / TOKENS_PER_PRICE_UNIT

    return CostBreakdown(
        model=resolved,
        total=total,
        currency=pricing.currency,
        unit=PRICE_UNIT,
        tier=tier,
        at=active.localize(at),
        pricing_effective_date=pricing.effective_date,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
        output_tokens=output_total,
        estimated=estimated,
        price_cache_hit_input=price_hit,
        price_cache_miss_input=price_miss,
        price_output=price_out,
        cost_cache_hit_input=cost_hit,
        cost_cache_miss_input=cost_miss,
        cost_output=cost_out,
        cache_savings=cache_savings,
        notes=tuple(extra_notes),
    )


def sum_cost_breakdowns(breakdowns: Iterable[CostBreakdown]) -> dict[str, Any]:
    """把多条明细汇总成报表口径（金额用 Decimal 累加，避免浮点误差）。"""
    total = Decimal(0)
    estimated_count = 0
    count = 0
    by_model: dict[str, Decimal] = {}
    for item in breakdowns:
        count += 1
        total += item.total
        by_model[item.model] = by_model.get(item.model, Decimal(0)) + item.total
        if item.estimated:
            estimated_count += 1
    return {
        "count": count,
        "total": total,
        "estimated_count": estimated_count,
        "by_model": by_model,
    }


# ======================================================================================
# 对账
# ======================================================================================
@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """自身汇总与供应商账单的对账结果。

    旧版的问题不是「对账函数写错了」，而是**根本没有对账入口**：
    分账按 Hash 聚合、总账按 List 求和、key 有的有 TTL 有的没有，
    两套数据源各自漂移，谁也不知道差多少。有了这个结构，
    「差多少、差多少比例、是否超阈值」就是可断言的事实。
    """

    our_total: Decimal
    provider_total: Decimal
    difference: Decimal
    abs_difference: Decimal
    ratio: Decimal
    ratio_percent: Decimal
    tolerance: Decimal
    mismatch: bool
    currency: str = "CNY"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "our_total": str(self.our_total),
            "provider_total": str(self.provider_total),
            "difference": str(self.difference),
            "abs_difference": str(self.abs_difference),
            "ratio": str(self.ratio),
            "ratio_percent": str(self.ratio_percent),
            "tolerance": str(self.tolerance),
            "mismatch": self.mismatch,
            "currency": self.currency,
            "note": self.note,
        }

    def summary(self) -> str:
        state = "❌ 不一致（mismatch=True）" if self.mismatch else "✅ 在容差内"
        return (
            f"对账{state}：本系统 {self.our_total} {self.currency}，"
            f"账单 {self.provider_total} {self.currency}，"
            f"差额 {self.difference}（{self.ratio_percent}%），容差 {self.tolerance * _HUNDRED}%"
        )


def reconcile(
    our_total: Decimal | float | int | str,
    provider_bill_total: Decimal | float | int | str,
    *,
    tolerance: Decimal | float | str = Decimal("0.01"),
) -> ReconcileResult:
    """把自身汇总与账单总额对比，返回差额与差异比例。

    * ``difference = our_total - provider_total``（正数表示本地多记）；
    * ``ratio = |difference| / provider_total``（账单为 0 且本地也为 0 → 0；
      账单为 0 而本地非 0 → ``inf``，并直接判为 mismatch）；
    * ``mismatch = ratio > tolerance``。

    :raises ValueError: 账单或本地金额为负，或容差为负。
    """
    ours = Decimal(str(our_total))
    theirs = Decimal(str(provider_bill_total))
    tol = Decimal(str(tolerance))

    if ours < 0:
        raise ValueError(f"本地汇总金额不能为负：{ours}")
    if theirs < 0:
        raise ValueError(f"账单金额不能为负：{theirs}")
    if tol < 0:
        raise ValueError(f"容差不能为负：{tol}")

    difference = ours - theirs
    abs_difference = abs(difference)

    note = ""
    if theirs == 0:
        if ours == 0:
            ratio = Decimal(0)
            note = "账单与本地汇总均为 0，视为一致"
        else:
            ratio = Decimal("Infinity")
            note = "账单为 0 但本地有计量数据，无法计算比例，直接判为不一致"
    else:
        ratio = abs_difference / theirs

    # 差异比例可能是 Infinity（账单为 0），此时 quantize 会抛 InvalidOperation，
    # 因此只对有限值做定点化。
    ratio_percent = (ratio * _HUNDRED).quantize(Decimal("0.0001")) if ratio.is_finite() else ratio
    return ReconcileResult(
        our_total=ours,
        provider_total=theirs,
        difference=difference,
        abs_difference=abs_difference,
        ratio=ratio,
        ratio_percent=ratio_percent,
        tolerance=tol,
        mismatch=ratio > tol,
        note=note,
    )


@dataclass(frozen=True, slots=True)
class LedgerSplitCheck:
    """分账之和 vs 总账的校验结果。"""

    total: Decimal
    split_sum: Decimal
    difference: Decimal
    balanced: bool
    detail: Mapping[str, Decimal] = field(default_factory=dict)

    def summary(self) -> str:
        state = "✅ 平" if self.balanced else "❌ 不平"
        return (
            f"总账 {self.total} vs 分账之和 {self.split_sum}，"
            f"差额 {self.difference} → {state}"
        )


def check_ledger_split(
    total: Decimal | float | int | str,
    split: Mapping[str, Decimal | float | int | str],
) -> LedgerSplitCheck:
    """校验「分账之和 == 总账」。

    旧版分账来自 Hash、总账来自 List，两者写入时机与 TTL 都不同，
    没有任何机制保证它们相等。这里把它变成一个可测试的断言。
    """
    total_value = Decimal(str(total))
    split_values = {name: Decimal(str(value)) for name, value in split.items()}
    split_sum = sum(split_values.values(), Decimal(0))
    difference = total_value - split_sum
    return LedgerSplitCheck(
        total=total_value,
        split_sum=split_sum,
        difference=difference,
        balanced=difference == 0,
        detail=split_values,
    )
