"""报表与异常检测。

本模块把存储层读出来的数据整理成可读报表：

* :class:`LedgerAnalytics` —— 按天/按维度聚合、分位数报表、MAD 离群、
  **总账 == 分账之和** 校验、与账单对账；
* :func:`detect_retry_bursts` —— 基于**滑动窗口**的无效重试识别
  （窗口秒数与次数阈值是两个独立参数）；
* :class:`WindowAnalytics` —— 在 Redis ZSET 滑动窗口上做实时计数查询。

旧版在这里踩了三个坑，本模块逐一修掉：

1. 用 ``LRANGE 0 -1`` 拉整个 List 再在 Python 里过滤，复杂度 O(全部历史)；
2. 分位数取法错误（``int(n*0.95)`` 落在末位），导致异常集合恒为空；
3. ``threshold=8``（次数）被口头说成「60 秒」，窗口与次数混为一谈。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from .cost import (
    InsufficientSample,
    LedgerSplitCheck,
    OutlierReport,
    ReconcileResult,
    check_ledger_split,
    detect_outliers_by_mad,
    quantile_summary,
)
from .cost import (
    reconcile as reconcile_totals,
)
from .pricing import DEFAULT_PRICING, PRICE_UNIT, PricingProfile
from .settings import Settings, get_settings
from .storage import (
    DEFAULT_DIMENSION_AXIS,
    IdempotentLedger,
    SlidingWindowCounter,
    decode,
)

__all__ = [
    "LedgerEntry",
    "DailyReport",
    "RetryBurst",
    "LedgerAnalytics",
    "WindowAnalytics",
    "detect_retry_bursts",
    "window_call_counts",
    "parse_bucket_key",
    "discover_days",
]


# ======================================================================================
# 数据结构
# ======================================================================================
@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """一条明细记录（来自 Redis 分账 Hash 或调用方自己的明细表）。

    ``at`` 与 ``cost`` 分别用于滑动窗口与分位数/MAD 分析。
    注意：Redis 分账 Hash 只有「维度值 -> 当日金额」的聚合值、**没有时间戳**，
    因此需要窗口分析时必须由调用方提供带 ``at`` 的明细。
    """

    dimension: str
    cost: Decimal
    day: date
    at: datetime | None = None
    response_ms: float = 0.0
    trace_id: str = ""
    status: str = "ok"
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    output_tokens: int = 0

    @property
    def module(self) -> str:
        """别名：很多调用方把维度值当作模块名。"""
        return self.dimension

    @property
    def cost_float(self) -> float:
        return float(self.cost)


@dataclass(frozen=True, slots=True)
class DailyReport:
    """单日（或某个视图）的报告。"""

    label: str
    currency: str
    unit: str
    total: Decimal
    count: int
    by_axis: Mapping[str, Mapping[str, Decimal]]
    tokens: Mapping[str, Decimal]
    quantiles: Mapping[str, Any]
    missing_cost: Decimal = Decimal(0)
    note: str = ""

    @property
    def by_dimension(self) -> Mapping[str, Decimal]:
        """主维度轴（model）的分账，方便直接取用。"""
        return self.by_axis.get(DEFAULT_DIMENSION_AXIS, {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "currency": self.currency,
            "unit": self.unit,
            "total": str(self.total),
            "count": self.count,
            "by_axis": {
                axis: {name: str(value) for name, value in values.items()}
                for axis, values in self.by_axis.items()
            },
            "tokens": {name: str(value) for name, value in self.tokens.items()},
            "quantiles": {
                key: (str(value) if isinstance(value, Decimal) else value)
                for key, value in self.quantiles.items()
            },
            "missing_cost": str(self.missing_cost),
            "note": self.note,
        }

    def render(self) -> str:
        """人类可读的文本报表。"""
        lines = [
            f"报表 {self.label}（单位：{self.unit}）",
            f"  总成本：{self.total} {self.currency}   分组条目数：{self.count}",
        ]
        for axis, values in self.by_axis.items():
            if not values:
                continue
            lines.append(f"  按 {axis} 分解：")
            for name, value in sorted(values.items(), key=lambda kv: -kv[1]):
                share = (
                    (value / self.total * 100).quantize(Decimal("0.01")) if self.total else Decimal(0)
                )
                lines.append(f"    - {name}: {value} {self.currency}（{share}%）")
        if self.tokens:
            lines.append("  token 合计：")
            for name, value in sorted(self.tokens.items()):
                lines.append(f"    - {name}: {value}")
        quantile_text = ", ".join(
            f"{key}={value}"
            for key, value in self.quantiles.items()
            if key.startswith("p") and value is not None
        )
        if quantile_text:
            lines.append(f"  分位数（按分组金额）：{quantile_text}")
        if self.missing_cost:
            lines.append(f"  未入账（存储不可用）金额：{self.missing_cost}")
        if self.note:
            lines.append(f"  备注：{self.note}")
        return "\n".join(lines)


# ======================================================================================
# 滑动窗口 / 无效重试
# ======================================================================================
@dataclass(frozen=True, slots=True)
class RetryBurst:
    """一次「窗口内调用次数超阈值」的突发。"""

    dimension: str
    start: datetime
    end: datetime
    count: int
    window_seconds: int
    max_calls: int
    span_seconds: float
    call_rate_per_second: float
    trace_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "count": self.count,
            "window_seconds": self.window_seconds,
            "max_calls": self.max_calls,
            "span_seconds": self.span_seconds,
            "call_rate_per_second": self.call_rate_per_second,
            "trace_ids": list(self.trace_ids),
        }


def detect_retry_bursts(
    entries: Iterable[LedgerEntry],
    *,
    window_seconds: int,
    max_calls: int,
    group_by: str = "dimension",
) -> list[RetryBurst]:
    """在内存里做滑动窗口统计，识别「短时间高频重复调用」（疑似无效重试）。

    **两个参数完全独立**：

    * ``window_seconds`` —— 只看最近 N 秒；
    * ``max_calls`` —— 这 N 秒内超过 M 次才算突发。

    旧版把 ``threshold=8``（次数）描述成「60 秒」，两个概念混用，
    导致既看不出真实窗口、也无法分别调参。本函数要求两个参数都显式传入，
    并有专门的回归测试证明二者互不串味。

    实现：按维度分组 → 按时间排序 → 双指针维护「以当前点为终点的最长合法窗口」，
    复杂度 O(n log n)（瓶颈在排序），不做任何 O(n²) 暴力扫描。

    合并策略：**一次突发只报一条**。如果每个「超阈值的窗口终点」都报一条，
    一次重试风暴会产出几十条几乎相同的记录，反而看不出问题；
    因此这里只保留「无法再向前扩展」的极大窗口（即下一个点的窗口会排除掉当前起点）。
    """
    if window_seconds < 1:
        raise ValueError(f"window_seconds 必须 >= 1，收到 {window_seconds}")
    if max_calls < 1:
        raise ValueError(f"max_calls 必须 >= 1，收到 {max_calls}")

    grouped: dict[str, list[LedgerEntry]] = defaultdict(list)
    for entry in entries:
        if entry.at is None:
            continue
        key = entry.dimension if group_by == "dimension" else getattr(entry, group_by, entry.dimension)
        grouped[str(key)].append(entry)

    bursts: list[RetryBurst] = []
    for dimension, items in grouped.items():
        ordered = sorted(items, key=lambda item: item.at)  # type: ignore[arg-type,return-value]
        count = len(ordered)
        start_index = 0
        for end_index in range(count):
            end_at = ordered[end_index].at
            assert end_at is not None  # 前面已过滤
            while end_at - ordered[start_index].at > timedelta(seconds=window_seconds):  # type: ignore[operator]
                start_index += 1
            if end_index - start_index + 1 <= max_calls:
                continue
            # 只保留极大窗口：下一个点若仍在窗口内，说明当前起点还能继续延展
            if end_index + 1 < count:
                next_at = ordered[end_index + 1].at
                assert next_at is not None
                if next_at - ordered[start_index].at <= timedelta(seconds=window_seconds):
                    continue
            window = ordered[start_index : end_index + 1]
            first_at = window[0].at
            assert first_at is not None
            span = (end_at - first_at).total_seconds()
            bursts.append(
                RetryBurst(
                    dimension=dimension,
                    start=first_at,
                    end=end_at,
                    count=len(window),
                    window_seconds=window_seconds,
                    max_calls=max_calls,
                    span_seconds=span,
                    call_rate_per_second=(len(window) / span) if span > 0 else float("inf"),
                    trace_ids=tuple(item.trace_id for item in window if item.trace_id),
                )
            )

    bursts.sort(key=lambda burst: (-burst.count, burst.dimension, burst.start))
    return bursts


def window_call_counts(
    timestamps: Sequence[float],
    *,
    end: float,
    window_seconds: int,
) -> int:
    """纯函数：统计 ``(end - window_seconds, end]`` 区间内的时间戳个数。

    左开右闭：正好落在 ``end - window_seconds`` 的点不算在内 ——
    与 Redis ``ZREMRANGEBYSCORE -inf cutoff``（两端都是闭区间删除）的语义保持一致，
    因此紧贴边界的成员在这两个实现里都会被丢弃。
    """
    if window_seconds < 1:
        raise ValueError(f"window_seconds 必须 >= 1，收到 {window_seconds}")
    cutoff = end - window_seconds
    return sum(1 for value in timestamps if cutoff < value <= end)


# ======================================================================================
# 报表
# ======================================================================================
class LedgerAnalytics:
    """基于 :class:`~costgovernor.storage.IdempotentLedger` 的报表与检测。"""

    def __init__(
        self,
        ledger: IdempotentLedger,
        settings: Settings | None = None,
        *,
        windows: SlidingWindowCounter | None = None,
        profile: PricingProfile | None = None,
        kind: str | None = None,
    ) -> None:
        self.ledger = ledger
        self.settings = settings or get_settings()
        self.windows = windows
        self.profile = profile or DEFAULT_PRICING
        self.kind = kind or ledger.kind

    # -------- 聚合 --------
    def daily_report(
        self,
        day: date,
        *,
        axes: Sequence[str] | None = None,
    ) -> DailyReport:
        """某一天的报告：总账、各维度轴分账、token 与分位数。"""
        target_axes = list(axes) if axes is not None else (self.ledger.known_axes(kind=self.kind) or [DEFAULT_DIMENSION_AXIS])
        by_axis: dict[str, dict[str, Decimal]] = {}
        for axis in target_axes:
            values = self.ledger.dimension_totals(day, axis=axis, kind=self.kind)
            if values:
                by_axis[axis] = values

        total = self.ledger.bucket_total(day, kind=self.kind)
        tokens = self.ledger.token_grand_totals(day, kind=self.kind)
        primary = by_axis.get(DEFAULT_DIMENSION_AXIS, {})
        quantiles: dict[str, Any] = {"count": len(primary)}
        if primary:
            quantiles = quantile_summary([float(value) for value in primary.values()])

        note = ""
        if not by_axis:
            note = (
                "该日没有分账数据；可能原因：尚未写入、TTL 已过期，"
                "或写入时 Redis 不可用（此时金额未被入账，不会伪造为 0）"
            )
        return DailyReport(
            label=f"{day.isoformat()}（{self.kind}）",
            currency="CNY",
            unit=PRICE_UNIT,
            total=total,
            count=len(primary),
            by_axis=by_axis,
            tokens=tokens,
            quantiles=quantiles,
            note=note,
        )

    def overview(self, *, days: Sequence[date] | None = None, axes: Sequence[str] | None = None) -> dict[str, Any]:
        """多天总览（默认自动发现所有有数据的天）。"""
        target_days = list(days) if days is not None else self.discover_days()
        per_day = [self.daily_report(day, axes=axes) for day in target_days]

        totals_by_axis: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
        tokens: dict[str, Decimal] = defaultdict(Decimal)
        total = Decimal(0)
        count = 0
        for report in per_day:
            total += report.total
            count += report.count
            for axis, values in report.by_axis.items():
                for name, value in values.items():
                    totals_by_axis[axis][name] += value
            for token_name, token_value in report.tokens.items():
                tokens[token_name] += Decimal(token_value)

        return {
            "kind": self.kind,
            "unit": PRICE_UNIT,
            "currency": "CNY",
            "days": [report.label for report in per_day],
            "total": total,
            "count": count,
            "by_axis": {axis: dict(values) for axis, values in totals_by_axis.items()},
            "tokens": dict(tokens),
            "per_day": [report.to_dict() for report in per_day],
        }

    def discover_days(self) -> list[date]:
        """发现有数据的天：优先用天索引集合，缺失时回退到 key 扫描。"""
        days = self.ledger.known_days(kind=self.kind)
        if days:
            return days
        return self.scan_days()

    def scan_days(self) -> list[date]:
        """扫描 ``<prefix>:<kind>:<YYYY-MM-DD>*`` 形式的 key（只读）。"""
        pattern = f"{self.ledger.keyspace.prefix}:{self.kind}:*"
        try:
            raw_keys = self.ledger.client.keys(pattern)
        except Exception as exc:  # noqa: BLE001 - 扫描失败不是致命错误
            self.ledger._log("scan_days_failed", error=repr(exc), pattern=pattern)  # noqa: SLF001
            return []
        found: set[date] = set()
        for raw in raw_keys or []:
            parsed = parse_bucket_key(str(decode(raw)), prefix=self.ledger.keyspace.prefix, kind=self.kind)
            if parsed is not None:
                found.add(parsed)
        return sorted(found)

    def outliers(
        self,
        day: date,
        *,
        axis: str = DEFAULT_DIMENSION_AXIS,
        threshold: float | None = None,
        min_sample_size: int | None = None,
    ) -> OutlierReport | InsufficientSample:
        """对某一天某个维度轴的分组金额做 MAD 离群检测。

        样本不足时返回 :class:`~costgovernor.cost.InsufficientSample`（明确「样本不足」），
        而不是像旧版那样硬算出一个恒为空的「异常集合」。

        注意口径：这里检测的是**分组维度值**（例如各模型的当日金额）是否为离群值。
        若要做「逐次调用成本」的离群检测，请把带 ``at`` 的明细通过
        :meth:`outliers_from_costs` 传入。
        """
        values = [
            float(value)
            for value in self.ledger.dimension_totals(day, axis=axis, kind=self.kind).values()
        ]
        return detect_outliers_by_mad(
            values,
            threshold=threshold if threshold is not None else self.settings.mad_threshold,
            min_sample_size=(
                min_sample_size if min_sample_size is not None else self.settings.min_sample_size
            ),
        )

    def outliers_from_costs(
        self,
        costs: Sequence[float],
        *,
        threshold: float | None = None,
        min_sample_size: int | None = None,
    ) -> OutlierReport | InsufficientSample:
        """对任意一串调用成本做 MAD 离群检测（逐次调用的口径）。"""
        return detect_outliers_by_mad(
            list(costs),
            threshold=threshold if threshold is not None else self.settings.mad_threshold,
            min_sample_size=(
                min_sample_size if min_sample_size is not None else self.settings.min_sample_size
            ),
        )

    # -------- 对账 --------
    def totals(self, days: Sequence[date] | None = None, *, axis: str = DEFAULT_DIMENSION_AXIS) -> dict[str, Decimal]:
        """按维度轴汇总（分账）。"""
        target_days = list(days) if days is not None else self.discover_days()
        by_value: dict[str, Decimal] = defaultdict(Decimal)
        for day in target_days:
            for name, value in self.ledger.dimension_totals(day, axis=axis, kind=self.kind).items():
                by_value[name] += value
        return dict(by_value)

    def grand_total(self, days: Sequence[date] | None = None) -> Decimal:
        """总账（与分账同源写入，见 :meth:`split_check`）。"""
        target_days = list(days) if days is not None else self.discover_days()
        return sum((self.ledger.bucket_total(day, kind=self.kind) for day in target_days), Decimal(0))

    def split_check(
        self,
        days: Sequence[date] | None = None,
        *,
        axis: str = DEFAULT_DIMENSION_AXIS,
    ) -> LedgerSplitCheck:
        """校验「分账之和 == 总账」。

        旧版最要命的账目问题：分账按 Hash 聚合、总账按 List 求和，
        写入时机与 TTL 都不同，没有任何机制保证相等。
        本版两者在同一段 Lua 里写入相同金额，因此这里应当恒为平。
        """
        target_days = list(days) if days is not None else self.discover_days()
        total = sum((self.ledger.bucket_total(day, kind=self.kind) for day in target_days), Decimal(0))
        split: dict[str, Decimal] = defaultdict(Decimal)
        for day in target_days:
            for name, value in self.ledger.dimension_totals(day, axis=axis, kind=self.kind).items():
                split[name] += value
        return check_ledger_split(total, dict(split))

    def reconcile(
        self,
        provider_bill_total: Decimal | float | str,
        *,
        days: Sequence[date] | None = None,
        tolerance: Decimal | float | str | None = None,
    ) -> ReconcileResult:
        """与供应商账单对账。"""
        active_tolerance = tolerance if tolerance is not None else self.settings.reconcile_tolerance
        return reconcile_totals(
            self.grand_total(days),
            provider_bill_total,
            tolerance=active_tolerance,
        )

    # -------- 无效重试 --------
    def retry_bursts(
        self,
        *,
        entries: Sequence[LedgerEntry] | None = None,
        days: Sequence[date] | None = None,
        window_seconds: int | None = None,
        max_calls: int | None = None,
    ) -> list[RetryBurst]:
        """滑动窗口重试识别。

        ``window_seconds`` / ``max_calls`` 两个参数独立，未传时分别取
        ``settings.retry_window_seconds`` / ``settings.retry_max_calls``。

        **需要带时间戳的明细**：Redis 分账 Hash 只有「维度值 -> 当日金额」的聚合值，
        没有时间戳，无法做时间窗口分析。两种情况：

        * 传入 ``entries``（例如来自自己的明细表，或用 ``recorder`` 落库的数据）；
        * 或覆写 :meth:`load_entries`。

        两者都没有时返回空列表并记录一条警告 —— 明确「这里没有数据可用」，
        而不是像旧版那样拿全量 List 硬滤，制造一个看起来在工作、实际算错的窗口。
        """
        active_window = (
            window_seconds if window_seconds is not None else self.settings.retry_window_seconds
        )
        active_max = max_calls if max_calls is not None else self.settings.retry_max_calls

        candidates = list(entries) if entries is not None else self.load_entries(days=days)
        timed = [entry for entry in candidates if entry.at is not None]
        if not timed:
            self.ledger._log(  # noqa: SLF001 - 复用存储层的日志通道
                "retry_bursts_skipped",
                reason="没有带时间戳的明细数据，无法做滑动窗口分析",
                hint="用 recorder 回调把逐次调用明细落库，或改用 WindowAnalytics 读 Redis ZSET 窗口",
            )
            return []
        return detect_retry_bursts(timed, window_seconds=active_window, max_calls=active_max)

    def load_entries(self, *, days: Sequence[date] | None = None) -> list[LedgerEntry]:
        """加载明细条目。

        基类返回**空列表**：Redis 分账 Hash 里没有逐次调用的时间戳。
        需要窗口分析的调用方应当覆写本方法（或直接把 ``entries`` 传给
        :meth:`retry_bursts`），也可以改为直接查询 Redis ZSET 滑动窗口
        （见 :class:`WindowAnalytics`）。
        """
        return []

    def window_counts(
        self,
        *,
        now: float | None = None,
        window_seconds: int | None = None,
    ) -> int:
        """从 Redis ZSET 滑动窗口读取「当前窗口内调用次数」（O(log N)，不拉历史）。"""
        if self.windows is None:
            raise RuntimeError("未注入 SlidingWindowCounter，无法查询 Redis 滑动窗口")
        return self.windows.count(now=now, window_seconds=window_seconds)


class WindowAnalytics:
    """直接基于 Redis ZSET 滑动窗口的实时查询。

    这才是「无效重试识别」的正确姿势：
    ``ZREMRANGEBYSCORE`` 清理 + ``ZCARD`` 计数，单次复杂度 O(log N)，
    不需要把任何历史数据拉回 Python（旧版是 ``LRANGE 0 -1`` + 逐条 ``json.loads``）。
    """

    def __init__(self, counter: SlidingWindowCounter, settings: Settings | None = None) -> None:
        self.counter = counter
        self.settings = settings or get_settings()

    def status(
        self,
        *,
        now: float | None = None,
        window_seconds: int | None = None,
        max_calls: int | None = None,
    ) -> dict[str, Any]:
        """当前窗口的计数与是否超阈值。"""
        active_window = (
            window_seconds if window_seconds is not None else self.settings.retry_window_seconds
        )
        active_max = max_calls if max_calls is not None else self.settings.retry_max_calls
        count = self.counter.count(now=now, window_seconds=active_window)
        return {
            "scope": self.counter.scope,
            "window_seconds": active_window,
            "max_calls": active_max,
            "count": count,
            "exceeded": count > active_max,
        }


# ======================================================================================
# 辅助纯函数
# ======================================================================================
def parse_bucket_key(key: str, *, prefix: str = "llm", kind: str = "cost") -> date | None:
    """从 ``llm:cost:2026-10-07[:model]`` 里解析出日期；解析不出返回 ``None``。"""
    parts = key.split(":")
    if len(parts) < 3 or parts[0] != prefix or parts[1] != kind:
        return None
    try:
        return date.fromisoformat(parts[2])
    except ValueError:
        return None


def discover_days(client: Any, *, prefix: str = "llm", kind: str = "cost") -> list[date]:
    """扫描 Redis key 发现所有有数据的天（只读）。"""
    pattern = f"{prefix}:{kind}:*"
    try:
        raw_keys = client.keys(pattern)
    except Exception:  # noqa: BLE001 - 发现失败返回空列表，由调用方决定是否提示
        return []
    found: set[date] = set()
    for raw in raw_keys or []:
        parsed = parse_bucket_key(str(decode(raw)), prefix=prefix, kind=kind)
        if parsed is not None:
            found.add(parsed)
    return sorted(found)


def build_entries_from_costs(
    costs: Sequence[float],
    *,
    dimension: str = "unknown",
    day: date | None = None,
    start: datetime | None = None,
    step_seconds: float = 1.0,
) -> list[LedgerEntry]:
    """测试/演示辅助：把一串成本构造成等间隔的明细条目。"""
    base_day = day or date.today()
    entries: list[LedgerEntry] = []
    for index, cost in enumerate(costs):
        at = None if start is None else start + timedelta(seconds=index * step_seconds)
        entries.append(LedgerEntry(dimension=dimension, cost=Decimal(str(cost)), day=base_day, at=at))
    return entries
