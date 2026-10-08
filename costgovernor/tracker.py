"""``@track_llm`` 装饰器：同步 + 异步、成功 + 失败都要记账。

旧版装饰器的两个致命问题：

1. ``def wrapper`` + ``result = func(*args, **kwargs)`` 写在 ``try`` 之外 ——
   被装饰的 **async** 函数返回的是协程对象，下一行 ``result.get("usage")``
   直接 ``AttributeError``，还被 ``except Exception`` 吞掉只 ``print`` 一行；
2. **函数抛异常时埋点代码根本不执行**，失败调用全部漏记 ——
   而失败调用（超时、限流、上游 5xx）恰恰是最该被看见、也最花钱的那部分。

本模块的做法：

* 用 ``inspect.iscoroutinefunction`` 判定，同步/异步各生成一个包装；
* 计时用 ``time.perf_counter()``（单调时钟，不受系统时间调整影响）；
* 失败记录在 ``except`` 分支里完成（``status="error"`` + 异常类型），然后**原样抛出**；
* 日志走 ``logging`` / ``structlog``，不出现 ``print``；
* usage 解析支持 dict / 带 ``.usage`` 属性的对象 / 直接传 ``usage=`` 关键字，
  拿不到时标记 ``estimated=True`` 并在 ``notes`` 里说明估算依据。
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from .cost import CostBreakdown, compute_cost
from .pricing import DEFAULT_PRICING, PricingProfile

__all__ = [
    "UsageInfo",
    "UsageParseError",
    "LLMCallRecord",
    "extract_usage",
    "parse_trace_id",
    "estimate_tokens_from_text",
    "track_llm",
    "structlog_configure",
    "get_logger",
    "CHARS_PER_TOKEN",
]

#: 估算兜底时假设的「字符/token」比值（中英文混排的经验值）。
#: 只用于 ``estimated=True`` 的场景，绝不可用于对账。
CHARS_PER_TOKEN = 4.0

_TRACE_ID_KEYS = ("trace_id", "traceid", "request_id", "requestid", "correlation_id")
_USAGE_KEYS = ("usage", "token_usage", "usages")
_MODEL_KEYS = ("model", "model_name")


class UsageParseError(Exception):
    """usage 无法解析时的内部信号（不会冒泡到业务代码，只用于切换估算分支）。"""


# --------------------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------------------
_STRUCTLOG_CONFIGURED = False


def structlog_configure(level: str = "INFO", fmt: str = "console") -> None:
    """配置 structlog（幂等）。

    刻意不在导入期执行：库不应该在 ``import`` 时改动宿主进程的日志配置。
    """
    global _STRUCTLOG_CONFIGURED
    if _STRUCTLOG_CONFIGURED:
        return
    try:  # pragma: no cover - 依赖 structlog
        import structlog

        processors: list[Any] = [
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.processors.StackInfoRenderer(),
        ]
        if fmt == "json":
            processors.append(structlog.processors.JSONRenderer(ensure_ascii=False))
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=False))
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        _STRUCTLOG_CONFIGURED = True
    except ImportError:  # pragma: no cover
        logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str = "costgovernor") -> Any:
    """返回一个 logger：优先 structlog，缺失时退回标准库 ``logging``。"""
    try:
        import structlog

        structlog_configure()
        return structlog.get_logger(name)
    except ImportError:  # pragma: no cover
        return logging.getLogger(name)


# --------------------------------------------------------------------------------------
# usage 解析
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class UsageInfo:
    """从响应里解析出来的 token 用量。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    estimated: bool = False
    source: str = "unknown"
    notes: tuple[str, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "estimated": self.estimated,
            "source": self.source,
            "notes": list(self.notes),
        }


def _coerce_int(value: Any) -> int | None:
    """把 token 字段转成整数；不能转就返回 ``None``（``None`` 表示缺失，``0`` 表示确实是 0）。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(Decimal(value))
        except Exception:  # noqa: BLE001 - 字符串解析失败就当作缺失
            return None
    return None


def _first_int(source: Any, names: tuple[str, ...]) -> tuple[int | None, str | None]:
    """按候选字段名取值，返回 ``(值, 命中的字段名)``。"""
    if source is None:
        return None, None
    for name in names:
        if isinstance(source, Mapping):
            if name not in source:
                continue
            raw = source[name]
        else:
            try:
                raw = getattr(source, name)
            except AttributeError:
                continue
        parsed = _coerce_int(raw)
        if parsed is not None:
            return parsed, name
    return None, None


def _find_usage_container(payload: Any) -> tuple[Any, str]:
    """在响应对象里定位 usage 容器。

    顺序（对「unknown」错误信息的可读性很重要）：

    1. 带 ``.usage`` / ``["usage"]`` 的常规 SDK 响应；
    2. 干脆传进来的就是 usage 容器本身（``{"input_tokens": ..}``、
       或被装饰函数把它作为普通返回值返回）—— 只在里面确实有 token 字段时才认，
       免得把「响应 dict 本身」误当成 usage 容器而解析出全 0；
    3. 再往 ``response`` / ``data`` / ``result`` 里钻一层。
    """
    for key in _USAGE_KEYS:
        if isinstance(payload, Mapping):
            if payload.get(key) is not None:
                return payload[key], f"mapping[{key!r}]"
            continue
        try:
            value = getattr(payload, key)
        except AttributeError:
            continue
        if value is not None:
            return value, f"attribute.{key}"

    if _first_int(payload, ("input_tokens", "prompt_tokens"))[0] is not None or (
        _first_int(payload, ("output_tokens", "completion_tokens"))[0] is not None
    ):
        return payload, "usage-container"

    for nested_key in ("response", "data", "result"):
        if isinstance(payload, Mapping):
            nested = payload.get(nested_key)
        else:
            nested = getattr(payload, nested_key, None)
        if nested is None or nested is payload:
            continue
        try:
            return _find_usage_container(nested)
        except UsageParseError:
            continue
    raise UsageParseError("响应里找不到 usage 字段")


def extract_usage(payload: Any, *extra_sources: Any) -> UsageInfo:
    """从任意响应形态里解析 token 用量。

    支持的形态：

    * ``{"usage": {"prompt_tokens": 10, "completion_tokens": 3}}``（OpenAI 风格）
    * ``{"usage": {"input_tokens": 10, "output_tokens": 3, "prompt_cache_hit_tokens": 8}}``
    * 带 ``.usage`` 属性的对象（SDK 返回的 pydantic 模型）
    * 直接给 usage 容器本身（``extract_usage({"input_tokens": 1})``）
    * 额外位置参数里的任意一个（如异常对象上挂的 usage）

    :raises UsageParseError: 所有候选都解析不到时抛出；
        装饰器会捕获它并退化为 ``estimated=True``。
    """
    candidates: list[Any] = [payload, *extra_sources]
    last_error: Exception | None = None

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            container, where = _find_usage_container(candidate)
        except UsageParseError as exc:
            last_error = exc
            continue

        input_tokens, input_field = _first_int(
            container, ("input_tokens", "prompt_tokens", "prompt_token_count")
        )
        output_tokens, output_field = _first_int(
            container, ("output_tokens", "completion_tokens", "completion_token_count")
        )
        if input_tokens is None and output_tokens is None:
            last_error = UsageParseError(f"{where} 里没有 input/output token 字段")
            continue

        hit_tokens, hit_field = _first_int(
            container,
            (
                "prompt_cache_hit_tokens",
                "cache_hit_tokens",
                "cached_tokens",
                "input_cache_hit_tokens",
            ),
        )
        miss_tokens, _miss_field = _first_int(
            container,
            (
                "prompt_cache_miss_tokens",
                "cache_miss_tokens",
                "input_cache_miss_tokens",
                "uncached_tokens",
            ),
        )

        notes: list[str] = []
        input_total = input_tokens or 0
        hit = hit_tokens or 0
        if miss_tokens is None:
            miss = max(input_total - hit, 0)
            if hit and hit_field:
                notes.append(
                    f"响应未提供缓存未命中字段，按 input_tokens({input_total}) - {hit_field}({hit}) 推算"
                )
            else:
                miss = input_total
                notes.append("响应未提供缓存命中信息，全部输入按缓存未命中计价（偏保守）")
        else:
            miss = miss_tokens
            if input_total and hit + miss != input_total:
                notes.append(
                    f"命中({hit}) + 未命中({miss}) != input_tokens({input_total})，以显式拆分为准"
                )

        return UsageInfo(
            input_tokens=input_total,
            output_tokens=output_tokens or 0,
            cache_hit_tokens=hit,
            cache_miss_tokens=miss,
            estimated=False,
            source=f"{where}:{input_field or '-'}/{output_field or '-'}",
            notes=tuple(notes),
        )

    raise UsageParseError(f"无法从响应中解析 usage：{last_error}")


def estimate_tokens_from_text(text: str | None, *, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """按字符数粗略估算 token 数。

    这是兜底手段，不是度量：真实 token 数只有 provider 的 usage 才权威。
    """
    if not text:
        return 0
    return max(int(len(text) / chars_per_token), 1)


def _result_text(result: Any) -> str:
    """从返回值里尽力取出「文本」，用于估算 token。"""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        for key in ("text", "content", "output_text", "answer"):
            if result.get(key):
                return str(result[key])
        return ""
    for key in ("text", "content", "output_text"):
        value = getattr(result, key, None)
        if value:
            return str(value)
    return ""


def parse_trace_id(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    explicit: str | None = None,
) -> str:
    """确定本次逻辑调用的 ``trace_id``。

    优先级：显式参数 → 调用方的 ``trace_id``/``request_id`` 关键字 → 新建 UUID4。
    重试必须复用同一个 ``trace_id``，所以一定要支持显式传入。
    """
    if explicit:
        return str(explicit)
    for key in _TRACE_ID_KEYS:
        if kwargs.get(key):
            return str(kwargs[key])
    for value in args:
        if isinstance(value, str) and value.startswith("trace-"):
            return value
    return f"trace-{uuid.uuid4().hex}"


def _extract_model(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    explicit: str | None,
) -> str:
    """尽力推断模型名。

    顺序：显式装饰器参数 → 调用时的 ``model=`` 关键字 → 被装饰函数的默认值
    （例如 ``def ask(prompt, model="deepseek-flash")`` 用 ``@track_llm`` 裸装饰时）
    → ``"unknown"``。

    返回 ``"unknown"`` 时价格表会抛
    :class:`~costgovernor.pricing.UnknownModelError`（显式失败），
    而不是像旧版那样静默按默认价计费，把配置漏写掩盖成「成本偏低」。
    """
    if explicit:
        return explicit
    for key in _MODEL_KEYS:
        if kwargs.get(key):
            return str(kwargs[key])
    return "unknown"


def _model_from_signature(target: Callable[..., Any]) -> str | None:
    """从函数签名的默认值里找模型名（供裸 ``@track_llm`` 用）。"""
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # pragma: no cover - 内建函数等没有签名
        return None
    for key in _MODEL_KEYS:
        parameter = signature.parameters.get(key)
        if parameter is None:
            continue
        default = parameter.default
        if isinstance(default, str) and default:
            return default
    return None


_PROMPT_KEYS = ("prompt", "prompt_text", "messages", "input", "question")


def _prompt_text_from_call(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    signature: inspect.Signature | None,
) -> str:
    """把调用参数里的 prompt 提取成文本，供「估算 token」分支使用。

    位置参数也要能取到（``ask("..." * 400)`` 这种写法很常见），
    所以优先用签名把位置参数绑定到形参名。
    """
    bound: Mapping[str, Any] = {}
    if signature is not None:
        try:
            bound = signature.bind_partial(*args, **kwargs).arguments
        except TypeError:  # pragma: no cover - 参数不匹配时退回关键字查找
            bound = dict(kwargs)
    else:  # pragma: no cover
        bound = dict(kwargs)

    for key in _PROMPT_KEYS:
        if key in bound and bound[key] is not None:
            return str(bound[key])
    for key in _PROMPT_KEYS:
        if kwargs.get(key) is not None:
            return str(kwargs[key])
    return ""


# --------------------------------------------------------------------------------------
# 调用记录
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    """一次调用的埋点记录（成功或失败）。"""

    trace_id: str
    model: str
    module: str
    status: str  # "ok" | "error"
    latency_ms: float
    at: datetime
    usage: UsageInfo
    breakdown: CostBreakdown
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def cost(self) -> Decimal:
        return self.breakdown.total

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "model": self.model,
            "module": self.module,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "at": self.at.isoformat(),
            "usage": self.usage.to_dict(),
            "cost": str(self.cost),
            "tier": self.breakdown.tier,
            "estimated": self.usage.estimated,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }


# --------------------------------------------------------------------------------------
# 装饰器
# --------------------------------------------------------------------------------------
def track_llm(
    func: Callable[..., Any] | None = None,
    *,
    model: str | None = None,
    module: str | None = None,
    ledger: Any | None = None,
    recorder: Callable[[LLMCallRecord], Any] | None = None,
    trace_id: str | None = None,
    profile: PricingProfile | None = None,
    logger: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
    now: Callable[[], datetime] = datetime.now,
) -> Any:
    """埋点装饰器：同步函数与异步函数都支持，成功与失败都记账。

    用法::

        @track_llm(model="deepseek-flash")
        def ask(prompt: str) -> dict: ...

        @track_llm(model="deepseek-v4-pro", module="report")
        async def ask_async(prompt: str) -> Any: ...

        # 也支持不带括号：模型名从调用参数推断（推断不出即为 "unknown"，会显式报错）
        @track_llm
        def ask(prompt, model="deepseek-flash"): ...

    :param ledger: 可选，任何带 ``add_entry(trace_id, amount, dimension=..., tokens=...)``
        的对象（通常是 :class:`costgovernor.storage.IdempotentLedger`）。
    :param recorder: 可选，接收 :class:`LLMCallRecord` 的回调。
    :param clock: 计时函数，默认 ``time.perf_counter``（单调时钟）。
    :param now: 取当前时间，默认 ``datetime.now``；测试可注入固定时刻。
    """
    active_profile = profile or DEFAULT_PRICING

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        active_logger = logger or get_logger(f"costgovernor.tracker.{target.__module__}")
        # 只解析一次签名：既用于推断模型默认值，也用于把位置参数绑定到形参名
        try:
            _signature: inspect.Signature | None = inspect.signature(target)
        except (TypeError, ValueError):  # pragma: no cover
            _signature = None
        signature_model = _model_from_signature(target)

        def _default_module() -> str:
            return f"{target.__module__}.{target.__qualname__}"

        def _build_usage(
            result: Any,
            kwargs: Mapping[str, Any],
            error: BaseException | None,
            call_args: tuple[Any, ...] = (),
        ) -> UsageInfo:
            """解析 usage；解析不到就退化为估算并明确标记 ``estimated=True``。"""
            extra: list[Any] = []
            if error is not None:
                extra.append(getattr(error, "usage", None))
            for key in ("usage", "token_usage"):
                if kwargs.get(key) is not None:
                    extra.append(kwargs[key])
            try:
                return extract_usage(result, *extra)
            except UsageParseError as exc:
                prompt_text = _prompt_text_from_call(call_args, kwargs, _signature)
                estimated_input = estimate_tokens_from_text(prompt_text + _result_text(result))
                return UsageInfo(
                    input_tokens=estimated_input,
                    output_tokens=0,
                    cache_hit_tokens=0,
                    cache_miss_tokens=estimated_input,
                    estimated=True,
                    source="estimated",
                    notes=(
                        f"未能解析 usage（{exc}）；已按约 {CHARS_PER_TOKEN:g} 字符/token 估算输入 "
                        "token，output_tokens 记为 0。该记录 estimated=True，金额仅供趋势参考，"
                        "不可用于账单对账。",
                    ),
                )

        def _emit(
            trace: str,
            model_name: str,
            module_name: str,
            status: str,
            latency_ms: float,
            at: datetime,
            usage: UsageInfo,
            error: BaseException | None,
        ) -> None:
            """统一的记账入口：成功和失败走同一条路径。"""
            breakdown = compute_cost(
                model_name,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_hit_tokens=usage.cache_hit_tokens,
                cache_miss_tokens=usage.cache_miss_tokens,
                at=at,
                estimated=usage.estimated,
                profile=active_profile,
                notes=usage.notes,
            )
            record = LLMCallRecord(
                trace_id=trace,
                model=breakdown.model,
                module=module_name,
                status=status,
                latency_ms=latency_ms,
                at=at,
                usage=usage,
                breakdown=breakdown,
                error_type=type(error).__name__ if error is not None else None,
                error_message=str(error) if error is not None else None,
                metadata={"usage_source": usage.source},
            )

            log_fields: dict[str, Any] = {
                "trace_id": trace,
                "model": record.model,
                "module": module_name,
                "status": status,
                "latency_ms": round(latency_ms, 3),
                "cost": str(record.cost),
                "tier": breakdown.tier,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_hit_tokens": usage.cache_hit_tokens,
                "estimated": usage.estimated,
            }
            if error is not None:
                log_fields["error_type"] = type(error).__name__
                active_logger.warning("llm_call_failed", **log_fields)
            else:
                active_logger.info("llm_call", **log_fields)

            if recorder is not None:
                try:
                    recorder(record)
                except Exception as exc:  # noqa: BLE001 - 旁路回调不能影响业务
                    active_logger.warning("recorder_error", trace_id=trace, error=repr(exc))
            if ledger is not None:
                try:
                    ledger.add_entry(
                        trace,
                        record.cost,
                        dimension=record.model,
                        # module 作为第二个维度轴：同一笔金额会按 model 与 module 各写一份分账，
                        # 因此「按模块汇总之和」也等于总账，不需要事后配平。
                        dimensions={"module": module_name},
                        at=at,
                        tokens={
                            "cache_hit": usage.cache_hit_tokens,
                            "cache_miss": usage.cache_miss_tokens,
                            "output": usage.output_tokens,
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - 埋点写入失败不影响业务
                    active_logger.warning("ledger_write_failed", trace_id=trace, error=repr(exc))

        def _resolve_model(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
            """装饰器参数 → 调用关键字 → 函数签名默认值 → "unknown"。"""
            resolved = _extract_model(args, kwargs, model)
            if resolved == "unknown" and signature_model:
                return signature_model
            return resolved

        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                trace = parse_trace_id(args, kwargs, explicit=trace_id)
                model_name = _resolve_model(args, kwargs)
                module_name = module or _default_module()
                started = clock()
                at = now()
                try:
                    result = await target(*args, **kwargs)
                except Exception as exc:
                    latency_ms = (clock() - started) * 1000.0
                    # 失败也要记账：这是旧版最严重的漏记（异常时埋点代码根本不执行）
                    _emit(
                        trace,
                        model_name,
                        module_name,
                        "error",
                        latency_ms,
                        at,
                        _build_usage(None, kwargs, exc, args),
                        exc,
                    )
                    raise
                latency_ms = (clock() - started) * 1000.0
                _emit(
                    trace,
                    model_name,
                    module_name,
                    "ok",
                    latency_ms,
                    at,
                    _build_usage(result, kwargs, None, args),
                    None,
                )
                return result

            async_wrapper.__costgov_async__ = True  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(target)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = parse_trace_id(args, kwargs, explicit=trace_id)
            model_name = _resolve_model(args, kwargs)
            module_name = module or _default_module()
            started = clock()
            at = now()
            try:
                result = target(*args, **kwargs)
            except Exception as exc:
                latency_ms = (clock() - started) * 1000.0
                _emit(
                    trace,
                    model_name,
                    module_name,
                    "error",
                    latency_ms,
                    at,
                    _build_usage(None, kwargs, exc, args),
                    exc,
                )
                raise
            latency_ms = (clock() - started) * 1000.0
            _emit(
                trace,
                model_name,
                module_name,
                "ok",
                latency_ms,
                at,
                _build_usage(result, kwargs, None, args),
                None,
            )
            return result

        return sync_wrapper

    if func is not None:
        # 直接当装饰器用：@track_llm
        return decorate(func)
    # 带参数用：@track_llm(model="...")
    return decorate
