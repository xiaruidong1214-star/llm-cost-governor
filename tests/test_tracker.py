"""``@track_llm`` 装饰器回归测试。

针对旧版缺陷 6 的三条硬性要求：

1. 异步函数必须能用（旧版对协程调用 ``.get`` 直接 ``AttributeError``）；
2. **失败调用必须被记录**（旧版异常时埋点代码根本不执行，失败全漏记）；
3. ``usage`` 解析要健壮（dict / 带 ``.usage`` 的对象 / 拿不到时标 ``estimated=True``）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest

from costgovernor.pricing import UnknownModelError
from costgovernor.storage import IdempotentLedger
from costgovernor.tracker import (
    CHARS_PER_TOKEN,
    UsageInfo,
    UsageParseError,
    estimate_tokens_from_text,
    extract_usage,
    get_logger,
    parse_trace_id,
    structlog_configure,
    track_llm,
)

PEAK = datetime(2026, 10, 8, 10, 0, 0)  # 周四 10:00（高峰）


# ======================================================================================
# 测试替身
# ======================================================================================
@dataclass
class FakeUsage:
    """模拟 SDK 的 usage 对象（OpenAI 风格）。"""

    prompt_tokens: int = 1_000_000
    completion_tokens: int = 500_000
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None


@dataclass
class FakeResponse:
    """模拟 SDK 的响应对象。"""

    text: str = "你好"
    usage: Any = None


class RecordingLogger:
    """捕获日志调用的替身，用来断言「失败也被记了日志」。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def _record(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    info = _record
    debug = _record
    warning = _record
    error = _record

    def event_names(self) -> list[str]:
        return [name for name, _ in self.events]

    def find(self, name: str) -> dict[str, Any] | None:
        for event, fields in self.events:
            if event == name:
                return fields
        return None


@pytest.fixture
def recorder() -> list[Any]:
    return []


@pytest.fixture
def logger() -> RecordingLogger:
    return RecordingLogger()


# ======================================================================================
# usage 解析
# ======================================================================================
def test_extract_usage_from_openai_dict() -> None:
    usage = extract_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 3}})
    assert usage.input_tokens == 10
    assert usage.output_tokens == 3
    assert usage.cache_hit_tokens == 0
    assert usage.cache_miss_tokens == 10
    assert usage.estimated is False
    assert "mapping['usage']" in usage.source


def test_extract_usage_from_object_with_usage_attribute() -> None:
    """OpenAI / DeepSeek SDK 返回的是对象，不是 dict（旧版直接下标取值会崩）。"""
    usage = extract_usage(FakeResponse(usage=FakeUsage()))
    assert usage.input_tokens == 1_000_000
    assert usage.output_tokens == 500_000
    assert usage.estimated is False


def test_extract_usage_with_explicit_cache_split() -> None:
    usage = extract_usage(
        {
            "usage": {
                "input_tokens": 1_000_000,
                "output_tokens": 100_000,
                "prompt_cache_hit_tokens": 750_000,
                "prompt_cache_miss_tokens": 250_000,
            }
        }
    )
    assert usage.cache_hit_tokens == 750_000
    assert usage.cache_miss_tokens == 250_000


def test_extract_usage_derives_miss_from_hit() -> None:
    usage = extract_usage({"usage": {"input_tokens": 1000, "prompt_cache_hit_tokens": 900}})
    assert usage.cache_hit_tokens == 900
    assert usage.cache_miss_tokens == 100
    assert any("推算" in note for note in usage.notes)


def test_extract_usage_accepts_container_directly() -> None:
    usage = extract_usage({"input_tokens": 5, "output_tokens": 6})
    assert (usage.input_tokens, usage.output_tokens) == (5, 6)


def test_extract_usage_from_nested_response() -> None:
    usage = extract_usage({"response": {"usage": {"input_tokens": 7, "output_tokens": 1}}})
    assert usage.input_tokens == 7


def test_extract_usage_raises_when_unparseable() -> None:
    with pytest.raises(UsageParseError):
        extract_usage({"no_usage": True})
    with pytest.raises(UsageParseError):
        extract_usage(None)


def test_extract_usage_reads_extra_sources() -> None:
    """装饰器会把异常对象上的 usage、以及 ``usage=`` 关键字参数作为额外来源传入。"""

    class Boom(Exception):
        usage = {"input_tokens": 3, "output_tokens": 4}

    usage = extract_usage(None, Boom().usage)
    assert usage.input_tokens == 3


def test_estimate_tokens_and_usage_info_helpers() -> None:
    assert estimate_tokens_from_text("") == 0
    assert estimate_tokens_from_text("a" * 40) == int(40 / CHARS_PER_TOKEN)
    info = UsageInfo(input_tokens=1, output_tokens=2, estimated=True)
    assert info.total_tokens == 3
    assert info.to_dict()["estimated"] is True


def test_parse_trace_id_priority() -> None:
    assert parse_trace_id((), {"trace_id": "from-kwargs"}) == "from-kwargs"
    assert parse_trace_id((), {"request_id": "req-1"}) == "req-1"
    assert parse_trace_id((), {}, explicit="explicit") == "explicit"
    assert parse_trace_id(("trace-in-args",), {}).startswith("trace-")
    assert parse_trace_id((), {}) != parse_trace_id((), {})  # 每次生成新的


# ======================================================================================
# 同步装饰
# ======================================================================================
def test_sync_success_is_recorded(recorder: list[Any], logger: RecordingLogger) -> None:
    @track_llm(
        model="deepseek-flash",
        module="demo",
        recorder=recorder.append,
        logger=logger,
        now=lambda: PEAK,
    )
    def ask(prompt: str) -> dict[str, Any]:
        return {"text": "hi", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}

    result = ask("你好")
    assert result["text"] == "hi"  # 返回值不被装饰器改变
    assert len(recorder) == 1
    record = recorder[0]
    assert record.status == "ok"
    assert record.model == "deepseek-flash"
    assert record.module == "demo"
    # 1M 输入 × 高峰 2 元/M = 2 元
    assert record.cost == Decimal("2.0")
    assert record.latency_ms >= 0
    assert logger.find("llm_call") is not None


def test_sync_exception_is_reraised_and_recorded(
    recorder: list[Any], logger: RecordingLogger
) -> None:
    """缺陷 6 的核心回归：函数抛异常时埋点**必须**执行，且异常原样抛出。"""

    class Boom(RuntimeError):
        pass

    @track_llm(
        model="deepseek-flash",
        module="failing",
        recorder=recorder.append,
        logger=logger,
        now=lambda: PEAK,
    )
    def ask(prompt: str) -> dict[str, Any]:
        raise Boom("上游 500")

    with pytest.raises(Boom, match="上游 500"):
        ask("你好")

    assert len(recorder) == 1  # 旧版这里会是 0
    record = recorder[0]
    assert record.status == "error"
    assert record.error_type == "Boom"
    assert record.error_message == "上游 500"
    assert record.model == "deepseek-flash"
    failed = logger.find("llm_call_failed")
    assert failed is not None
    assert failed["status"] == "error"
    assert failed["error_type"] == "Boom"


def test_failed_call_still_costs_when_usage_is_available(recorder: list[Any]) -> None:
    """失败调用若响应里带 usage，也要按 usage 计价（失败不等于免费）。"""

    class Boom(Exception):
        usage = {"input_tokens": 1_000_000, "output_tokens": 0}

    @track_llm(model="deepseek-flash", recorder=recorder.append, now=lambda: PEAK)
    def ask(prompt: str) -> None:
        raise Boom("超时")

    with pytest.raises(Boom):
        ask("你好")
    assert recorder[0].status == "error"
    assert recorder[0].cost == Decimal("2.0")  # 1M × 2 元/M


def test_failed_call_without_usage_is_marked_estimated(recorder: list[Any]) -> None:
    @track_llm(model="deepseek-flash", recorder=recorder.append, now=lambda: PEAK)
    def ask(prompt: str) -> None:
        raise TimeoutError("读超时")

    with pytest.raises(TimeoutError):
        ask("a" * 400)
    record = recorder[0]
    assert record.usage.estimated is True
    assert record.usage.input_tokens == int(400 / CHARS_PER_TOKEN)
    assert any("不可用于账单对账" in note for note in record.usage.notes)


def test_usage_object_response(recorder: list[Any]) -> None:
    @track_llm(model="deepseek-v4-pro", recorder=recorder.append, now=lambda: PEAK)
    def ask(prompt: str) -> FakeResponse:
        return FakeResponse(usage=FakeUsage())

    ask("你好")
    record = recorder[0]
    assert record.usage.estimated is False
    # 1M 输入 × 9 元/M + 0.5M 输出 × 27 元/M = 9 + 13.5 = 22.5
    assert record.cost == Decimal("22.5")


def test_estimated_flag_when_usage_missing(recorder: list[Any]) -> None:
    @track_llm(model="deepseek-flash", recorder=recorder.append, now=lambda: PEAK)
    def ask(prompt: str) -> dict[str, Any]:
        return {"text": "回答"}

    ask("问题" * 100)
    record = recorder[0]
    assert record.usage.estimated is True
    assert record.breakdown.estimated is True
    assert record.to_dict()["estimated"] is True


def test_usage_keyword_argument_is_used(recorder: list[Any]) -> None:
    @track_llm(model="deepseek-flash", recorder=recorder.append, now=lambda: PEAK)
    def ask(prompt: str, usage: dict[str, int] | None = None) -> str:
        return "ok"

    ask("hi", usage={"input_tokens": 1_000_000, "output_tokens": 0})
    assert recorder[0].usage.estimated is False
    assert recorder[0].cost == Decimal("2.0")


def test_model_inferred_from_kwargs(recorder: list[Any]) -> None:
    @track_llm(recorder=recorder.append, logger=RecordingLogger(), now=lambda: PEAK)
    def ask(prompt: str, model: str = "deepseek-v4-pro") -> dict[str, Any]:
        return {"usage": {"input_tokens": 1_000_000}}

    ask("hi", model="deepseek-v4-pro")
    assert recorder[0].model == "deepseek-v4-pro"
    assert recorder[0].cost == Decimal("9.0")


def test_plain_decorator_without_parentheses() -> None:
    """``@track_llm`` 直接这样用也要能工作。"""

    @track_llm
    def ask(prompt: str, model: str = "deepseek-flash") -> dict[str, Any]:
        return {"usage": {"input_tokens": 1_000_000}}

    result = ask("hi")
    assert result["usage"]["input_tokens"] == 1_000_000
    assert ask.__name__ == "ask"  # functools.wraps 保住了元数据


def test_unknown_model_raises_instead_of_silent_default(recorder: list[Any]) -> None:
    """推断不出模型名时明确报错，而不是静默按默认价计费。"""

    @track_llm(recorder=recorder.append, logger=RecordingLogger(), now=lambda: PEAK)
    def ask(prompt: str) -> dict[str, Any]:
        return {"usage": {"input_tokens": 1}}

    with pytest.raises(UnknownModelError):
        ask("hi")


def test_trace_id_is_stable_across_retries(recorder: list[Any]) -> None:
    """显式传入的 trace_id 在重试间复用（幂等入账的前提）。"""

    @track_llm(
        model="deepseek-flash",
        recorder=recorder.append,
        logger=RecordingLogger(),
        trace_id="trace-fixed",
        now=lambda: PEAK,
    )
    def ask(prompt: str) -> dict[str, Any]:
        return {"usage": {"input_tokens": 1}}

    ask("hi")
    ask("hi")
    assert {record.trace_id for record in recorder} == {"trace-fixed"}


def test_trace_id_from_kwargs(recorder: list[Any]) -> None:
    @track_llm(model="deepseek-flash", recorder=recorder.append, logger=RecordingLogger(), now=lambda: PEAK)
    def ask(prompt: str, trace_id: str = "") -> dict[str, Any]:
        return {"usage": {"input_tokens": 1}}

    ask("hi", trace_id="trace-from-call")
    assert recorder[0].trace_id == "trace-from-call"


def test_ledger_writes_once_per_trace_id(recorder: list[Any], fake_redis, settings) -> None:
    """装饰器 + 幂等账本：重试三次只入账一次。"""
    ledger = IdempotentLedger(fake_redis, settings)

    @track_llm(
        model="deepseek-flash",
        ledger=ledger,
        recorder=recorder.append,
        logger=RecordingLogger(),
        trace_id="trace-retry",
        now=lambda: PEAK,
    )
    def ask(prompt: str) -> dict[str, Any]:
        return {"usage": {"input_tokens": 1_000_000}}

    for _ in range(3):
        ask("hi")
    assert len(recorder) == 3
    assert ledger.bucket_total(datetime(2026, 10, 8).date()) == Decimal("2.0")


def test_recorder_and_ledger_errors_do_not_break_business(recorder: list[Any], logger: RecordingLogger) -> None:
    """旁路逻辑（回调 / 存储）出问题不能影响业务返回值。"""

    def broken_recorder(record: Any) -> None:
        raise RuntimeError("回调炸了")

    class BrokenLedger:
        def add_entry(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("redis 炸了")

    @track_llm(
        model="deepseek-flash",
        recorder=broken_recorder,
        ledger=BrokenLedger(),
        logger=logger,
        now=lambda: PEAK,
    )
    def ask(prompt: str) -> str:
        return "ok"

    assert ask("hi") == "ok"
    assert logger.find("recorder_error") is not None
    assert logger.find("ledger_write_failed") is not None


def test_module_defaults_to_qualified_name(recorder: list[Any]) -> None:
    @track_llm(model="deepseek-flash", recorder=recorder.append, logger=RecordingLogger(), now=lambda: PEAK)
    def ask(prompt: str) -> dict[str, Any]:
        return {"usage": {"input_tokens": 1}}

    ask("hi")
    assert recorder[0].module.endswith("test_module_defaults_to_qualified_name.<locals>.ask")


# ======================================================================================
# 异步装饰
# ======================================================================================
async def test_async_function_is_supported(recorder: list[Any], logger: RecordingLogger) -> None:
    """缺陷 6 的另一半：async 函数必须能用（旧版会 AttributeError 并吞掉）。"""

    @track_llm(
        model="deepseek-flash",
        module="async-demo",
        recorder=recorder.append,
        logger=logger,
        now=lambda: PEAK,
    )
    async def ask(prompt: str) -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"text": "hi", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}

    # 装饰后仍然是协程函数（否则没法被 await / 被框架调度）
    import inspect

    assert inspect.iscoroutinefunction(ask)
    result = await ask("你好")
    assert result["text"] == "hi"
    assert len(recorder) == 1
    assert recorder[0].status == "ok"
    assert recorder[0].cost == Decimal("2.0")
    assert recorder[0].module == "async-demo"


async def test_async_exception_is_reraised_and_recorded(recorder: list[Any]) -> None:
    class Boom(ValueError):
        pass

    @track_llm(model="deepseek-flash", recorder=recorder.append, logger=RecordingLogger(), now=lambda: PEAK)
    async def ask(prompt: str) -> None:
        raise Boom("异步失败")

    with pytest.raises(Boom):
        await ask("你好")
    assert len(recorder) == 1
    assert recorder[0].status == "error"
    assert recorder[0].error_type == "Boom"


async def test_async_object_usage(recorder: list[Any]) -> None:
    @track_llm(model="deepseek-v4-pro", recorder=recorder.append, logger=RecordingLogger(), now=lambda: PEAK)
    async def ask(prompt: str) -> FakeResponse:
        return FakeResponse(usage=FakeUsage(prompt_tokens=1_000_000, completion_tokens=0))

    await ask("hi")
    assert recorder[0].cost == Decimal("9.0")


async def test_async_cancellation_is_not_swallowed(recorder: list[Any]) -> None:
    """``asyncio.CancelledError`` 在 3.8+ 继承自 BaseException，不能被当成业务异常吞掉。

    装饰器只捕获 ``Exception``，因此取消会直接向上传播；这是刻意行为。
    """

    @track_llm(model="deepseek-flash", recorder=recorder.append, logger=RecordingLogger(), now=lambda: PEAK)
    async def ask(prompt: str) -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await ask("hi")


# ======================================================================================
# 日志
# ======================================================================================
def test_logger_is_created_without_print() -> None:
    structlog_configure()
    logger = get_logger("costgovernor.test")
    assert hasattr(logger, "info")
    # 记录里不应该出现 print（静态检查：模块源码里没有 print 调用）
    import inspect

    from costgovernor import tracker

    source = inspect.getsource(tracker)
    assert "print(" not in source
