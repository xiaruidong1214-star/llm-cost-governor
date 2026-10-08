"""可选的 Celery 任务（写入路径的异步化）。

关键约束：**导入期不连接任何 broker**。
``build_celery_app()`` 在 ``settings.celery_broker_url`` 为空时直接返回 ``None``，
只有在真正需要投递任务时才构造 Celery 应用。

旧版的问题：Celery 任务用 ``max_retries=3``，而每次重试都会重新执行
``hincrbyfloat(COST_KEY, field, cost)``，同一次逻辑调用的成本被重复累加 3 次，
且没有任何 ``trace_id`` 可以把重试归并回同一个逻辑请求。

本模块的修法：
``trace_id`` 作为任务的**第一个参数**，重试时原样复用；
入账走 :class:`~costgovernor.storage.IdempotentLedger` 的 Lua 幂等脚本，
因此「重试 N 次」在账本上等价于「只入账 1 次」。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any

from .bootstrap import Container
from .cost import compute_cost
from .settings import Settings, get_settings
from .tracker import get_logger

__all__ = ["build_celery_app", "record_llm_usage", "compute_and_record"]

_logger = get_logger("costgovernor.tasks")


def build_celery_app(settings: Settings | None = None) -> Any | None:
    """构造 Celery 应用；未配置 broker 时返回 ``None``。

    **不会**在调用时连接 broker（``Celery(...)`` 是惰性的），
    更不会在模块导入期执行任何连接操作。
    """
    active = settings or get_settings()
    if not active.celery_broker_url:
        _logger.info(
            "celery_disabled",
            reason="未配置 CG_CELERY_BROKER_URL，异步写入路径关闭（同步写入仍然可用）",
        )
        return None
    try:
        from celery import Celery
    except ImportError:  # pragma: no cover - worker extra 未安装
        _logger.warning(
            "celery_missing",
            hint="需要 celery：pip install -e '.[worker]'",
        )
        return None

    app = Celery(
        "costgovernor",
        broker=active.celery_broker_url,
        backend=active.celery_result_backend or None,
    )
    app.conf.update(
        task_default_queue=active.celery_queue,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        # 重试要复用 trace_id，所以重试本身是安全的（幂等入账）
        task_default_retry_delay=2,
    )
    return app


def compute_and_record(
    trace_id: str,
    model: str,
    usage: Mapping[str, int],
    *,
    at: datetime | None = None,
    container: Container | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """纯执行逻辑：算成本并幂等入账。任务函数与直接调用共用这一份。

    单独抽出来是为了在没有 Celery 的环境里也能端到端测试入账路径。
    """
    active_settings = settings or (container.settings if container else get_settings())
    active_container = container or Container(settings=active_settings)
    moment = at or datetime.now()

    breakdown = compute_cost(
        model,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cache_hit_tokens=int(usage.get("cache_hit_tokens", 0)),
        cache_miss_tokens=usage.get("cache_miss_tokens"),
        at=moment,
    )
    result = active_container.ledger.add_entry(
        trace_id,
        breakdown.total,
        dimension=breakdown.model,
        at=moment,
        tokens={
            "cache_hit": breakdown.cache_hit_tokens,
            "cache_miss": breakdown.cache_miss_tokens,
            "output": breakdown.output_tokens,
        },
    )
    if result.duplicate:
        _logger.info("celery_task_duplicate", trace_id=trace_id, model=breakdown.model)
    return {
        "trace_id": trace_id,
        "model": breakdown.model,
        "cost": str(breakdown.total),
        "tier": breakdown.tier,
        "duplicate": result.duplicate,
        "total": str(result.total),
    }


def record_llm_usage(
    self: Any,
    trace_id: str,
    model: str,
    usage: Mapping[str, int],
    at_iso: str | None = None,
) -> dict[str, Any]:
    """Celery 任务体：记录一次 LLM 调用的成本。

    ``self`` 是 Celery 绑定的任务实例（``bind=True``）。

    重试语义：``max_retries`` 表示**额外重试次数上限**（``max_retries=3`` ⇒ 最多执行 4 次），
    每次重试都带同一个 ``trace_id``，因此账本上只累加一次。
    旧版把这点写错成「重复累加 3 次」，本模块用 ``trace_id`` + Lua 幂等脚本修正。

    注意：本函数**不导入 celery**，因此在没有安装 celery（或没配置 broker）的环境里
    也能用裸 ``self`` 替身直接调用测试。
    """
    moment = datetime.fromisoformat(at_iso) if at_iso else datetime.now()
    try:
        return compute_and_record(trace_id, model, usage, at=moment)
    except Exception as exc:
        retries = getattr(getattr(self, "request", None), "retries", 0)
        max_retries = int(getattr(self, "max_retries", 3) or 3)
        if retries >= max_retries:
            _logger.warning(
                "celery_task_giving_up",
                trace_id=trace_id,
                model=model,
                retries=retries,
                error=repr(exc),
            )
            try:  # pragma: no cover - 只有装了 celery 的 worker 才会走到
                from celery.exceptions import Ignore

                raise Ignore() from exc
            except ImportError:
                raise
        _logger.warning(
            "celery_task_retry",
            trace_id=trace_id,
            model=model,
            attempt=retries + 1,
            error=repr(exc),
        )
        # 重试仍然复用同一个 trace_id，保证幂等
        return self.retry(exc=exc, args=(trace_id, model, usage, at_iso))


def make_celery_task(app: Any) -> Any:
    """把 :func:`record_llm_usage` 注册成 Celery 任务。

    只有真的需要 worker 时才调用（例如在部署时的 ``worker.py`` 里），
    避免模块导入期就构造 Celery 应用。
    """
    if app is None:
        return None
    return app.task(name="costgovernor.record_llm_usage", bind=True, max_retries=3)(record_llm_usage)


def total_of(amounts: Any) -> Decimal:
    """把一串金额汇总成 Decimal（避免浮点累加误差）。"""
    return sum((Decimal(str(item)) for item in amounts), Decimal(0))
