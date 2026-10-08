"""命令行入口。

子命令：

* ``report``      —— 按天/按维度的成本报表（含分位数）
* ``outliers``    —— MAD 离群检测（样本不足时明确说明）
* ``retries``     —— 滑动窗口无效重试识别（窗口秒数与次数阈值分开传）
* ``reconcile``   —— 与供应商账单对账
* ``price-check`` —— 打印价格表与指定时刻的时段判定（**纯函数，不需要 Redis**）

设计要点：

* ``--client {auto,redis,fake}`` 决定用真实 Redis、自动探测还是内存替身；
  默认 ``auto``：能连上就用 Redis，连不上就明确告知并给出可用命令，绝不静默成功。
* ``price-check`` 完全不碰存储，因此在没有任何 Redis 的环境里也能跑（CI 里就是这么用的）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time as time_module
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from . import __version__
from .analytics import LedgerAnalytics, WindowAnalytics
from .bootstrap import Container
from .pricing import (
    DEFAULT_PRICING,
    MODEL_PRICING,
    PRICE_CHECKED_AT,
    PRICE_SOURCE_URL,
    PRICE_UNIT,
    price_for,
)
from .settings import Settings, get_settings
from .storage import IdempotentLedger, SlidingWindowCounter, utc_day

__all__ = ["main", "build_parser"]


# ======================================================================================
# 辅助
# ======================================================================================
def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD，收到 {value!r}") from exc


def _parse_moment(value: str) -> datetime:
    """解析 ``YYYY-MM-DD HH:MM`` / ``YYYY-MM-DDTHH:MM:SS`` / ``YYYY-MM-DD``。"""
    cleaned = value.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"时刻格式应为 'YYYY-MM-DD[ HH:MM[:SS]]'，收到 {value!r}"
    )


def _resolve_days(args: argparse.Namespace, analytics: LedgerAnalytics) -> list[date]:
    """确定要报告哪些天。"""
    if args.day:
        return [args.day]
    if args.days:
        return [utc_day() - timedelta(days=offset) for offset in range(args.days)]
    days = analytics.discover_days()
    return days


def _make_settings(args: argparse.Namespace) -> Settings:
    """按命令行参数覆盖配置（不修改环境变量，避免污染宿主进程）。"""
    overrides: dict[str, Any] = {}
    if args.redis_url:
        overrides["redis_url"] = args.redis_url
    if args.key_prefix:
        overrides["key_prefix"] = args.key_prefix
    if getattr(args, "window_seconds", None):
        overrides["retry_window_seconds"] = args.window_seconds
    if getattr(args, "max_calls", None):
        overrides["retry_max_calls"] = args.max_calls
    if getattr(args, "threshold", None):
        overrides["mad_threshold"] = args.threshold
    if getattr(args, "min_sample_size", None):
        overrides["min_sample_size"] = args.min_sample_size
    if not overrides:
        return get_settings()
    return Settings(**overrides)


def _build_client(args: argparse.Namespace, settings: Settings) -> tuple[Any | None, str]:
    """构造存储客户端。

    :return: ``(client, mode)``；``client`` 为 ``None`` 表示无法连接。

    ``fake`` 用内置的 :class:`~costgovernor.testing.InMemoryRedis`，
    它按本项目 Lua 脚本的语义执行，因此 ``--demo`` 之类的演示能给出**真实一致**的结果。
    如果退回 ``fakeredis``：**必须先确认它支持 EVAL**（fakeredis 需要 lupa 才有 EVAL 能力），
    否则幂等入账与滑动窗口都会静默失效、报表全是 0 —— 这比直接报错危险得多。
    """
    from .testing import InMemoryRedis

    mode = args.client
    if mode == "fake":
        return InMemoryRedis(), "fake"

    if mode == "redis":
        container = Container(settings=settings)
        if container.ping():
            return container.redis, "redis"
        print(
            f"❌ 无法连接 Redis：{settings.redis_url}\n"
            "   请确认服务已启动（docker compose up -d redis），"
            "或改用 --client fake 在内存里演练。",
            file=sys.stderr,
        )
        return None, "unavailable"

    # auto：先探活真实 Redis，连不上再退回内存替身（并明确告知）
    container = Container(settings=settings)
    if container.ping():
        return container.redis, "redis"

    fallback = InMemoryRedis()
    print(
        f"⚠️  未能连接 {settings.redis_url}，已改用内置内存替身演示；结果不具备持久性，"
        "且等价于 --client fake。",
        file=sys.stderr,
    )
    return fallback, "fake"


def _seed_demo(
    ledger: IdempotentLedger,
    *,
    days: int,
    outliers: bool = False,
    extra_models: int = 0,
) -> int:
    """往账本里写一批可复现的演示数据，返回写入条数。

    * 基础数据：两个官方模型，每天各 3 条，金额 0.2 ~ 0.8 元；
    * ``extra_models``：再补 N 个「分组维度值」（报表按模型分组，因此分组数 = 样本数），
      用于把 MAD 检测的样本量凑到阈值以上；
    * ``outliers=True``：额外塞一条 500 元的离群记录。

    演示数据的 trace_id 是确定性的，因此重复执行不会重复累加（这正是幂等入账的体现）。
    """
    written = 0
    today = utc_day()
    models = ["deepseek-flash", "deepseek-v4-pro"]
    models.extend(f"internal-model-{index:02d}" for index in range(extra_models))

    for offset in range(days):
        day = today - timedelta(days=offset)
        at = datetime(day.year, day.month, day.day, 11, 0, 0)
        for model_index, model in enumerate(models):
            # 每个模型每天 3 条；金额在 0.2 ~ 0.9 元之间小幅抖动，模拟真实分布
            for index in range(3):
                amount = Decimal("0.2") + Decimal(index) * Decimal("0.3") + Decimal(
                    (model_index % 4) * 0.01
                ).quantize(Decimal("0.01"))
                ledger.add_entry(
                    f"demo-{model}-{day.isoformat()}-{index}",
                    amount,
                    dimension=model,
                    at=at,
                    tokens={
                        "cache_hit": 1000 * (index + 1),
                        "cache_miss": 500 * (index + 1),
                        "output": 200 * (index + 1),
                    },
                )
                written += 1

    if outliers:
        # 一条明显的离群记录（500 元），用于演示 MAD 检测。
        # 用独立的维度值，避免把「离群点」混进正常模型的分账里。
        at = datetime(today.year, today.month, today.day, 12, 0, 0)
        ledger.add_entry(
            f"demo-outlier-{today.isoformat()}",
            Decimal("500"),
            dimension="runaway-batch-job",
            at=at,
            tokens={"cache_hit": 0, "cache_miss": 5_000_000, "output": 1_000_000},
        )
        written += 1
    return written


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _note(message: str, *, json_mode: bool = False) -> None:
    """人类可读的提示信息。

    ``--json`` 模式下写到 stderr，保证 stdout 始终是可直接 ``json.loads`` 的纯 JSON。
    """
    print(message, file=sys.stderr if json_mode else sys.stdout)


# ======================================================================================
# 子命令实现
# ======================================================================================
def cmd_report(args: argparse.Namespace) -> int:
    settings = _make_settings(args)
    client, mode = _build_client(args, settings)
    if client is None:
        return 2
    ledger = IdempotentLedger(client, settings)
    if args.demo:
        written = _seed_demo(ledger, days=args.days or 1, outliers=False)
        _note(f"已写入 {written} 条演示数据（幂等：重复执行不会重复累加）", json_mode=args.json)
    analytics = LedgerAnalytics(ledger, settings)
    days = _resolve_days(args, analytics)
    if not days:
        _note("没有找到任何计量数据。可加 --demo 生成演示数据，或用 --day YYYY-MM-DD 指定某天。", json_mode=args.json)
        return 0

    overview = analytics.overview(days=days)
    reports = [analytics.daily_report(day) for day in days]

    if args.json:
        _print_json(
            {
                "mode": mode,
                "unit": PRICE_UNIT,
                "overview": {
                    **overview,
                    "total": str(overview["total"]),
                    "by_axis": {
                        axis: {k: str(v) for k, v in values.items()}
                        for axis, values in overview["by_axis"].items()
                    },
                    "tokens": {k: str(v) for k, v in overview["tokens"].items()},
                },
                "per_day": [report.to_dict() for report in reports],
                "split_check": analytics.split_check(days).summary(),
            }
        )
        return 0

    for report in reports:
        print(report.render())
        print()
    print("—— 汇总 ——")
    print(f"  天数：{len(days)}    分组条目数：{overview['count']}")
    print(f"  总成本：{overview['total']} {overview['currency']}")
    for axis, values in overview["by_axis"].items():
        print(f"  按 {axis} 汇总：")
        for name, value in sorted(values.items(), key=lambda kv: -kv[1]):
            print(f"    - {name}: {value} {overview['currency']}")
    if overview["tokens"]:
        print("  token 合计：" + ", ".join(f"{k}={v}" for k, v in sorted(overview["tokens"].items())))
    print(f"  {analytics.split_check(days).summary()}")
    return 0


def cmd_outliers(args: argparse.Namespace) -> int:
    settings = _make_settings(args)
    client, mode = _build_client(args, settings)
    if client is None:
        return 2
    ledger = IdempotentLedger(client, settings)
    if args.demo:
        written = _seed_demo(
            ledger,
            days=max(args.days or 1, 1),
            outliers=True,
            extra_models=args.demo_models,
        )
        _note(f"已写入 {written} 条演示数据（{args.demo_models + 2} 个分组 + 1 个离群点）", json_mode=args.json)
    analytics = LedgerAnalytics(ledger, settings)
    days = _resolve_days(args, analytics)
    if not days:
        _note("没有找到任何计量数据。可加 --demo 生成演示数据。", json_mode=args.json)
        return 0

    exit_code = 0
    payload: list[dict[str, Any]] = []
    for day in days:
        result = analytics.outliers(day, threshold=args.threshold, min_sample_size=args.min_sample_size)
        if result.enough:
            assert hasattr(result, "outliers")
            summary = result.to_dict()  # type: ignore[union-attr]
            payload.append({"day": day.isoformat(), **summary})
            if not args.json:
                print(
                    f"{day.isoformat()}：样本 {result.sample_size}，中位数 {result.median:.6f}，"
                    f"MAD {result.mad:.6f}，阈值 {result.threshold}，"
                    f"离群 {result.outlier_count} 个"
                )
                for item in result.outliers:  # type: ignore[union-attr]
                    print(
                        f"    - 第 {item.index} 条：{item.value:.6f} "
                        f"（modified z={item.modified_z:.3f}，{item.direction}）"
                    )
                if result.note:  # type: ignore[union-attr]
                    print(f"    备注：{result.note}")
        else:
            payload.append(
                {
                    "day": day.isoformat(),
                    "insufficient_sample": True,
                    "sample_size": result.sample_size,
                    "required": result.required,
                    "reason": result.reason,
                }
            )
            if not args.json:
                print(f"{day.isoformat()}：{result}")
            if args.strict:
                exit_code = 1
    if args.json:
        _print_json({"mode": mode, "results": payload})
    return exit_code


def cmd_retries(args: argparse.Namespace) -> int:
    settings = _make_settings(args)
    client, mode = _build_client(args, settings)
    if client is None:
        return 2

    window_seconds = args.window_seconds or settings.retry_window_seconds
    max_calls = args.max_calls or settings.retry_max_calls

    if args.demo:
        # 演示：把 N 次调用排在最近 N 秒内（而不是历史上某个固定时刻），
        # 这样「当前时间 − 窗口」的默认查询就能看到它们，演示才有意义。
        counter = SlidingWindowCounter(client, settings, scope="demo")
        now = time_module.time()
        for index in range(args.demo):
            counter.record_and_count(
                f"demo-trace-{index:03d}",
                now=now - (args.demo - 1 - index),
                window_seconds=window_seconds,
                max_calls=max_calls,
            )
        _note(f"已写入 {args.demo} 次演示调用（最近 {args.demo} 秒内）", json_mode=args.json)
    else:
        counter = SlidingWindowCounter(client, settings, scope=args.scope)
        now = None

    analytics = WindowAnalytics(counter, settings)
    status = analytics.status(now=now, window_seconds=window_seconds, max_calls=max_calls)
    members = counter.members(now=now, window_seconds=window_seconds, limit=args.limit)

    if args.json:
        _print_json({"mode": mode, **status, "members": members})
    else:
        print(f"作用域：{status['scope']}（存储后端：{mode}）")
        print(f"时间窗口：{status['window_seconds']} 秒    {status['max_calls']} 次")
        print(f"窗口内调用次数：{status['count']}")
        print(f"是否超阈值：{'是（疑似无效重试）' if status['exceeded'] else '否'}")
        if members:
            print(f"窗口内成员（最多显示 {args.limit} 个）：")
            for member in members:
                print(f"    - {member}")

    exit_code = 1 if (status["exceeded"] and args.strict) else 0
    return exit_code


def cmd_reconcile(args: argparse.Namespace) -> int:
    settings = _make_settings(args)
    client, mode = _build_client(args, settings)
    if client is None:
        return 2
    ledger = IdempotentLedger(client, settings)
    if args.demo:
        written = _seed_demo(ledger, days=args.days or 1, outliers=False)
        _note(f"已写入 {written} 条演示数据", json_mode=args.json)
    analytics = LedgerAnalytics(ledger, settings)
    days = _resolve_days(args, analytics)
    result = analytics.reconcile(args.bill_total, days=days, tolerance=args.tolerance)

    if args.json:
        _print_json({"mode": mode, **result.to_dict()})
    else:
        print(f"存储后端：{mode}")
        print(result.summary())
        if result.note:
            print(f"备注：{result.note}")
    return 1 if (result.mismatch and args.strict) else 0


def cmd_price_check(args: argparse.Namespace) -> int:
    """价格表自检：纯函数，不需要 Redis。"""
    at = args.at or datetime.now()
    models = [args.model] if args.model else sorted(MODEL_PRICING)
    payload: dict[str, Any] = {
        "unit": PRICE_UNIT,
        "source": PRICE_SOURCE_URL,
        "checked_at": PRICE_CHECKED_AT,
        "at": at.isoformat(),
        "tier": DEFAULT_PRICING.tier_for(at),
        "tier_explain": DEFAULT_PRICING.describe_tier(at),
        "models": [],
    }

    for model in models:
        try:
            pricing = DEFAULT_PRICING.pricing_for(model, at)
        except Exception as exc:  # noqa: BLE001 - 未知模型/未生效都要给出可读提示
            payload["models"].append({"model": model, "error": str(exc)})
            if not args.json:
                print(f"❌ {model}: {exc}")
            continue
        entry = {
            "model": model,
            "effective_date": pricing.effective_date,
            "input_cache_hit": str(
                price_for(model, at, kind="input", cache_tier="cache_hit")
            ),
            "input_cache_miss": str(
                price_for(model, at, kind="input", cache_tier="cache_miss")
            ),
            "output": str(price_for(model, at, kind="output", cache_tier="cache_miss")),
        }
        payload["models"].append(entry)
        if not args.json:
            print(
                f"{model}：输入·缓存命中 {entry['input_cache_hit']}，"
                f"输入·缓存未命中 {entry['input_cache_miss']}，"
                f"输出 {entry['output']}（{PRICE_UNIT}，生效日期 {entry['effective_date']}）"
            )

    if args.json:
        _print_json(payload)
    else:
        print(f"时段判定：{payload['tier_explain']}")
        print(f"单位：{PRICE_UNIT}")
        print(f"价格来源：{PRICE_SOURCE_URL}（核对日期 {PRICE_CHECKED_AT}）")
    return 0


# ======================================================================================
# 参数解析
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="costgov",
        description="llm-cost-governor：LLM 成本治理与可观测性命令行工具",
    )
    parser.add_argument("--version", action="version", version=f"llm-cost-governor {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--redis-url", default=None, help="Redis 连接串（默认取 CG_REDIS_URL）")
        sub.add_argument("--key-prefix", default=None, help="key 前缀（默认取 CG_KEY_PREFIX，即 llm）")
        sub.add_argument(
            "--client",
            choices=("auto", "redis", "fake"),
            default="auto",
            help="存储后端：auto=能连就用 Redis，否则退回内存替身；fake=强制内存替身",
        )
        sub.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本消费）")

    report = subparsers.add_parser("report", help="成本报表（按天/按维度 + 分位数）")
    add_common(report)
    report.add_argument("--day", type=_parse_day, default=None, help="只报告某一天（YYYY-MM-DD）")
    report.add_argument("--days", type=int, default=None, help="报告最近 N 天")
    report.add_argument("--demo", action="store_true", help="先写入一批演示数据（幂等）")
    report.set_defaults(func=cmd_report)

    outliers = subparsers.add_parser("outliers", help="MAD 离群检测")
    add_common(outliers)
    outliers.add_argument("--day", type=_parse_day, default=None, help="只检测某一天")
    outliers.add_argument("--days", type=int, default=None, help="检测最近 N 天")
    outliers.add_argument("--threshold", type=float, default=None, help="modified z-score 阈值（默认 3.5）")
    outliers.add_argument("--min-sample-size", type=int, default=None, help="最小样本量（默认 30）")
    outliers.add_argument("--demo", action="store_true", help="写入含离群点的演示数据")
    outliers.add_argument(
        "--demo-models",
        type=int,
        default=40,
        help="演示数据里额外补多少个分组（凑够 MAD 的最小样本量，默认 40）",
    )
    outliers.add_argument("--strict", action="store_true", help="样本不足时以退出码 1 结束")
    outliers.set_defaults(func=cmd_outliers)

    retries = subparsers.add_parser("retries", help="滑动窗口无效重试识别")
    add_common(retries)
    retries.add_argument("--scope", default="llm", help="窗口作用域（例如模型名或模块名）")
    retries.add_argument("--window-seconds", type=int, default=None, help="时间窗口（秒），默认 60")
    retries.add_argument("--max-calls", type=int, default=None, help="窗口内次数阈值，默认 8")
    retries.add_argument("--limit", type=int, default=20, help="最多显示多少个窗口成员")
    retries.add_argument("--demo", type=int, default=0, metavar="N", help="写入 N 次演示调用")
    retries.add_argument("--strict", action="store_true", help="超阈值时以退出码 1 结束")
    retries.set_defaults(func=cmd_retries)

    reconcile = subparsers.add_parser("reconcile", help="与供应商账单对账")
    add_common(reconcile)
    reconcile.add_argument("bill_total", type=str, help="账单总额（元）")
    reconcile.add_argument("--day", type=_parse_day, default=None)
    reconcile.add_argument("--days", type=int, default=None)
    reconcile.add_argument("--tolerance", type=str, default=None, help="允许的相对差异，默认 0.01")
    reconcile.add_argument("--demo", action="store_true", help="先写入一批演示数据")
    reconcile.add_argument("--strict", action="store_true", help="不一致时以退出码 1 结束")
    reconcile.set_defaults(func=cmd_reconcile)

    price_check = subparsers.add_parser("price-check", help="价格表自检（纯函数，无需 Redis）")
    price_check.add_argument("--model", default=None, help="只检查某个模型；未知模型会被明确指出")
    price_check.add_argument("--at", type=_parse_moment, default=None, help="按指定时刻判定高峰/空闲")
    price_check.add_argument("--json", action="store_true", help="以 JSON 输出")
    price_check.set_defaults(func=cmd_price_check)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("已中断", file=sys.stderr)
        return 130
    except ValueError as exc:
        # 参数语义错误（例如负容差、非法日期范围）统一变成可读提示 + 退出码 2，
        # 而不是抛一大段 traceback 给运维看。
        print(f"❌ 参数错误：{exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"❌ 数据缺失：{exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
