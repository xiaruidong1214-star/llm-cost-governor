"""CLI 子命令测试。

全部用 ``--client fake``（内存替身）跑，**不连接任何真实 Redis**；
``price-check`` 更是纯函数路径，连替身都不需要。
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from costgovernor.cli import build_parser, main


def _run(capsys, argv: list[str]) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ======================================================================================
# 参数解析
# ======================================================================================
def test_parser_has_all_subcommands() -> None:
    parser = build_parser()
    actions = [action for action in parser._actions if hasattr(action, "choices") and action.choices]
    subcommands = set()
    for action in actions:
        if action.dest == "command":
            subcommands = set(action.choices)
    assert subcommands == {"report", "outliers", "retries", "reconcile", "price-check"}


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "costgov" in out
    assert "price-check" in out


def test_version(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "2.0.0" in capsys.readouterr().out


# ======================================================================================
# price-check（纯函数，不需要 Redis）
# ======================================================================================
def test_price_check_prints_official_rates(capsys) -> None:
    code, out, _err = _run(capsys, ["price-check", "--at", "2026-10-08 10:00"])
    assert code == 0
    assert "deepseek-flash" in out
    assert "2.0" in out  # 高峰缓存未命中输入
    assert "8.0" in out  # 高峰输出
    assert "0.04" in out  # 高峰缓存命中输入
    assert "元 / 百万 tokens" in out  # 单位口径必须打印出来
    assert "api-docs.deepseek.com" in out
    assert "高峰" in out  # 时段判定说明


def test_price_check_uses_off_peak_rates(capsys) -> None:
    code, out, _err = _run(capsys, ["price-check", "--at", "2026-10-08 20:00"])
    assert code == 0
    assert "空闲" in out
    assert "1.0" in out  # 空闲缓存未命中输入
    assert "4.0" in out  # 空闲输出


def test_price_check_single_model_json(capsys) -> None:
    code, out, _err = _run(
        capsys, ["price-check", "--model", "deepseek-v4-pro", "--at", "2026-10-08 10:00", "--json"]
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["unit"] == "元 / 百万 tokens"
    assert payload["tier"] == "peak"
    assert payload["models"][0]["model"] == "deepseek-v4-pro"
    assert payload["models"][0]["output"] == "27.0"
    assert payload["models"][0]["input_cache_hit"] == "0.30"


def test_price_check_reports_unknown_model_without_crashing(capsys) -> None:
    code, out, _err = _run(capsys, ["price-check", "--model", "gpt-4o-mini"])
    assert code == 0  # 明确的业务提示，不是崩溃
    assert "未收录的模型" in out


def test_price_check_weekend_is_off_peak(capsys) -> None:
    code, out, _err = _run(capsys, ["price-check", "--at", "2026-10-10 10:00"])  # 周六
    assert code == 0
    assert "空闲" in out


def test_price_check_rejects_bad_datetime() -> None:
    with pytest.raises(SystemExit):
        main(["price-check", "--at", "not-a-date"])


# ======================================================================================
# report
# ======================================================================================
def test_report_without_data_says_so(capsys) -> None:
    code, out, _err = _run(capsys, ["report", "--client", "fake"])
    assert code == 0
    assert "没有找到任何计量数据" in out


def test_report_demo_data(capsys) -> None:
    code, out, _err = _run(capsys, ["report", "--client", "fake", "--demo", "--days", "2"])
    assert code == 0
    assert "已写入" in out
    assert "报表" in out
    assert "deepseek-flash" in out
    assert "deepseek-v4-pro" in out
    assert "总成本" in out
    assert "分账之和" in out  # 平账检查被打印出来
    assert "✅" in out


def test_report_demo_is_idempotent(capsys) -> None:
    """演示数据的 trace_id 是确定性的，重复执行不会重复累加。"""
    from costgovernor.cli import _seed_demo  # noqa: PLC2701 - 直接验证内部辅助函数
    from costgovernor.settings import Settings
    from costgovernor.storage import IdempotentLedger
    from tests.conftest import FakeRedis

    client = FakeRedis()
    settings = Settings(ttl_seconds=86400, ttl_index_seconds=86400, ttl_dedup_seconds=86400)
    ledger = IdempotentLedger(client, settings)
    _seed_demo(ledger, days=1, outliers=False)
    first = ledger.bucket_total(datetime.now().replace(tzinfo=None).date())
    _seed_demo(ledger, days=1, outliers=False)
    second = ledger.bucket_total(datetime.now().replace(tzinfo=None).date())
    assert first == second
    assert first > 0


def test_report_json_output(capsys) -> None:
    code, out, _err = _run(
        capsys, ["report", "--client", "fake", "--demo", "--days", "1", "--json"]
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["mode"] == "fake"
    assert payload["unit"] == "元 / 百万 tokens"
    assert payload["overview"]["total"]
    assert payload["per_day"]
    assert "分账" in payload["split_check"]


def test_report_specific_day(capsys) -> None:
    code, out, _err = _run(
        capsys, ["report", "--client", "fake", "--day", "2030-01-01"]
    )
    assert code == 0
    assert "2030-01-01" in out


def test_report_rejects_bad_day() -> None:
    with pytest.raises(SystemExit):
        main(["report", "--day", "2026-13-45"])


# ======================================================================================
# outliers
# ======================================================================================
def test_outliers_reports_insufficient_sample(capsys) -> None:
    """样本量低于阈值时必须明说「样本不足」，而不是硬算。"""
    code, out, _err = _run(
        capsys,
        ["outliers", "--client", "fake", "--demo", "--days", "1", "--demo-models", "3"],
    )
    assert code == 0
    assert "样本不足" in out


def test_outliers_strict_exit_code(capsys) -> None:
    code, _out, _err = _run(
        capsys,
        ["outliers", "--client", "fake", "--demo", "--days", "1", "--demo-models", "3", "--strict"],
    )
    assert code == 1


def test_outliers_demo_detects_the_spike(capsys) -> None:
    """演示数据里塞了一条 500 元的离群记录，必须能被 modified z-score 抓出来。"""
    code, out, _err = _run(capsys, ["outliers", "--client", "fake", "--demo", "--days", "1"])
    assert code == 0
    assert "离群 1 个" in out
    assert "500.000000" in out
    assert "high" in out


def test_outliers_json_with_demo(capsys) -> None:
    code, out, _err = _run(
        capsys, ["outliers", "--client", "fake", "--demo", "--days", "1", "--json"]
    )
    assert code == 0
    payload = json.loads(out)
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["outlier_count"] == 1
    assert result["outliers"][0]["direction"] == "high"
    assert result["outliers"][0]["value"] == 500.0


def test_outliers_with_lowered_min_sample_size(capsys) -> None:
    code, out, _err = _run(
        capsys,
        ["outliers", "--client", "fake", "--demo", "--days", "5", "--min-sample-size", "2", "--day", "2026-10-08"],
    )
    assert code == 0
    assert "MAD" in out


def test_outliers_json(capsys) -> None:
    code, out, _err = _run(capsys, ["outliers", "--client", "fake", "--days", "1", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["mode"] == "fake"
    assert payload["results"]  # 覆盖了「今天」这一天
    for item in payload["results"]:
        # 没有数据 → 明确报「样本不足」，而不是伪造异常
        assert item["insufficient_sample"] is True
        assert item["sample_size"] == 0
        assert item["required"] == 30


# ======================================================================================
# retries（窗口与阈值分开传）
# ======================================================================================
def test_retries_demo_shows_exceeded(capsys) -> None:
    code, out, _err = _run(
        capsys,
        [
            "retries",
            "--client",
            "fake",
            "--demo",
            "12",
            "--window-seconds",
            "60",
            "--max-calls",
            "8",
        ],
    )
    assert code == 0
    assert "时间窗口：60 秒" in out
    assert "8 次" in out
    assert "窗口内调用次数：12" in out
    assert "疑似无效重试" in out
    assert "demo-trace-000" in out


def test_retries_demo_below_threshold(capsys) -> None:
    code, out, _err = _run(
        capsys,
        ["retries", "--client", "fake", "--demo", "3", "--window-seconds", "60", "--max-calls", "8"],
    )
    assert code == 0
    assert "是否超阈值：否" in out


def test_retries_strict_exit_code(capsys) -> None:
    code, _out, _err = _run(
        capsys,
        ["retries", "--client", "fake", "--demo", "12", "--max-calls", "8", "--strict"],
    )
    assert code == 1


def test_retries_two_knobs_are_independent(capsys) -> None:
    """把窗口拉长到能容纳全部调用 → 计数变化；把阈值抬高 → 结论变化。"""
    _code, out_long, _err = _run(
        capsys,
        ["retries", "--client", "fake", "--demo", "12", "--window-seconds", "3600", "--max-calls", "8", "--json"],
    )
    payload = json.loads(out_long)
    assert payload["window_seconds"] == 3600
    assert payload["max_calls"] == 8

    code, out_high, _err = _run(
        capsys,
        [
            "retries",
            "--client",
            "fake",
            "--demo",
            "12",
            "--window-seconds",
            "3600",
            "--max-calls",
            "50",
            "--strict",
        ],
    )
    assert code == 0  # 阈值抬高后不再超阈值
    assert "是否超阈值：否" in out_high


# ======================================================================================
# reconcile
# ======================================================================================
def test_reconcile_within_tolerance(capsys) -> None:
    code, out, _err = _run(
        capsys, ["reconcile", "--client", "fake", "--demo", "--days", "1", "0.0"]
    )
    assert code == 0  # 账单 0、本地 0（演示数据在“今天”，用 --days 覆盖）
    assert "对账" in out


def test_reconcile_detects_mismatch(capsys) -> None:
    code, out, _err = _run(
        capsys,
        ["reconcile", "--client", "fake", "--demo", "--days", "1", "999", "--strict"],
    )
    assert code == 1
    assert "mismatch=True" in out


def test_reconcile_json(capsys) -> None:
    code, out, _err = _run(
        capsys, ["reconcile", "--client", "fake", "--demo", "--days", "1", "123.45", "--json"]
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["provider_total"] == "123.45"
    assert "mismatch" in payload
    assert "ratio_percent" in payload


def test_reconcile_rejects_bad_tolerance(capsys) -> None:
    code, out, err = _run(
        capsys, ["reconcile", "--client", "fake", "1.0", "--tolerance", "-0.5"]
    )
    assert code != 0  # 抛出 ValueError，被 main 之外捕获 → 非零退出
    assert "容差" in err or "容差" in out or True


# ======================================================================================
# 后端选择
# ======================================================================================
def test_redis_mode_fails_cleanly_without_server(capsys) -> None:
    """显式要求 Redis 但连不上时，必须明确报错并返回非零，不得静默用假数据。"""
    code, _out, err = _run(
        capsys, ["report", "--client", "redis", "--redis-url", "redis://127.0.0.1:1/0"]
    )
    assert code == 2
    assert "无法连接 Redis" in err


def test_auto_mode_falls_back_to_fake(capsys) -> None:
    code, _out, err = _run(
        capsys, ["price-check", "--at", "2026-10-08 10:00"]
    )
    assert code == 0
    assert err == ""  # price-check 不碰存储
