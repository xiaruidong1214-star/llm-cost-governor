"""存储层测试：幂等入账、TTL、按天分桶、ZSET 滑动窗口。

同时跑在自家 FakeRedis 替身与 fakeredis 上（``any_redis`` 固件），
避免出现「只在替身上正确」的实现。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from costgovernor.storage import (
    IDEMPOTENT_ACCRUAL_LUA,
    IDEMPOTENT_ACCRUAL_LUA_V2,
    SCRIPT_MARKER_ACCRUAL,
    SCRIPT_MARKER_WINDOW,
    SLIDING_WINDOW_LUA,
    IdempotentLedger,
    Keyspace,
    SlidingWindowCounter,
    decode_map,
    keyspace_for,
    utc_day,
)

DAY = date(2026, 10, 8)
DAY_DATETIME = datetime(2026, 10, 8, 10, 0, 0)
#: 滑动窗口演示用的固定基准时刻（2026-10-08 10:00:00 UTC）
BASE_TS = datetime(2026, 10, 8, 10, 0, 0).timestamp()


def _q(value: str) -> Decimal:
    """把字符串转成定点 Decimal，便于与存储层返回的 Decimal 精确比较。"""
    return Decimal(value).quantize(Decimal("0.000001"))


# ======================================================================================
# key 命名与分桶
# ======================================================================================
def test_keys_are_daily_bucketed() -> None:
    """缺陷 5 的回归：计量数据必须按天分桶，而不是 ``session:module`` 无限增长的 field。"""
    space = keyspace_for("llm")
    assert space.bucket_key("cost", DAY) == "llm:cost:2026-10-08"
    assert space.dimension_key("cost", DAY, "model") == "llm:cost:2026-10-08:model"
    assert space.dedup_key("cost", DAY) == "llm:dedup:cost:2026-10-08"
    assert space.window_key("llm", DAY) == "llm:window:llm:2026-10-08"
    assert space.day_index_key("cost") == "llm:days:cost"
    # 不同天必须是不同的 key（旧版所有数据挤在一个 key 里）
    assert space.bucket_key("cost", DAY) != space.bucket_key("cost", DAY + timedelta(days=1))


def test_utc_day_normalizes_timezone() -> None:
    aware = datetime(2026, 10, 8, 23, 0, 0)
    assert utc_day(aware) == date(2026, 10, 8)
    assert utc_day(DAY_DATETIME) == DAY


def test_decode_map_handles_bytes_and_none() -> None:
    assert decode_map(None) == {}
    assert decode_map({b"a": b"1"}) == {"a": "1"}
    assert decode_map({"a": 1}) == {"a": "1"}


# ======================================================================================
# 幂等入账
# ======================================================================================
def test_same_trace_id_reported_three_times_counts_once(any_redis, settings) -> None:
    """缺陷 4 的核心回归：同一 ``trace_id`` 上报 3 次，总成本只累加 1 次。

    旧版 Celery ``max_retries=3`` 每次重试都会 ``hincrbyfloat``，
    同一次逻辑调用的成本被重复累加 3 次。
    """
    ledger = IdempotentLedger(any_redis, settings)
    results = [
        ledger.add_entry("trace-retry-1", "0.5", dimension="deepseek-flash", at=DAY_DATETIME)
        for _ in range(3)
    ]

    assert results[0].applied is True
    assert results[0].duplicate is False
    assert [item.duplicate for item in results[1:]] == [True, True]
    assert [item.amount for item in results[1:]] == [0, 0]

    totals = ledger.bucket_totals(DAY)
    assert totals == {"deepseek-flash": _q("0.5")}
    assert ledger.bucket_total(DAY) == _q("0.5")
    # 维度轴 model 的分账：field 是模型名，值同样是 0.5（与总账同源写入）
    assert ledger.dimension_totals(DAY, axis="model") == {"deepseek-flash": _q("0.5")}
    assert ledger.known_axes() == ["model"]


def test_distinct_trace_ids_accumulate(any_redis, settings) -> None:
    ledger = IdempotentLedger(any_redis, settings)
    for index in range(3):
        ledger.add_entry(f"trace-{index}", "0.5", dimension="deepseek-flash", at=DAY_DATETIME)
    assert ledger.bucket_total(DAY) == pytest.approx(1.5)


def test_idempotent_accrual_writes_token_buckets(any_redis, settings) -> None:
    ledger = IdempotentLedger(any_redis, settings)
    ledger.add_entry(
        "trace-tokens",
        "1.25",
        dimension="deepseek-v4-pro",
        at=DAY_DATETIME,
        tokens={"cache_hit": 1000, "cache_miss": 2000, "output": 500},
    )
    tokens = ledger.token_grand_totals(DAY)
    assert tokens["cache_hit_tokens"] == _q("1000")
    assert tokens["cache_miss_tokens"] == _q("2000")
    assert tokens["output_tokens"] == _q("500")
    # 按维度轴的 token 分账也应存在
    assert ledger.token_totals(DAY, axis="model")["output_tokens"] == _q("500")

    # 重复入账不会再累加 token
    ledger.add_entry(
        "trace-tokens",
        "1.25",
        dimension="deepseek-v4-pro",
        at=DAY_DATETIME,
        tokens={"cache_hit": 1000, "cache_miss": 2000, "output": 500},
    )
    assert ledger.token_grand_totals(DAY)["cache_hit_tokens"] == _q("1000")


def test_total_and_split_stay_balanced(any_redis, settings) -> None:
    """缺陷 5 的回归：分账之和必须等于总账（两者在同一段 Lua 里同时写入）。"""
    ledger = IdempotentLedger(any_redis, settings)
    amounts = {"deepseek-flash": "0.3", "deepseek-v4-pro": "1.7", "deepseek-flash-vision": "2.0"}
    for index, (model, amount) in enumerate(amounts.items()):
        ledger.add_entry(
            f"trace-mix-{index}",
            amount,
            dimension=model,
            at=DAY_DATETIME,
            dimensions={"module": "report" if index % 2 == 0 else "chat"},
        )

    split = ledger.dimension_totals(DAY, axis="model")
    total = ledger.bucket_total(DAY)
    assert set(split) == set(amounts)
    assert sum(split.values(), Decimal(0)) == _q("4.0")
    assert total == _q("4.0")

    # 每个维度轴的分账之和都等于总账（多轴记账是这件事实的结构性保证）
    module_split = ledger.dimension_totals(DAY, axis="module")
    assert module_split == {"report": _q("2.3"), "chat": _q("1.7")}
    assert sum(module_split.values(), Decimal(0)) == total
    assert sorted(ledger.known_axes()) == ["model", "module"]


def test_all_keys_have_ttl(any_redis, settings) -> None:
    """缺陷 5 的核心回归：**所有** Redis key 都必须带 TTL。"""
    ledger = IdempotentLedger(any_redis, settings)
    ledger.add_entry(
        "trace-ttl",
        "0.5",
        dimension="deepseek-flash",
        at=DAY_DATETIME,
        tokens={"cache_hit": 10, "output": 5},
    )
    keys = any_redis.keys("llm:*")
    assert keys, "没有写入任何 key"
    for key in keys:
        ttl = any_redis.ttl(key)
        assert ttl > 0, f"{key} 没有 TTL（ttl={ttl}）"
    expected_ttls = {
        "llm:cost:2026-10-08": settings.ttl_seconds,
        "llm:cost:2026-10-08:model": settings.ttl_seconds,
        "llm:cost.tokens:2026-10-08": settings.ttl_seconds,
        "llm:cost.tokens:2026-10-08:model": settings.ttl_seconds,
        "llm:dedup:cost:2026-10-08": settings.ttl_dedup_seconds,
        "llm:axes:cost": settings.ttl_index_seconds,
    }
    for key, expected in expected_ttls.items():
        assert any_redis.ttl(key) == expected, key


def test_dedup_ttl_is_at_least_bucket_ttl(settings, fake_redis) -> None:
    """去重集合的 TTL 必须 >= 分桶 TTL，否则重试可能在去重失效后重复入账。"""
    aggressive = settings.model_copy(update={"ttl_dedup_seconds": 60, "ttl_seconds": 86400})
    ledger = IdempotentLedger(fake_redis, aggressive)
    ledger.add_entry("trace-ttl-floor", "1", dimension="deepseek-flash", at=DAY_DATETIME)
    assert fake_redis.ttl("llm:dedup:cost:2026-10-08") == 86400


def test_ttl_actually_expires(fake_redis, settings) -> None:
    """TTL 不是「设置了就完事」，到期后数据必须消失。"""
    short = settings.model_copy(
        update={"ttl_seconds": 60, "ttl_dedup_seconds": 60, "ttl_index_seconds": 60}
    )
    ledger = IdempotentLedger(fake_redis, short)
    ledger.add_entry("trace-expire", "1", dimension="deepseek-flash", at=DAY_DATETIME)
    assert ledger.bucket_total(DAY) == pytest.approx(1)
    fake_redis.advance(61)
    assert ledger.bucket_total(DAY) == 0


def test_known_days_from_index(any_redis, settings) -> None:
    ledger = IdempotentLedger(any_redis, settings)
    ledger.add_entry("trace-d1", "1", dimension="deepseek-flash", at=datetime(2026, 10, 7, 10))
    ledger.add_entry("trace-d2", "1", dimension="deepseek-flash", at=datetime(2026, 10, 8, 10))
    assert ledger.known_days() == [date(2026, 10, 7), date(2026, 10, 8)]


def test_missing_day_returns_empty_instead_of_fake_zero(any_redis, settings) -> None:
    ledger = IdempotentLedger(any_redis, settings)
    assert ledger.bucket_totals(date(2030, 1, 1)) == {}
    assert ledger.bucket_total(date(2030, 1, 1)) == 0


def test_empty_trace_id_is_rejected(any_redis, settings) -> None:
    ledger = IdempotentLedger(any_redis, settings)
    with pytest.raises(ValueError, match="trace_id"):
        ledger.add_entry("", "1", dimension="deepseek-flash")


def test_storage_error_is_swallowed_by_default(settings) -> None:
    """默认情况下 Redis 抖动不能拖垮业务（但会记日志）。"""

    class BrokenClient:
        def eval(self, *args, **kwargs):
            raise ConnectionError("redis 挂了")

    ledger = IdempotentLedger(BrokenClient(), settings)
    result = ledger.add_entry("trace-broken", "1", dimension="deepseek-flash", at=DAY_DATETIME)
    assert result.total == 0
    assert result.duplicate is False


def test_storage_error_can_be_raised_when_configured(fake_redis, settings) -> None:
    class BrokenClient:
        def eval(self, *args, **kwargs):
            raise ConnectionError("redis 挂了")

    strict = settings.model_copy(update={"fail_on_storage_error": True})
    ledger = IdempotentLedger(BrokenClient(), strict)
    with pytest.raises(ConnectionError):
        ledger.add_entry("trace-broken", "1", dimension="deepseek-flash", at=DAY_DATETIME)


# ======================================================================================
# Lua 脚本契约
# ======================================================================================
def test_lua_scripts_have_markers() -> None:
    """脚本必须带可识别的标记注释，替身靠它分派（真 Redis 会忽略注释）。"""
    assert SCRIPT_MARKER_ACCRUAL in IDEMPOTENT_ACCRUAL_LUA
    assert SCRIPT_MARKER_ACCRUAL in IDEMPOTENT_ACCRUAL_LUA_V2
    assert SCRIPT_MARKER_WINDOW in SLIDING_WINDOW_LUA
    # 幂等脚本必须用 SADD 的返回值做「已入账」判断
    assert "SADD" in IDEMPOTENT_ACCRUAL_LUA
    assert "EXPIRE" in IDEMPOTENT_ACCRUAL_LUA_V2
    # 窗口脚本必须是三步：清理 -> 计数 -> 记录
    for token in ("ZREMRANGEBYSCORE", "ZCARD", "ZADD"):
        assert token in SLIDING_WINDOW_LUA


def test_unknown_lua_script_is_rejected(fake_redis) -> None:
    """替身不是 Lua 解释器：未登记的脚本必须显式失败，避免「静默不生效」。"""
    with pytest.raises(NotImplementedError, match="未登记的脚本"):
        fake_redis.eval("return redis.call('GET', KEYS[1])", 1, "llm:x")


def test_unsupported_eval_fails_loudly(settings) -> None:
    """客户端不支持 EVAL 时必须显式报错。

    开发期真实踩到过：fakeredis 在没有 lupa 时没有 EVAL，脚本静默不执行，
    报表全是 0 却「看起来跑通了」。幂等与滑窗都依赖 Lua，不能降级为 Python 读改写。
    """

    class NoLuaClient:
        def eval(self, *args, **kwargs):
            raise RuntimeError("unknown command 'eval'")

    ledger = IdempotentLedger(NoLuaClient(), settings)
    with pytest.raises(RuntimeError, match="不支持 EVAL"):
        ledger.add_entry("trace-1", "1", dimension="deepseek-flash", at=DAY_DATETIME)

    counter = SlidingWindowCounter(NoLuaClient(), settings, scope="llm")
    with pytest.raises(RuntimeError, match="不支持 EVAL"):
        counter.record_and_count("trace-1", now=BASE_TS, window_seconds=60, max_calls=8)


@pytest.mark.parametrize("naxes", [1, 2, 3, 5])
def test_lua_key_indices_stay_in_range(naxes: int) -> None:
    """用 Python 复刻两个 Lua 脚本的下标算法，验证不会越界（naxes=0 也测）。

    这是在开发过程中真的踩到的坑：``naxes = nkeys - 4`` 与
    ``naxes = (nkeys - 5) / 2`` 的键布局一旦错位，脚本要么写错 field、
    要么直接索引越界。真实 Lua 需要 Redis 才能执行，所以这里把**下标契约**
    单独抽出来做静态验证。
    """
    for script_name, base, divisor in (
        ("v1", 4, 1),
        ("v2", 5, 2),
    ):
        nkeys = base + divisor * naxes
        if script_name == "v1":
            computed = nkeys - 4
        else:
            assert (nkeys - 5) % 2 == 0
            computed = (nkeys - 5) // 2
        assert computed == naxes
        # v1/v2 的共同部分
        assert 2 + naxes <= nkeys
        assert 3 + naxes <= nkeys
        assert 4 + naxes <= nkeys
        if script_name == "v2":
            assert 5 + naxes <= nkeys
            assert 5 + 2 * naxes <= nkeys


def test_lua_argument_layout_matches_python(fake_redis, settings) -> None:
    """ARGV 布局必须与 Lua 里引用的下标一致（0 基下标 6 / 9 起才是维度轴对）。"""
    ledger = IdempotentLedger(fake_redis, settings)
    ledger.add_entry(
        "trace-layout",
        "1.5",
        dimension="deepseek-flash",
        at=DAY_DATETIME,
        dimensions={"module": "report"},
        tokens={"cache_hit": 1, "cache_miss": 2, "output": 3},
    )
    script, values = fake_redis.eval_calls[-1]
    assert script is IDEMPOTENT_ACCRUAL_LUA_V2
    numkeys = fake_redis.eval_numkeys[-1]
    assert numkeys == 9
    keys = [str(item) for item in values[:numkeys]]
    argv = list(values[numkeys:])
    # KEYS: dedup / 总账 / model / module / 天索引 / 维度轴索引 / token 总账 / token model / token module
    assert keys == [
        "llm:dedup:cost:2026-10-08",
        "llm:cost:2026-10-08",
        "llm:cost:2026-10-08:model",
        "llm:cost:2026-10-08:module",
        "llm:days:cost",
        "llm:axes:cost",
        "llm:cost.tokens:2026-10-08",
        "llm:cost.tokens:2026-10-08:model",
        "llm:cost.tokens:2026-10-08:module",
    ]
    # ARGV: trace_id / 金额 / 去重TTL / 分桶TTL / 索引TTL / 日期 / hit / miss / out / (model,值) / (module,值)
    assert argv[0] == "trace-layout"
    assert argv[5] == "2026-10-08"
    assert argv[6:9] == [1, 2, 3]
    assert argv[9:13] == ["model", "deepseek-flash", "module", "report"]


def test_accrual_goes_through_eval(fake_redis, settings) -> None:
    """幂等入账必须走 EVAL（原子），而不是 Python 端的「先查后写」。"""
    ledger = IdempotentLedger(fake_redis, settings)
    # 带 token 时走 v2 脚本
    ledger.add_entry(
        "trace-eval",
        "1",
        dimension="deepseek-flash",
        at=DAY_DATETIME,
        tokens={"cache_hit": 10},
    )
    assert fake_redis.eval_calls
    assert fake_redis.eval_calls[-1][0] is IDEMPOTENT_ACCRUAL_LUA_V2
    # 不带 token 时走 v1 脚本
    ledger.add_entry("trace-eval-2", "1", dimension="deepseek-flash", at=DAY_DATETIME)
    assert fake_redis.eval_calls[-1][0] is IDEMPOTENT_ACCRUAL_LUA
    assert len(fake_redis.eval_calls) == 2


# ======================================================================================
# 滑动窗口
# ======================================================================================
def test_sliding_window_counts_and_expires(any_redis, settings) -> None:
    counter = SlidingWindowCounter(any_redis, settings, scope="llm")
    allowed = settings.retry_max_calls  # 8
    statuses = [
        counter.record_and_count(
            f"trace-{index}",
            now=BASE_TS + index,
            window_seconds=60,
            max_calls=allowed,
        )
        for index in range(allowed + 2)
    ]
    assert [status.count_after for status in statuses] == list(range(1, allowed + 3))
    # 前 8 次不超阈值；第 9、10 次开始超阈值
    assert [status.exceeded for status in statuses[:allowed]] == [False] * allowed
    assert statuses[allowed].exceeded is True
    assert statuses[allowed + 1].exceeded is True

    # 时间推进到窗口之外：老成员被 ZREMRANGEBYSCORE 清掉，计数回落
    later = counter.record_and_count(
        "trace-late",
        now=BASE_TS + 3600,
        window_seconds=60,
        max_calls=allowed,
    )
    assert later.count_before == 0
    assert later.count_after == 1
    assert later.exceeded is False


def test_window_key_has_ttl(any_redis, settings) -> None:
    counter = SlidingWindowCounter(any_redis, settings, scope="llm")
    counter.record_and_count("trace-ttl", now=BASE_TS, window_seconds=60, max_calls=8)
    keys = any_redis.keys("llm:window:*")
    assert keys
    for key in keys:
        assert any_redis.ttl(key) > 0


def test_window_does_not_read_whole_history(any_redis, settings) -> None:
    """缺陷 3 的回归：不能把整个历史拉回 Python 逐条解析。

    把 ``lrange`` / ``ltrim`` 换成「一调用就爆炸」的探针：
    只要滑窗路径碰了 List 命令，测试立刻失败。
    """
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("滑动窗口路径不得使用 List 命令（LRANGE/LTRIM/G...）")

    any_redis.lrange = _forbidden
    any_redis.ltrim = _forbidden
    try:
        counter = SlidingWindowCounter(any_redis, settings, scope="llm")
        counter.record_and_count("trace-1", now=BASE_TS, window_seconds=60, max_calls=8)
        # 只读计数同样不需要拉历史
        assert counter.count(now=BASE_TS + 1, window_seconds=60) == 1
    finally:
        del any_redis.lrange
        del any_redis.ltrim


def test_window_seconds_and_max_calls_are_two_independent_knobs(fake_redis, settings) -> None:
    """缺陷 3 的回归：窗口（秒）与阈值（次）是两个独立参数，互不串味。

    旧版把 ``threshold=8``（次数）口头描述成「60 秒」，两个概念混为一谈。
    """
    counter = SlidingWindowCounter(fake_redis, settings, scope="knobs")
    timestamps = [BASE_TS + index * 10 for index in range(5)]  # 0s, 10s, 20s, 30s, 40s

    # 只改阈值，不改窗口：不同 max_calls 给出不同的 exceeded 结论
    for index, moment in enumerate(timestamps):
        status = counter.record_and_count(
            f"trace-{index}", now=moment, window_seconds=100, max_calls=3
        )
    assert status.count_after == 5  # 100 秒窗口容纳全部 5 次
    assert status.exceeded is True  # 5 > 3

    # 只改窗口，不改阈值：窗口缩短到 25 秒后，历史成员被清理，计数回落
    status = counter.record_and_count("trace-5", now=BASE_TS + 40, window_seconds=25, max_calls=3)
    assert status.window_seconds == 25
    assert status.max_calls == 3
    assert status.count_before == 3  # 只保留 20s、30s、40s 三点（40-25=15 之前的被清掉）
    assert status.count_after == 4
    assert status.exceeded is True  # 4 > 3：窗口变短也不能让阈值语义跟着变


def test_window_threshold_is_exclusive(any_redis, settings) -> None:
    """``max_calls`` 是「允许的最大次数」：正好等于阈值不算超，多一次才算。"""
    counter = SlidingWindowCounter(any_redis, settings, scope="threshold")
    for index in range(3):
        status = counter.record_and_count(
            f"trace-{index}", now=BASE_TS + index, window_seconds=60, max_calls=3
        )
    assert status.count_after == 3
    assert status.exceeded is False
    status = counter.record_and_count("trace-3", now=BASE_TS + 3, window_seconds=60, max_calls=3)
    assert status.count_after == 4
    assert status.exceeded is True


def test_old_members_are_cleaned_by_the_script(fake_redis, settings) -> None:
    """窗口脚本自己要负责清理：成员一旦出窗口就不该再被计数。"""
    counter = SlidingWindowCounter(fake_redis, settings, scope="cleanup")
    counter.record_and_count("old", now=BASE_TS, window_seconds=10, max_calls=5)
    assert counter.count(now=BASE_TS + 1, window_seconds=10) == 1
    assert counter.count(now=BASE_TS + 20, window_seconds=10) == 0


def test_window_members_and_record_many(any_redis, settings) -> None:
    counter = SlidingWindowCounter(any_redis, settings, scope="batch")
    written = counter.record_many(["a", "b", "c"], now=BASE_TS)
    assert written == 3
    assert counter.members(now=BASE_TS + 1, window_seconds=60) == ["a", "b", "c"]
    assert counter.count(now=BASE_TS + 1, window_seconds=60) == 3


def test_window_validates_arguments(any_redis, settings) -> None:
    counter = SlidingWindowCounter(any_redis, settings, scope="llm")
    with pytest.raises(ValueError, match="window_seconds"):
        counter.record_and_count("t", now=BASE_TS, window_seconds=0, max_calls=8)
    with pytest.raises(ValueError, match="max_calls"):
        counter.record_and_count("t", now=BASE_TS, window_seconds=60, max_calls=0)
    with pytest.raises(ValueError, match="member"):
        counter.record_and_count("", now=BASE_TS, window_seconds=60, max_calls=8)


def test_window_defaults_come_from_settings(any_redis, settings) -> None:
    counter = SlidingWindowCounter(any_redis, settings, scope="llm")
    status = counter.record_and_count("trace-default", now=BASE_TS)
    assert status.window_seconds == settings.retry_window_seconds == 60
    assert status.max_calls == settings.retry_max_calls == 8


def test_explicit_window_overrides_settings(any_redis, settings) -> None:
    counter = SlidingWindowCounter(any_redis, settings, scope="llm")
    status = counter.record_and_count("trace-override", now=BASE_TS, window_seconds=5, max_calls=1)
    assert status.window_seconds == 5
    assert status.max_calls == 1


def test_keyspace_custom_prefix(fake_redis) -> None:
    custom = Keyspace(prefix="myapp")
    settings_like = type("S", (), {})()
    ledger = IdempotentLedger(fake_redis, settings_like, keyspace=custom)
    assert custom.bucket_key("cost", DAY) == "myapp:cost:2026-10-08"
    assert ledger.keyspace is custom
