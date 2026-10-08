"""Redis 存储层：按天分桶 + TTL + Lua 幂等入账 + ZSET 滑动窗口。

三个核心设计（见 README「设计决策」）：

1. **命令表面很小**：本模块只使用
   ``get / set / setex / hget / hgetall / hsetnx / hincrbyfloat / expire / sadd /
   smembers / zadd / zremrangebyscore / zcard / zrange / keys / pipeline / eval``。
   刻意限制是为了能用极小的测试替身（``tests/conftest.py`` 的 ``FakeRedis``）精确模拟，
   在没有真实 Redis 的环境里也能验证「幂等」「TTL」「原子性」这些关键性质。
   ``eval`` 是唯一例外：替身里只实现本项目这两个脚本的语义。

2. **所有 key 都带 TTL**，且计量数据**按天分桶**（``llm:cost:2026-10-07``），
   避免旧版 ``session:module`` 这种永久 field 把单个 Hash 撑到无限大。

3. **凡是「读-改-写」都放进 Lua 原子执行**：
   * :data:`IDEMPOTENT_ACCRUAL_LUA` —— 「同一 trace_id 只入账一次」；
   * :data:`SLIDING_WINDOW_LUA` —— 「清理过期 + 计数 + 记录」三步原子完成。
   旧版用 ``LRANGE 0 -1`` 把整个 List 拉回 Python 再逐条 ``json.loads``，
   复杂度是 O(全部历史)，而且窗口实际由 ``LTRIM 10000`` 决定、不由时间决定。

维度模型（这是旧版最混乱的地方，这里显式定死）：

* **维度轴（axis）**：``model`` 或 ``module`` —— 报表按哪个角度分组；
* **维度值（value）**：``deepseek-flash``、``report`` 这类具体取值。

因此 ``llm:cost:2026-10-07:model`` 是「按模型分组的分账 Hash」，
它的 field 才是模型名；``llm:cost:2026-10-07:module`` 同理。
每次入账会把同一笔金额同时写进总账与**每个**维度轴的分账，
所以「分账之和 == 总账」是结构上成立的，而不是靠事后对账去凑。
"""

from __future__ import annotations

import time as time_module
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

__all__ = [
    "SCRIPT_MARKER_ACCRUAL",
    "SCRIPT_MARKER_WINDOW",
    "PROBE_KEY_SUFFIX",
    "DEFAULT_DIMENSION_AXIS",
    "IDEMPOTENT_ACCRUAL_LUA",
    "IDEMPOTENT_ACCRUAL_LUA_V2",
    "SLIDING_WINDOW_LUA",
    "IDEMPOTENT_RESULT_KEYS",
    "Keyspace",
    "keyspace_for",
    "decode",
    "decode_map",
    "RedisLike",
    "EntryResult",
    "IdempotentLedger",
    "WindowStatus",
    "SlidingWindowCounter",
    "LedgerSnapshot",
    "snapshot",
    "utc_day",
]

#: Lua 脚本里的标记注释。测试替身靠它识别「这是本项目的哪个脚本」，
#: 真正的 Redis 会把注释当普通 Lua 注释忽略。
SCRIPT_MARKER_ACCRUAL = "-- costgovernor:accrue:v1"
SCRIPT_MARKER_WINDOW = "-- costgovernor:window:v1"

#: ``EVAL`` 返回数组的下标契约（Lua 里返回的是数组，没有 dict）。
IDEMPOTENT_RESULT_KEYS = ("duplicate", "applied_to", "amount", "total")

#: 探针 key 后缀：用于建连时确认服务端到底是不是真的 Redis。
PROBE_KEY_SUFFIX = "probe"

DEFAULT_DIMENSION_AXIS = "model"

# --------------------------------------------------------------------------------------
# Lua 脚本
# --------------------------------------------------------------------------------------
# 幂等入账 v1：只有金额，没有 token 明细。
#
# KEYS 布局（naxes 个维度轴，naxes = #KEYS - 4）：
#   1 dedup / 2 总账 / [3 .. 2+naxes] 各维度轴分账 / [3+naxes] 天索引 / [4+naxes] 维度轴索引
# ARGV 布局：
#   1 trace_id / 2 金额 / 3 去重 TTL / 4 分桶 TTL / 5 索引 TTL / 6 当天日期
#   7+ 每个维度轴一对：(轴名, 该轴取值)
#
# 返回 {duplicate, applied_to, amount, total}
IDEMPOTENT_ACCRUAL_LUA = """
-- costgovernor:accrue:v1
-- 幂等入账：同一 trace_id 只累加一次
local nkeys = #KEYS
local naxes = nkeys - 4
if redis.call('SADD', KEYS[1], ARGV[1]) == 0 then
  return {1, ARGV[7], '0', redis.call('HGET', KEYS[2], ARGV[7]) or '0'}
end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
-- 总账：所有维度都累加到同一个 field，保证「分账之和 == 总账」
local total = redis.call('HINCRBYFLOAT', KEYS[2], ARGV[7], ARGV[2])
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[4]))
-- 分账：每个维度轴各写一份（field 用该轴自己的取值），金额完全相同
for i = 1, naxes do
  redis.call('HINCRBYFLOAT', KEYS[2 + i], ARGV[6 + 2 * i], ARGV[2])
  redis.call('EXPIRE', KEYS[2 + i], tonumber(ARGV[4]))
end
-- 登记日期与维度轴名，便于报表枚举（旧版靠人工记忆有哪些 field，容易漏）
redis.call('SADD', KEYS[3 + naxes], ARGV[6])
redis.call('EXPIRE', KEYS[3 + naxes], tonumber(ARGV[5]))
for i = 1, naxes do
  redis.call('SADD', KEYS[4 + naxes], ARGV[5 + 2 * i])
end
redis.call('EXPIRE', KEYS[4 + naxes], tonumber(ARGV[5]))
return {0, ARGV[7], ARGV[2], total}
"""

# 幂等入账 v2：除金额外再记账 token 三分类（cache_hit / cache_miss / output）。
#
# KEYS 布局（naxes 个维度轴，naxes = (#KEYS - 5) / 2）：
#   1 dedup / 2 总账
#   [3 .. 2+naxes]             各维度轴金额分账
#   [3+naxes]                  天索引
#   [4+naxes]                  维度轴名索引
#   [5+naxes]                  token 总账
#   [6+naxes .. 5+2*naxes]     各维度轴 token 分账
# ARGV 布局：
#   1 trace_id / 2 金额 / 3 去重 TTL / 4 分桶 TTL / 5 索引 TTL / 6 当天日期
#   7 cache_hit tokens / 8 cache_miss tokens / 9 output tokens
#   10+ 每个维度轴一对：(轴名, 该轴取值)
IDEMPOTENT_ACCRUAL_LUA_V2 = """
-- costgovernor:accrue:v1
-- 幂等入账 v2：同一 trace_id 只累加一次，同时记账 token 三分类
local nkeys = #KEYS
local naxes = (nkeys - 5) / 2
if redis.call('SADD', KEYS[1], ARGV[1]) == 0 then
  return {1, ARGV[10], '0', redis.call('HGET', KEYS[2], ARGV[10]) or '0'}
end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
local total = redis.call('HINCRBYFLOAT', KEYS[2], ARGV[10], ARGV[2])
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[4]))
for i = 1, naxes do
  redis.call('HINCRBYFLOAT', KEYS[2 + i], ARGV[9 + 2 * i], ARGV[2])
  redis.call('EXPIRE', KEYS[2 + i], tonumber(ARGV[4]))
end
redis.call('SADD', KEYS[3 + naxes], ARGV[6])
redis.call('EXPIRE', KEYS[3 + naxes], tonumber(ARGV[5]))
for i = 1, naxes do
  redis.call('SADD', KEYS[4 + naxes], ARGV[8 + 2 * i])
end
redis.call('EXPIRE', KEYS[4 + naxes], tonumber(ARGV[5]))
local token_total = KEYS[5 + naxes]
local cache_hit = tonumber(ARGV[7])
local cache_miss = tonumber(ARGV[8])
local output = tonumber(ARGV[9])
if cache_hit > 0 then
  redis.call('HINCRBYFLOAT', token_total, 'cache_hit_tokens', ARGV[7])
end
if cache_miss > 0 then
  redis.call('HINCRBYFLOAT', token_total, 'cache_miss_tokens', ARGV[8])
end
if output > 0 then
  redis.call('HINCRBYFLOAT', token_total, 'output_tokens', ARGV[9])
end
redis.call('EXPIRE', token_total, tonumber(ARGV[4]))
for i = 1, naxes do
  local token_axis_key = KEYS[5 + naxes + i]
  if cache_hit > 0 then
    redis.call('HINCRBYFLOAT', token_axis_key, 'cache_hit_tokens', ARGV[7])
  end
  if cache_miss > 0 then
    redis.call('HINCRBYFLOAT', token_axis_key, 'cache_miss_tokens', ARGV[8])
  end
  if output > 0 then
    redis.call('HINCRBYFLOAT', token_axis_key, 'output_tokens', ARGV[9])
  end
  redis.call('EXPIRE', token_axis_key, tonumber(ARGV[4]))
end
return {0, ARGV[10], ARGV[2], total}
"""

# 滑动窗口：ZREMRANGEBYSCORE 清理 → ZCARD 计数 → ZADD 记录 → EXPIRE 兜底，原子完成。
# KEYS[1] 窗口 ZSET
# ARGV[1] now(秒，允许小数)  ARGV[2] window_seconds  ARGV[3] max_calls  ARGV[4] member
#        ARGV[5] ttl_seconds
# 返回 {count_before, count_after, exceeded, oldest_score}
SLIDING_WINDOW_LUA = """
-- costgovernor:window:v1
-- 滑动窗口：ZREMRANGEBYSCORE 清理 + ZCARD 计数 + ZADD 记录
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_calls = tonumber(ARGV[3])
local cutoff = now - window
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', cutoff)
local before = redis.call('ZCARD', KEYS[1])
local exceeded = 0
if before >= max_calls then
  exceeded = 1
end
redis.call('ZADD', KEYS[1], ARGV[1], ARGV[4])
local after = redis.call('ZCARD', KEYS[1])
local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[5]))
local oldest_score = ''
if #oldest > 0 then
  oldest_score = oldest[2]
end
return {before, after, exceeded, oldest_score}
"""


# --------------------------------------------------------------------------------------
# key 命名
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Keyspace:
    """key 命名规则。

    统一前缀 + **按天分桶** + 明确的业务语义 + **维度轴**，
    既能用 ``KEYS llm:cost:2026-*`` 找历史分桶，又不会让单个 key 无限膨胀。
    """

    prefix: str = "llm"

    def bucket_key(self, kind: str, day: date) -> str:
        """总账分桶 key，例如 ``llm:cost:2026-10-07``。"""
        return f"{self.prefix}:{kind}:{day.isoformat()}"

    def dimension_key(self, kind: str, day: date, axis: str) -> str:
        """按**维度轴**分组的分账 key，例如 ``llm:cost:2026-10-07:model``。

        注意：``axis`` 是 ``model`` / ``module`` 这样的**分组角度**，
        具体取值（``deepseek-flash``）是这个 Hash 的 field。
        """
        return f"{self.bucket_key(kind, day)}:{axis}"

    def tokens_bucket_key(self, kind: str, day: date) -> str:
        """token 总账，例如 ``llm:cost.tokens:2026-10-07``。"""
        return self.bucket_key(f"{kind}.tokens", day)

    def tokens_dimension_key(self, kind: str, day: date, axis: str) -> str:
        return self.dimension_key(f"{kind}.tokens", day, axis)

    def axis_index_key(self, kind: str) -> str:
        """维度轴索引，例如 ``llm:axes:cost``（记录出现过哪些维度轴）。"""
        return f"{self.prefix}:axes:{kind}"

    def day_index_key(self, kind: str) -> str:
        """枚举历史分桶用的索引集合，例如 ``llm:days:cost``。"""
        return f"{self.prefix}:days:{kind}"

    def dedup_key(self, kind: str, day: date) -> str:
        """trace_id 幂等去重集合，例如 ``llm:dedup:cost:2026-10-07``。"""
        return f"{self.prefix}:dedup:{kind}:{day.isoformat()}"

    def window_key(self, scope: str, day: date) -> str:
        """滑动窗口 ZSET，例如 ``llm:window:llm:2026-10-07``。"""
        return f"{self.prefix}:window:{scope}:{day.isoformat()}"

    def probe_key(self) -> str:
        return f"{self.prefix}:{PROBE_KEY_SUFFIX}"


def keyspace_for(prefix: str = "llm") -> Keyspace:
    """构造 key 命名器。"""
    return Keyspace(prefix=prefix)


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
#: 本模块实际用到的 Redis 命令表面。类型标注用 Any 是刻意的：
#: 真实 ``redis.Redis``、``fakeredis.FakeRedis`` 和测试替身都应该能用。
RedisLike = Any


def decode(value: Any) -> Any:
    """把 ``bytes`` 解码成 ``str``；其他类型原样返回。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def decode_map(raw: Mapping[Any, Any] | None) -> dict[str, str]:
    """把 ``HGETALL`` / ``HGET`` 的返回值统一成 ``dict[str, str]``。"""
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(decode(key)): str(decode(value)) for key, value in raw.items()}


def utc_day(moment: datetime | None = None) -> date:
    """取 UTC 日期，作为分桶键。"""
    moment = moment or datetime.now()
    if moment.tzinfo is not None:

        moment = moment.astimezone(UTC).replace(tzinfo=None)
    return moment.date()


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(decode(value))


def _to_int(value: Any) -> int:
    return int(_to_float(value))


def _quantize(amount: Decimal) -> str:
    """金额转字符串时做微小的四舍五入，避免浮点尾巴在报表里乱飞。"""
    return str(amount.quantize(Decimal("0.0000000001")))


class _RedisCommandMixin:
    """统一的命令调用与错误处理。

    埋点属于旁路逻辑：默认不允许因为 Redis 抖动而拖垮业务请求
    （``settings.fail_on_storage_error`` 控制是否向上抛）。
    """

    settings: Any  # 由子类提供（Settings 实例）

    def _log(self, event: str, **fields: Any) -> None:
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.warning(event, **fields)

    def _maybe_raise(self, exc: Exception, *, op: str) -> None:
        if getattr(self.settings, "fail_on_storage_error", False):
            raise exc
        self._log("storage_error", op=op, error=repr(exc))

    def _run(self, op: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 统一走「记日志或抛出」策略
            self._maybe_raise(exc, op=op)
            return None

    def _eval(self, op: str, script: str, keys: list[str], argv: list[Any]) -> Any:
        """执行 Lua 脚本。

        与 :meth:`_run` 的区别：``EVAL`` 不受支持属于**配置/环境错误**
        （例如用了一个没有 Lua 能力的 fakeredis 替身），必须**始终**显式失败。
        否则脚本不生效、报表全是 0，看起来「跑通了」其实什么都没记 —— 这比报错危险得多。
        """
        try:
            return self.client.eval(script, len(keys), *keys, *argv)
        except Exception as exc:  # noqa: BLE001 - 需要区分「不支持 EVAL」与其他错误
            if isinstance(exc, NotImplementedError) or "eval" in str(exc).lower():
                raise RuntimeError(
                    f"{op}: 当前存储客户端不支持 EVAL（Lua 脚本）。"
                    "幂等入账与滑动窗口都依赖 Lua，无法降级为 Python 端读改写。"
                    "请连接真实 Redis，或使用 costgovernor.testing.InMemoryRedis 替身。"
                    f"原始错误：{exc!r}"
                ) from exc
            self._maybe_raise(exc, op=op)
            return None


# --------------------------------------------------------------------------------------
# 幂等账本
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EntryResult:
    """一次入账尝试的结果。"""

    duplicate: bool
    applied_to: str
    amount: Decimal
    total: Decimal
    day: date
    kind: str

    @property
    def applied(self) -> bool:
        return not self.duplicate

    def to_dict(self) -> dict[str, Any]:
        return {
            "duplicate": self.duplicate,
            "applied": self.applied,
            "applied_to": self.applied_to,
            "amount": str(self.amount),
            "total": str(self.total),
            "day": self.day.isoformat(),
            "kind": self.kind,
        }


class IdempotentLedger(_RedisCommandMixin):
    """幂等账本：``trace_id`` 唯一，金额只累加一次。

    旧版的问题：Celery ``max_retries=3`` 时每次重试都会再执行一次
    ``hincrbyfloat(COST_KEY, field, cost)``，同一次逻辑调用的成本被重复累加，
    而且没有任何 ``trace_id``，无法把重试归并回同一个逻辑调用。

    这里把「去重 + 累加总账 + 累加各维度轴分账 + 登记索引 + 续 TTL」
    全部塞进一个 Lua 脚本，任何一步都不会被其他写者穿插；
    总账与分账在同一段脚本里写入相同金额，因此**结构上就保证了对得平**。
    """

    kind = "cost"

    def __init__(self, client: RedisLike, settings: Any, *, keyspace: Keyspace | None = None) -> None:
        self.client = client
        self.settings = settings
        self.keyspace = keyspace or keyspace_for(getattr(settings, "key_prefix", "llm"))

    # -------- 写 --------
    def add_entry(
        self,
        trace_id: str,
        amount: Decimal | float | str,
        *,
        dimension: str,
        dimensions: Mapping[str, str] | None = None,
        at: datetime | None = None,
        kind: str | None = None,
        tokens: Mapping[str, int] | None = None,
    ) -> EntryResult:
        """入账一条成本。

        :param trace_id: 逻辑调用 ID；重试必须复用同一个 ID。
        :param amount: 金额（元）。
        :param dimension: 主维度值（默认写入 ``model`` 轴），例如 ``deepseek-flash``。
        :param dimensions: 额外的维度轴，例如 ``{"module": "report", "tenant": "acme"}``；
            会自动与 ``{"model": dimension}`` 合并。同一笔金额会按每个轴各写一份分账。
        :param tokens: 可选，``{"cache_hit": .., "cache_miss": .., "output": ..}``，
            给了就顺带写 token 三分类（走 v2 脚本）。
        :return: :class:`EntryResult`；``duplicate=True`` 表示本次没有累加任何金额。
        """
        if not trace_id:
            raise ValueError("trace_id 不能为空：幂等入账必须有一个稳定的逻辑调用 ID")

        bucket_day = utc_day(at)
        active_kind = kind or self.kind
        ttl = int(self.settings.ttl_seconds)
        index_ttl = int(self.settings.ttl_index_seconds)
        dedup_ttl = max(int(self.settings.ttl_dedup_seconds), ttl)

        axes: dict[str, str] = {DEFAULT_DIMENSION_AXIS: dimension}
        for axis, value in (dimensions or {}).items():
            if value:
                axes[str(axis)] = str(value)

        token_counts = {
            "cache_hit": int((tokens or {}).get("cache_hit", 0)),
            "cache_miss": int((tokens or {}).get("cache_miss", 0)),
            "output": int((tokens or {}).get("output", 0)),
        }
        write_tokens = any(value > 0 for value in token_counts.values())

        keys: list[str] = [
            self.keyspace.dedup_key(active_kind, bucket_day),
            self.keyspace.bucket_key(active_kind, bucket_day),
        ]
        # 所有维度轴的金额分账必须连续排列：Lua 靠 ``KEYS[2+i]`` 定址
        keys.extend(self.keyspace.dimension_key(active_kind, bucket_day, axis) for axis in axes)
        keys.append(self.keyspace.day_index_key(active_kind))
        keys.append(self.keyspace.axis_index_key(active_kind))
        if write_tokens:
            keys.append(self.keyspace.tokens_bucket_key(active_kind, bucket_day))
            keys.extend(
                self.keyspace.tokens_dimension_key(active_kind, bucket_day, axis) for axis in axes
            )

        argv: list[Any] = [
            trace_id,
            _quantize(Decimal(str(amount))),
            dedup_ttl,
            ttl,
            index_ttl,
            bucket_day.isoformat(),
        ]
        if write_tokens:
            argv.extend([token_counts["cache_hit"], token_counts["cache_miss"], token_counts["output"]])
        # 每个维度轴一对 (轴名, 取值)：Lua 靠它既写对 field，也登记轴名
        for axis, value in axes.items():
            argv.extend([axis, value])

        script = IDEMPOTENT_ACCRUAL_LUA_V2 if write_tokens else IDEMPOTENT_ACCRUAL_LUA
        raw = self._eval("eval_accrual", script, keys, argv)
        if raw is None:
            # 存储不可用时返回「未入账」的结果，而不是伪造一个 0 元的成功
            return EntryResult(
                duplicate=False,
                applied_to=dimension,
                amount=Decimal(0),
                total=Decimal(0),
                day=bucket_day,
                kind=active_kind,
            )

        duplicate, applied_to, applied_amount, total = _parse_eval_array(raw, IDEMPOTENT_RESULT_KEYS)
        return EntryResult(
            duplicate=_to_int(duplicate) == 1,
            applied_to=str(applied_to),
            amount=Decimal(str(_to_float(applied_amount))),
            total=Decimal(str(_to_float(total))),
            day=bucket_day,
            kind=active_kind,
        )

    # -------- 读 --------
    def bucket_totals(self, day: date, *, kind: str | None = None) -> dict[str, Decimal]:
        """某一天的总账（``维度值 -> 金额``）。"""
        raw = self._run(
            "hgetall_bucket",
            self.client.hgetall,
            self.keyspace.bucket_key(kind or self.kind, day),
        )
        return {name: Decimal(value) for name, value in decode_map(raw).items()}

    def bucket_total(self, day: date, *, kind: str | None = None) -> Decimal:
        """某一天的总账金额（元）。"""
        return sum(self.bucket_totals(day, kind=kind).values(), Decimal(0))

    def dimension_totals(
        self,
        day: date,
        *,
        axis: str = DEFAULT_DIMENSION_AXIS,
        kind: str | None = None,
    ) -> dict[str, Decimal]:
        """某一天某个**维度轴**的分账，例如 ``axis="model"`` 得到「每个模型花了多少」。"""
        raw = self._run(
            "hgetall_dimension",
            self.client.hgetall,
            self.keyspace.dimension_key(kind or self.kind, day, axis),
        )
        return {name: Decimal(value) for name, value in decode_map(raw).items()}

    def axis_totals(
        self,
        day: date,
        *,
        kind: str | None = None,
        axes: Iterable[str] | None = None,
    ) -> dict[str, dict[str, Decimal]]:
        """某一天所有维度轴的分账（``轴 -> {取值 -> 金额}``）。"""
        target_axes = list(axes) if axes is not None else self.known_axes(kind=kind)
        return {axis: self.dimension_totals(day, axis=axis, kind=kind) for axis in target_axes}

    def known_axes(self, *, kind: str | None = None) -> list[str]:
        """出现过的维度轴（由 Lua 脚本在入账时登记）。"""
        raw = self._run(
            "smembers_axes", self.client.smembers, self.keyspace.axis_index_key(kind or self.kind)
        )
        if not raw:
            return []
        return sorted(str(decode(item)) for item in raw)

    def token_totals(
        self,
        day: date,
        *,
        axis: str = DEFAULT_DIMENSION_AXIS,
        kind: str | None = None,
    ) -> dict[str, Decimal]:
        """某一天某个维度轴的 token 三分类合计。

        默认取 ``model`` 轴；由于 token 值在同一个模型的各科条目上是同构的，
        先按轴合计再求和即可得到全天 token 量。
        """
        raw = self._run(
            "hgetall_tokens",
            self.client.hgetall,
            self.keyspace.tokens_dimension_key(kind or self.kind, day, axis),
        )
        return {name: Decimal(value) for name, value in decode_map(raw).items()}

    def token_grand_totals(self, day: date, *, kind: str | None = None) -> dict[str, Decimal]:
        """全天 token 总计（直接读 token 总账，不做跨轴求和）。"""
        raw = self._run(
            "hgetall_token_bucket",
            self.client.hgetall,
            self.keyspace.tokens_bucket_key(kind or self.kind, day),
        )
        return {name: Decimal(value) for name, value in decode_map(raw).items()}

    def known_days(self, *, kind: str | None = None) -> list[date]:
        """天索引里登记过的日期（升序）。"""
        raw = self._run("smembers", self.client.smembers, self.keyspace.day_index_key(kind or self.kind))
        if not raw:
            return []
        days: list[date] = []
        for item in raw:
            label = str(decode(item))
            try:
                days.append(date.fromisoformat(label))
            except ValueError:
                continue
        return sorted(days)

    def trace_seen(self, trace_id: str, day: date, *, kind: str | None = None) -> bool:
        """某个 trace_id 是否已入账（只用于测试与排障，不要放在热路径上）。"""
        raw = self._run(
            "sismember_dedup",
            self.client.smembers,
            self.keyspace.dedup_key(kind or self.kind, day),
        )
        if not raw:
            return False
        return trace_id in {str(decode(item)) for item in raw}


def _parse_eval_array(raw: Any, names: tuple[str, ...] | list[str]) -> list[Any]:
    """把 ``EVAL`` 返回的数组规整成固定长度的列表。

    真 Redis + redis-py 返回 ``list``，部分客户端返回 ``tuple``，
    ``None`` 代表脚本返回了 nil。这里统一处理，避免下游出现下标错误。
    """
    if raw is None:
        return [None] * len(names)
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    else:  # 单值
        values = [raw]
    while len(values) < len(names):
        values.append(None)
    return values[: len(names)]


# --------------------------------------------------------------------------------------
# 滑动窗口
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WindowStatus:
    """滑动窗口的一次查询结果。"""

    scope: str
    member: str
    now: float
    window_seconds: int
    max_calls: int
    count_before: int
    count_after: int
    exceeded: bool
    oldest_score: float | None
    day: date

    @property
    def count(self) -> int:
        return self.count_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "member": self.member,
            "now": self.now,
            "window_seconds": self.window_seconds,
            "max_calls": self.max_calls,
            "count_before": self.count_before,
            "count_after": self.count_after,
            "exceeded": self.exceeded,
            "oldest_score": self.oldest_score,
            "day": self.day.isoformat(),
        }


class SlidingWindowCounter(_RedisCommandMixin):
    """基于 Redis ZSET 的滑动窗口计数器。

    ``ZADD key score=时间戳 member=trace_id``，查询时先 ``ZREMRANGEBYSCORE``
    清掉窗口外的成员，再 ``ZCARD`` 计数，然后 ``ZADD`` 记录本次调用 ——
    三步在 :data:`SLIDING_WINDOW_LUA` 里原子完成。

    **窗口秒数与次数阈值是两个完全独立的参数**（``window_seconds`` / ``max_calls``）。
    旧版把 ``threshold=8``（次数）口头描述成「60 秒」，两个概念混为一谈，
    既看不出真实窗口，也没法分别调参。
    """

    def __init__(
        self,
        client: RedisLike,
        settings: Any,
        *,
        scope: str = "llm",
        keyspace: Keyspace | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.scope = scope
        self.keyspace = keyspace or keyspace_for(getattr(settings, "key_prefix", "llm"))

    def window_key(self, day: date) -> str:
        return self.keyspace.window_key(self.scope, day)

    def _bucket_day(self, moment: float, at: datetime | None, window_seconds: int) -> date:
        """窗口成员可能横跨零点，用「当前时刻 - 窗口」作为分桶锚点，避免误删未过期成员。"""
        if at is not None:
            return utc_day(at)
        return utc_day(datetime.fromtimestamp(moment - window_seconds))

    def record_and_count(
        self,
        member: str,
        *,
        now: float | None = None,
        window_seconds: int | None = None,
        max_calls: int | None = None,
        at: datetime | None = None,
    ) -> WindowStatus:
        """记录一次调用并返回窗口状态。

        :param window_seconds: 时间窗口（秒）；None 取 ``settings.retry_window_seconds``。
        :param max_calls: 窗口内允许的最大次数；None 取 ``settings.retry_max_calls``。
        :param now: 当前 UNIX 时间戳（秒，可为小数）。测试里注入以模拟时间流逝。
        """
        if not member:
            raise ValueError("member 不能为空：滑动窗口需要一个稳定的成员标识（通常是 trace_id）")

        active_window = int(
            window_seconds if window_seconds is not None else self.settings.retry_window_seconds
        )
        active_max = int(max_calls if max_calls is not None else self.settings.retry_max_calls)
        if active_window < 1:
            raise ValueError(f"window_seconds 必须 >= 1，收到 {active_window}")
        if active_max < 1:
            raise ValueError(f"max_calls 必须 >= 1，收到 {active_max}")

        moment = float(now) if now is not None else time_module.time()
        bucket_day = self._bucket_day(moment, at, active_window)
        ttl = max(int(self.settings.ttl_window_seconds), active_window * 2)

        raw = self._eval("eval_window", SLIDING_WINDOW_LUA, [self.window_key(bucket_day)], [
            moment,
            active_window,
            active_max,
            member,
            ttl,
        ])
        if raw is None:
            return WindowStatus(
                scope=self.scope,
                member=member,
                now=moment,
                window_seconds=active_window,
                max_calls=active_max,
                count_before=0,
                count_after=0,
                exceeded=False,
                oldest_score=None,
                day=bucket_day,
            )

        before, after, exceeded, oldest = _parse_eval_array(
            raw, ("count_before", "count_after", "exceeded", "oldest_score")
        )
        oldest_text = str(decode(oldest)) if oldest is not None else ""
        return WindowStatus(
            scope=self.scope,
            member=member,
            now=moment,
            window_seconds=active_window,
            max_calls=active_max,
            count_before=_to_int(before),
            count_after=_to_int(after),
            exceeded=_to_int(exceeded) == 1,
            oldest_score=_to_float(oldest_text) if oldest_text else None,
            day=bucket_day,
        )

    def count(
        self,
        *,
        now: float | None = None,
        window_seconds: int | None = None,
        at: datetime | None = None,
    ) -> int:
        """只读计数，不写入成员（``ZREMRANGEBYSCORE`` + ``ZCARD``，供报表使用）。"""
        active_window = int(
            window_seconds if window_seconds is not None else self.settings.retry_window_seconds
        )
        moment = float(now) if now is not None else time_module.time()
        key = self.window_key(self._bucket_day(moment, at, active_window))
        self._run("zremrangebyscore", self.client.zremrangebyscore, key, "-inf", moment - active_window)
        return _to_int(self._run("zcard", self.client.zcard, key))

    def members(
        self,
        *,
        now: float | None = None,
        window_seconds: int | None = None,
        limit: int = 100,
        at: datetime | None = None,
    ) -> list[str]:
        """返回窗口内的成员（按时间升序）。"""
        active_window = int(
            window_seconds if window_seconds is not None else self.settings.retry_window_seconds
        )
        moment = float(now) if now is not None else time_module.time()
        key = self.window_key(self._bucket_day(moment, at, active_window))
        self._run("zremrangebyscore", self.client.zremrangebyscore, key, "-inf", moment - active_window)
        raw = self._run("zrange", self.client.zrange, key, 0, max(limit - 1, 0))
        if not raw:
            return []
        return [str(decode(item)) for item in raw]

    def record_many(
        self,
        members: Iterable[str],
        *,
        now: float | None = None,
        at: datetime | None = None,
    ) -> int:
        """批量写入成员（用 pipeline 一次往返），返回写入条数。"""
        member_list = [item for item in members if item]
        if not member_list:
            return 0
        moment = float(now) if now is not None else time_module.time()
        day = utc_day(at) if at is not None else utc_day(datetime.fromtimestamp(moment))
        key = self.window_key(day)
        ttl = max(int(self.settings.ttl_window_seconds), int(self.settings.retry_window_seconds) * 2)

        def _write() -> int:
            pipe = self.client.pipeline()
            for index, member in enumerate(member_list):
                pipe.zadd(key, {member: moment + index * 1e-6})
            pipe.expire(key, ttl)
            pipe.execute()
            return len(member_list)

        return int(self._run("pipeline_record_many", _write) or 0)


@dataclass
class LedgerSnapshot:
    """一天的账目快照（便于对账/调试时一眼看全）。"""

    day: date
    total: Decimal
    by_axis: dict[str, dict[str, Decimal]] = field(default_factory=dict)
    tokens: dict[str, Decimal] = field(default_factory=dict)
    kind: str = "cost"

    @property
    def split_sum(self) -> Decimal:
        """主维度轴的分账之和。"""
        axis_totals = self.by_axis.get(DEFAULT_DIMENSION_AXIS, {})
        return sum(axis_totals.values(), Decimal(0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "kind": self.kind,
            "total": str(self.total),
            "by_axis": {
                axis: {name: str(value) for name, value in values.items()}
                for axis, values in self.by_axis.items()
            },
            "tokens": {name: str(value) for name, value in self.tokens.items()},
        }


def snapshot(ledger: IdempotentLedger, day: date) -> LedgerSnapshot:
    """对某一天做一次快照。"""
    return LedgerSnapshot(
        day=day,
        total=ledger.bucket_total(day),
        by_axis=ledger.axis_totals(day),
        tokens=ledger.token_grand_totals(day),
        kind=ledger.kind,
    )
