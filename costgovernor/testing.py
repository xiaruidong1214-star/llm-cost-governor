"""内存版 Redis 替身（仅用于演示、本地试用与测试）。

**它不是 Redis，也不打算成为 Redis。** 它只实现 ``storage.py`` 声明的那一小块命令表面，
好处有两个：

* ``costgovernor.cli --client fake`` 在没有 Redis、也没装 ``lupa`` 的机器上依然能演示
  完整的「幂等入账 + 滑动窗口」流程；
* 测试可以拿它当基准替身（``tests/conftest.py`` 里的 ``FakeRedis`` 继承它，
  再加时钟控制、TTL 查询、``eval`` 调用记录等测试专用能力）。

关于 ``eval``：这里**不是 Lua 解释器**，而是按脚本文本的标记注释分派到等价的 Python 实现。
真正的 Lua 语法/语义由真实 Redis 负责；本模块负责把「幂等」「窗口清理」「TTL 续期」
这些**契约**变成可执行的断言。未登记的脚本一律抛 ``NotImplementedError``，
避免出现「脚本没生效但程序照样跑」的静默错误。
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping
from typing import Any

from .storage import (
    IDEMPOTENT_ACCRUAL_LUA,
    IDEMPOTENT_ACCRUAL_LUA_V2,
    SLIDING_WINDOW_LUA,
)

__all__ = ["InMemoryRedis", "InMemoryPipeline", "decode_value"]


def decode_value(value: Any) -> Any:
    """``bytes`` → ``str``；其他类型原样返回。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _parse_score_bound(value: Any) -> float:
    """解析 ZSET 分数边界（支持 ``-inf`` / ``+inf``）。"""
    text = str(decode_value(value)).strip().lower()
    if text in ("-inf", "inf", "+inf"):
        return float("-inf") if text == "-inf" else float("inf")
    return float(text)


class InMemoryRedis:
    """极小的 Redis 替身，使用 ``decode_responses=True`` 的语义。"""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.expires: dict[str, float] = {}
        self.closed = False
        self.clock = 0.0
        #: 每次 EVAL 的 (脚本, 参数) 与 KEYS 个数，便于测试断言「确实走了原子脚本」
        self.eval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.eval_numkeys: list[int] = []

    # ---------------- 内部 ----------------
    def _tick(self) -> float:
        return self.clock

    def advance(self, seconds: float) -> None:
        """把内部时钟向前推（用于验证 TTL 是否真的会过期）。"""
        self.clock += seconds

    def _expired(self, key: Any) -> bool:
        name = str(decode_value(key))
        deadline = self.expires.get(name)
        if deadline is None:
            return False
        if deadline <= self._tick():
            self.store.pop(name, None)
            self.expires.pop(name, None)
            return True
        return False

    def _purge_expired(self) -> None:
        for name in list(self.expires):
            self._expired(name)

    def _typed(self, key: Any, kind: str, *, create: bool = False) -> dict[Any, Any] | None:
        name = str(decode_value(key))
        self._expired(name)
        if name not in self.store:
            if not create:
                return None
            self.store[name] = {}
        value = self.store[name]
        if not isinstance(value, dict):
            raise TypeError(f"{name} 不是 {kind}（实际类型 {type(value).__name__}）")
        return value

    def _hash(self, key: Any, *, create: bool = False) -> dict[str, str] | None:
        return self._typed(key, "Hash", create=create)  # type: ignore[return-value]

    def _set(self, key: Any, *, create: bool = False) -> dict[str, None] | None:
        return self._typed(key, "Set", create=create)  # type: ignore[return-value]

    def _zset(self, key: Any, *, create: bool = False) -> dict[str, float] | None:
        return self._typed(key, "ZSET", create=create)  # type: ignore[return-value]

    # ---------------- 字符串 ----------------
    def get(self, key: Any) -> Any:
        self._expired(key)
        return self.store.get(str(decode_value(key)))

    def set(self, key: Any, value: Any) -> bool:
        name = str(decode_value(key))
        self.store[name] = str(decode_value(value))
        self.expires.pop(name, None)
        return True

    def setex(self, key: Any, ttl: Any, value: Any) -> bool:
        name = str(decode_value(key))
        self.store[name] = str(decode_value(value))
        self.expires[name] = self._tick() + float(ttl)
        return True

    # ---------------- Hash ----------------
    def hset(self, key: Any, field: Any = None, value: Any = None, mapping: Mapping[Any, Any] | None = None) -> int:
        target = self._hash(key, create=True)
        assert target is not None
        added = 0
        pairs: dict[Any, Any] = dict(mapping or {})
        if field is not None:
            pairs[field] = value
        for name, item in pairs.items():
            field_name = str(decode_value(name))
            if field_name not in target:
                added += 1
            target[field_name] = str(decode_value(item))
        return added

    def hsetnx(self, key: Any, field: Any, value: Any) -> int:
        target = self._hash(key, create=True)
        assert target is not None
        name = str(decode_value(field))
        if name in target:
            return 0
        target[name] = str(decode_value(value))
        return 1

    def hget(self, key: Any, field: Any) -> Any:
        target = self._hash(key)
        if not target:
            return None
        return target.get(str(decode_value(field)))

    def hincrbyfloat(self, key: Any, field: Any, amount: Any) -> float:
        target = self._hash(key, create=True)
        assert target is not None
        name = str(decode_value(field))
        updated = float(target.get(name, "0") or 0.0) + float(amount)
        target[name] = repr(updated)
        return updated

    def hgetall(self, key: Any) -> dict[str, str]:
        target = self._hash(key)
        return dict(target) if target else {}

    # ---------------- Set ----------------
    def sadd(self, key: Any, *members: Any) -> int:
        target = self._set(key, create=True)
        assert target is not None
        added = 0
        for member in members:
            name = str(decode_value(member))
            if name not in target:
                target[name] = None
                added += 1
        return added

    def smembers(self, key: Any) -> set[str]:
        target = self._set(key)
        return set(target) if target else set()

    def sismember(self, key: Any, member: Any) -> bool:
        target = self._set(key)
        if not target:
            return False
        return str(decode_value(member)) in target

    # ---------------- Sorted Set ----------------
    def zadd(self, key: Any, mapping: Mapping[Any, Any]) -> int:
        target = self._zset(key, create=True)
        assert target is not None
        added = 0
        for member, score in mapping.items():
            name = str(decode_value(member))
            if name not in target:
                added += 1
            target[name] = float(score)
        return added

    def zremrangebyscore(self, key: Any, minimum: Any, maximum: Any) -> int:
        target = self._zset(key)
        if not target:
            return 0
        low = _parse_score_bound(minimum)
        high = _parse_score_bound(maximum)
        doomed = [name for name, score in target.items() if low <= score <= high]
        for name in doomed:
            del target[name]
        return len(doomed)

    def zcard(self, key: Any) -> int:
        target = self._zset(key)
        return len(target) if target else 0

    def zrange(self, key: Any, start: Any, end: Any, withscores: bool = False) -> list[Any]:
        target = self._zset(key)
        if not target:
            return []
        ordered = sorted(target.items(), key=lambda item: (item[1], item[0]))
        start_index = int(start)
        end_index = int(end)
        if end_index < 0:
            end_index = len(ordered) + end_index
        window = ordered[start_index : end_index + 1]
        if withscores:
            flat: list[Any] = []
            for member, score in window:
                flat.extend([member, str(score)])
            return flat
        return [member for member, _score in window]

    # ---------------- TTL / key ----------------
    def expire(self, key: Any, ttl: Any) -> bool:
        name = str(decode_value(key))
        if self._expired(name) or name not in self.store:
            return False
        self.expires[name] = self._tick() + float(ttl)
        return True

    def ttl(self, key: Any) -> int:
        """``-1`` 表示存在但无 TTL，``-2`` 表示不存在。"""
        name = str(decode_value(key))
        if self._expired(name) or name not in self.store:
            return -2
        deadline = self.expires.get(name)
        if deadline is None:
            return -1
        return int(deadline - self._tick())

    def keys(self, pattern: str = "*") -> list[str]:
        self._purge_expired()
        return sorted(name for name in self.store if fnmatch.fnmatchcase(name, pattern))

    def flushall(self) -> bool:
        self.store.clear()
        self.expires.clear()
        return True

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True

    def pipeline(self, transaction: bool = True) -> InMemoryPipeline:
        return InMemoryPipeline(self)

    # ---------------- EVAL ----------------
    def eval(self, script: str, numkeys: int, *args: Any, **kwargs: Any) -> Any:
        """``EVAL`` 的**替身**实现：按标记注释分派到等价 Python 逻辑（不是 Lua 解释器）。"""
        values = list(args) + list(kwargs.values())
        self.eval_calls.append((script, tuple(values)))
        nkeys = int(numkeys)
        self.eval_numkeys.append(nkeys)
        keys = [str(decode_value(item)) for item in values[:nkeys]]
        argv = list(values[nkeys:])

        if script is IDEMPOTENT_ACCRUAL_LUA or script is IDEMPOTENT_ACCRUAL_LUA_V2:
            return self.eval_accrual(script, keys, argv)
        if script is SLIDING_WINDOW_LUA:
            return self.eval_window(keys, argv)
        raise NotImplementedError(
            "InMemoryRedis.eval 只实现了本项目登记的 Lua 脚本语义；"
            f"收到未登记的脚本：{script.strip().splitlines()[:1]}"
        )

    def eval_accrual(self, script: str, keys: list[str], argv: list[Any]) -> list[Any]:
        """幂等入账脚本的语义（与 storage 里的 KEYS/ARGV 布局一一对应）。"""
        with_tokens = script is IDEMPOTENT_ACCRUAL_LUA_V2
        naxes = (len(keys) - 5) // 2 if with_tokens else len(keys) - 4

        trace_id = str(decode_value(argv[0]))
        amount = str(decode_value(argv[1]))
        dedup_ttl = float(argv[2])
        ttl = float(argv[3])
        index_ttl = float(argv[4])
        day_label = str(decode_value(argv[5]))
        offset = 9 if with_tokens else 6
        axes = {
            str(decode_value(argv[offset + 2 * index])): str(decode_value(argv[offset + 2 * index + 1]))
            for index in range(naxes)
        }
        primary = next(iter(axes.values()))

        dedup_key, total_key = keys[0], keys[1]
        axis_keys = keys[2 : 2 + naxes]
        day_index_key = keys[2 + naxes]
        axis_index_key = keys[3 + naxes]
        token_total_key = keys[4 + naxes] if with_tokens else None
        token_axis_keys = keys[5 + naxes : 5 + 2 * naxes] if with_tokens else []

        if self.sadd(dedup_key, trace_id) == 0:
            existing = self._hash(total_key) or {}
            return [1, primary, "0", existing.get(primary, "0")]

        self.expire(dedup_key, dedup_ttl)
        total = self.hincrbyfloat(total_key, primary, amount)
        self.expire(total_key, ttl)
        # strict=True：轴 key 与轴取值必须等长。长度不匹配说明 key 布局算错了，
        # 应当在测试里立刻暴露，而不是静默截断到较短的一方。
        for axis_key, axis_value in zip(axis_keys, axes.values(), strict=True):
            self.hincrbyfloat(axis_key, axis_value, amount)
            self.expire(axis_key, ttl)
        self.sadd(day_index_key, day_label)
        self.expire(day_index_key, index_ttl)
        for axis_name in axes:
            self.sadd(axis_index_key, axis_name)
        self.expire(axis_index_key, index_ttl)

        if with_tokens:
            cache_hit, cache_miss, output = (float(item) for item in argv[6:9])
            for field, value in (
                ("cache_hit_tokens", cache_hit),
                ("cache_miss_tokens", cache_miss),
                ("output_tokens", output),
            ):
                if value > 0:
                    self.hincrbyfloat(token_total_key, field, str(value))
                    for axis_key in token_axis_keys:
                        self.hincrbyfloat(axis_key, field, str(value))
            self.expire(token_total_key, ttl)
            for axis_key in token_axis_keys:
                self.expire(axis_key, ttl)

        return [0, primary, amount, repr(total)]

    def eval_window(self, keys: list[str], argv: list[Any]) -> list[Any]:
        """滑动窗口脚本的语义（与 storage.SLIDING_WINDOW_LUA 一一对应）。"""
        key = keys[0]
        now = float(argv[0])
        window = float(argv[1])
        max_calls = float(argv[2])
        member = str(decode_value(argv[3]))
        ttl = float(argv[4])

        self.zremrangebyscore(key, "-inf", now - window)
        before = self.zcard(key)
        exceeded = 1 if before >= max_calls else 0
        self.zadd(key, {member: now})
        after = self.zcard(key)
        oldest = self.zrange(key, 0, 0, withscores=True)
        oldest_score = oldest[1] if oldest else ""
        self.expire(key, ttl)
        return [before, after, exceeded, oldest_score]


class InMemoryPipeline:
    """``pipeline()`` 的替身：命令入队，``execute`` 时按顺序执行。"""

    def __init__(self, client: InMemoryRedis) -> None:
        self.client = client
        self.queue: list[tuple[str, tuple[Any, ...]]] = []

    def zadd(self, key: Any, mapping: Mapping[Any, Any]) -> InMemoryPipeline:
        self.queue.append(("zadd", (key, mapping)))
        return self

    def expire(self, key: Any, ttl: Any) -> InMemoryPipeline:
        self.queue.append(("expire", (key, ttl)))
        return self

    def execute(self) -> list[Any]:
        results = [getattr(self.client, name)(*args) for name, args in self.queue]
        self.queue.clear()
        return results


def supports_eval(client: Any) -> bool:
    """探测一个客户端是否真的支持 ``EVAL``（fakeredis 需要 lupa，通常没有）。"""
    try:
        client.eval("return 1", 0)
        return True
    except Exception:  # noqa: BLE001 - 不支持就是 False，不需要区分具体异常
        return False


def collect_axis_names(axis_keys: Iterable[str]) -> set[str]:
    """辅助：从维度轴 key（``llm:cost:2026-10-08:model``）里取出轴名。"""
    return {key.rsplit(":", 1)[-1] for key in axis_keys}
