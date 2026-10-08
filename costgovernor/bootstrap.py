"""依赖装配：集中创建 Redis 客户端与各组件，便于测试注入替身。

原则：**导入期不连接任何外部服务**。
``Container`` 里的所有东西都是惰性创建的：只有在真正用到 ``ledger`` / ``windows`` 时
才会去构造 Redis 客户端，构造客户端本身也不会发起网络 IO（``redis.Redis`` 是惰性连接）。
只有显式调用 :meth:`Container.ping` 才会真正往返一次。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .settings import Settings, get_settings
from .storage import IdempotentLedger, Keyspace, SlidingWindowCounter, keyspace_for
from .tracker import get_logger, structlog_configure

__all__ = ["Container", "build_container", "build_redis_client"]


def build_redis_client(settings: Settings, *, client: Any | None = None) -> Any:
    """构造 Redis 客户端。

    :param client: 已存在的客户端（例如 ``fakeredis.FakeRedis`` 或测试替身）。
        给了就直接返回，不会新建连接 —— 这是测试注入的入口。
    """
    if client is not None:
        return client

    import redis  # 延迟导入：不使用 Redis 的场景（纯函数计算、CLI price-check）无需依赖它

    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=2.0,
        socket_connect_timeout=2.0,
        health_check_interval=30,
    )


@dataclass
class Container:
    """依赖容器。

    只做装配，不含业务逻辑；所有属性都是惰性求值，
    因此 ``Container(settings)`` 在测试里可以零成本地构造。

    用法::

        container = Container(client=fake_redis)   # 测试：注入替身
        container.ledger.add_entry("trace-1", 0.5, dimension="deepseek-flash")

        container = Container()                    # 生产：按 settings.redis_url 连接
    """

    settings: Settings = field(default_factory=get_settings)
    client: Any | None = None
    scope: str = "llm"
    _keyspace: Keyspace | None = field(default=None, init=False, repr=False)
    _ledger: IdempotentLedger | None = field(default=None, init=False, repr=False)
    _windows: dict[str, SlidingWindowCounter] = field(default_factory=dict, init=False, repr=False)
    _logger: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.settings.log_level:
            structlog_configure(level=self.settings.log_level, fmt=self.settings.log_format)

    # -------- 基础 --------
    @property
    def keyspace(self) -> Keyspace:
        """key 命名器。"""
        if self._keyspace is None:
            self._keyspace = keyspace_for(self.settings.key_prefix)
        return self._keyspace

    @property
    def logger(self) -> Any:
        if self._logger is None:
            self._logger = get_logger("costgovernor.container")
        return self._logger

    @property
    def redis(self) -> Any:
        """Redis 客户端（惰性创建；构造本身不发起网络 IO）。"""
        if self.client is None:
            self.client = build_redis_client(self.settings)
        return self.client

    # -------- 组件 --------
    @property
    def ledger(self) -> IdempotentLedger:
        """幂等账本。"""
        if self._ledger is None:
            self._ledger = IdempotentLedger(self.redis, self.settings, keyspace=self.keyspace)
        return self._ledger

    def windows(self, scope: str | None = None) -> SlidingWindowCounter:
        """滑动窗口计数器（按 scope 缓存实例）。"""
        active_scope = scope or self.scope
        if active_scope not in self._windows:
            self._windows[active_scope] = SlidingWindowCounter(
                self.redis, self.settings, scope=active_scope, keyspace=self.keyspace
            )
        return self._windows[active_scope]

    # -------- 运维 --------
    def ping(self) -> bool:
        """真正探测一次 Redis 连通性。

        只有这里会发起网络 IO；返回 ``False`` 而不是抛异常，方便 CLI 做降级提示。
        """
        try:
            return bool(self.redis.ping())
        except Exception as exc:  # noqa: BLE001 - 探活失败不应该让 CLI 崩掉
            self.logger.warning("redis_ping_failed", error=repr(exc), url=self.settings.redis_url)
            return False

    def close(self) -> None:
        """关闭客户端（若底层支持）。"""
        if self.client is not None:
            close = getattr(self.client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("redis_close_failed", error=repr(exc))

    def __enter__(self) -> Container:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def build_container(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
    scope: str = "llm",
) -> Container:
    """便捷构造函数。"""
    return Container(settings=settings or get_settings(), client=client, scope=scope)
