"""集中配置。

所有配置项都通过环境变量 ``CG_`` 前缀注入（pydantic-settings 的 ``env_prefix``），
默认值对本地开发友好：不配置任何环境变量也能导入、也能跑纯函数。

注意：本模块**不会**在导入期建立任何网络连接。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .pricing import PeakWindow, parse_peak_windows

__all__ = ["Settings", "get_settings", "reset_settings_cache"]


def _parse_csv_dates(value: object) -> object:
    """把 ``"2026-01-01,2026-01-02"`` 之类的逗号分隔串解析成元组。

    纯函数（不依赖 self），便于单独测试。
    """
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return value


def _parse_peak_windows(value: object) -> object:
    """把 ``"09:00-12:00,14:00-18:00"`` 解析成 :class:`PeakWindow` 元组。"""
    if isinstance(value, str):
        return parse_peak_windows(value)
    if isinstance(value, (list, tuple)):
        # 允许直接传 ["09:00-12:00"] 形式
        flat = ",".join(str(item) for item in value)
        return parse_peak_windows(flat)
    return value


CSVDateTuple = Annotated[tuple[str, ...], BeforeValidator(_parse_csv_dates)]
PeakWindowTuple = Annotated[tuple[PeakWindow, ...], BeforeValidator(_parse_peak_windows)]


class Settings(BaseSettings):
    """运行期配置。

    分组顺序：Redis → TTL → 时段判定 → 统计 → 滑动窗口 → 可观测性 → Celery。
    """

    model_config = SettingsConfigDict(
        env_prefix="CG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    # ---------------- Redis ----------------
    redis_url: str = Field(
        default="redis://127.0.0.1:6379/0",
        description="Redis 连接串；storage 层只使用很小的 Redis 命令表面。",
    )
    key_prefix: str = Field(default="llm", description="所有业务 key 的统一前缀。")

    # ---------------- TTL（秒）----------------
    ttl_seconds: int = Field(default=172_800, ge=60, description="按天分桶的计量 key TTL。")
    ttl_index_seconds: int = Field(
        default=691_200, ge=60, description="按天索引集合的 TTL，用于枚举历史分桶。"
    )
    ttl_dedup_seconds: int = Field(
        default=604_800,
        ge=60,
        description="trace_id 幂等去重集合 TTL，必须大于业务最长重试周期。",
    )
    ttl_window_seconds: int = Field(
        default=3_600, ge=1, description="滑动窗口 ZSET 的兜底 TTL（防止极端情况下 key 永存）。"
    )

    # ---------------- 时段判定 ----------------
    timezone: str = Field(
        default="Asia/Shanghai",
        description="高峰时段判定所用时区；官方按北京时间判定。",
    )
    peak_windows: PeakWindowTuple = Field(
        default_factory=lambda: parse_peak_windows("09:00-12:00,14:00-18:00"),
        description="高峰时段（左闭右开）区间列表。",
    )
    holidays: CSVDateTuple = Field(
        default=(),
        description=(
            "中国法定节假日日期（YYYY-MM-DD）。官方规则是节假日全天按空闲时段计价，"
            "但节假日安排逐年公布，本库不内置日历，需运维人工维护。"
        ),
    )

    # ---------------- 统计 ----------------
    min_sample_size: int = Field(
        default=30, ge=2, description="MAD 离群检测的最小样本量，低于该值返回“样本不足”。"
    )
    mad_threshold: float = Field(
        default=3.5, gt=0, description="modified z-score 阈值，默认 3.5（Iglewicz & Hoaglin）。"
    )
    reconcile_tolerance: float = Field(
        default=0.01,
        ge=0,
        description="对账允许的相对差异；超过该值即标记 mismatch=True。",
    )

    # ---------------- 滑动窗口 ----------------
    retry_window_seconds: int = Field(
        default=60,
        ge=1,
        description="无效重试识别的**时间窗口（秒）**。与次数阈值完全独立。",
    )
    retry_max_calls: int = Field(
        default=8,
        ge=1,
        description="窗口内允许的最大调用**次数**。与窗口秒数完全独立。",
    )

    # ---------------- 可观测性 ----------------
    log_level: str = Field(default="INFO", description="日志级别。")
    log_format: Literal["json", "console"] = Field(
        default="console", description="structlog 渲染格式。"
    )
    fail_on_storage_error: bool = Field(
        default=False,
        description="埋点写入失败时是否向上抛出。默认 False：成本埋点不应该拖垮业务请求。",
    )

    # ---------------- Celery（可选）----------------
    celery_broker_url: str | None = Field(
        default=None,
        description="为空则 build_celery_app() 返回 None，导入期不会连接任何 broker。",
    )
    celery_result_backend: str | None = Field(default=None)
    celery_queue: str = Field(default="costgov")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程级单例配置（读取环境变量与 ``.env``）。"""
    return Settings()


def reset_settings_cache() -> None:
    """清空配置缓存。测试里改了环境变量后需要调用它。"""
    get_settings.cache_clear()
