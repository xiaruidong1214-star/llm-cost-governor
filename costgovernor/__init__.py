"""llm-cost-governor：LLM 可观测性与成本治理中间件。

设计目标（也是相对旧版的三条主线）：

1. **单位与口径正确**：价格表按官方口径以「百万 tokens」为单位，区分高峰/空闲时段与缓存命中/未命中；
2. **算术可测试**：成本、分位数、MAD 离群、对账全部是纯函数，不依赖 Redis；
3. **写入可靠**：Redis 侧用 Lua 脚本做幂等入账与滑动窗口，所有 key 带 TTL 且按天分桶。
"""

from __future__ import annotations

__version__ = "2.0.0"

__all__ = ["__version__"]
