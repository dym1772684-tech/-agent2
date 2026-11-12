"""可观测性组件（Prometheus 指标）。

    metrics.py  指标定义、meter sink 注入、/metrics ASGI app、HTTP 埋点

请求级追踪 ID 在 core/request_id.py（属基础设施，路由/中间件都要用）。
"""
from . import metrics  # noqa: F401

__all__ = ["metrics"]
