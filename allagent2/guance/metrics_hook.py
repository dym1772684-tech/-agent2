"""指标钩子：allagent2 侧的"只发不认"埋点口。

============================================================
为什么是钩子而不是直接 import prometheus_client
------------------------------------------------------------
1. 依赖方向：fastap2 依赖 allagent2，反过来不行。指标的具体实现
   （Prometheus 注册表、label 名称）属于部署关注点，应该由 fastap2 决定。
2. 可用性：单独跑 allagent2（测试、脚本、离线评测）时不应该被
   prometheus_client 绑架；没装包时 meter 必须是空操作。
3. 可替换：换成 OTel metrics / 自建统计 / 直接打日志，只改 sink，不动埋点。

用法（fastap2 侧）：
    from allagent2.guance.metrics_hook import set_sink
    set_sink(my_dispatch)      # 签名 (name: str, value: float, labels: dict) -> None

用法（allagent2 侧）：
    meter("agent_node_seconds", 1.234, node="think")
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

Sink = Callable[[str, float, Dict[str, Any]], None]

_SINK: Optional[Sink] = None


def set_sink(fn: Optional[Sink]) -> None:
    """注入指标实现；传 None 关闭。"""
    global _SINK
    _SINK = fn


def has_sink() -> bool:
    return _SINK is not None


def meter(name: str, value: float = 1.0, **labels: Any) -> None:
    """打一个点。任何失败都被吞掉 —— 指标永远不能影响主链路。"""
    sink = _SINK
    if sink is None:
        return
    try:
        sink(name, float(value), labels)
    except Exception:
        pass
