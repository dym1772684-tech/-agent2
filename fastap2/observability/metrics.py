"""Prometheus 指标（fastap2 侧实现，通过 hook 注入 allagent2）。

============================================================
设计约束
------------------------------------------------------------
1. allagent2 不能 import fastap2 → 指标实现必须由 fastap2 注入
   （`allagent2.guance.metrics_hook.set_sink`）。
2. prometheus-client 是**软依赖**：没装也要能起服务，只是 /metrics 不挂。
3. label 集合固定且有限：node/tool/rule 都是枚举，不会造成 label 爆炸；
   HTTP 的 path 用**路由模板**（`/api/v1/agent/chat`）而不是原始 URL。
4. 埋点位置只有一个：`observe.emit_event` 的事件出口，所以这里只要
   把名字映射对就行，业务节点零改动。

============================================================
指标清单
------------------------------------------------------------
  agent_node_seconds{node}            histogram  节点耗时
  agent_node_total{node}              counter    节点执行次数
  agent_node_errors_total{node}       counter    节点异常次数
  agent_llm_seconds                   histogram  单次模型调用耗时
  agent_llm_tokens_total{kind}        counter    input/output/cache_read
  agent_llm_cost_yuan_total           counter    累计成本（元）
  agent_tool_seconds{tool}            histogram  工具耗时
  agent_tool_total{tool,status}       counter    工具调用次数（ok/error）
  agent_guard_total{rule}             counter    规则命中（emergency/identity/slot）
  agent_interrupt_total{node}         counter    中断次数
  agent_structured_degraded_total     counter    结构化输出降级次数
  rag_search_seconds                  histogram  单次检索耗时
  agent_http_seconds{method,path,status} histogram  HTTP 接口耗时
  agent_mcp_tools                     gauge      已连接 MCP 工具数
  agent_checkpointer_persistent       gauge      1=Redis 持久化 0=内存
  agent_topology_info{topology,engine} gauge     当前拓扑（值恒为 1）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("agent.metrics")

_AVAILABLE = False
_REGISTRY: Dict[str, Any] = {}
_KINDS: Dict[str, str] = {}       # name -> counter|histogram|gauge
_LABELS: Dict[str, tuple] = {}    # name -> labelnames

try:  # 软依赖：缺包时 install 返回 False，服务照常启动
    from prometheus_client import Counter, Gauge, Histogram

    _AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _AVAILABLE = False
    logger.warning("[metrics] prometheus-client 未安装，/metrics 不可用：%s", exc)


def _def(name: str, kind: str, labels: tuple = (), doc: str = "") -> None:
    if not _AVAILABLE or name in _REGISTRY:
        return
    try:
        if kind == "counter":
            metric = Counter(name, doc or name, labelnames=list(labels))
        elif kind == "histogram":
            metric = Histogram(name, doc or name, labelnames=list(labels))
        else:
            metric = Gauge(name, doc or name, labelnames=list(labels))
        _REGISTRY[name] = metric
        _KINDS[name] = kind
        _LABELS[name] = labels
    except Exception as exc:  # 重复注册 / 名字非法
        logger.warning("[metrics] 指标 %s 注册失败：%s", name, exc)


def _define_all() -> None:
    _def("agent_node_seconds", "histogram", ("node",), "智能体节点耗时")
    _def("agent_node_total", "counter", ("node",), "节点执行次数")
    _def("agent_node_errors_total", "counter", ("node",), "节点异常次数")
    _def("agent_llm_seconds", "histogram", (), "单次模型调用耗时")
    _def("agent_llm_tokens_total", "counter", ("kind",), "模型 token 用量")
    _def("agent_llm_cost_yuan_total", "counter", (), "累计成本（元）")
    _def("agent_tool_seconds", "histogram", ("tool",), "工具调用耗时")
    _def("agent_tool_total", "counter", ("tool", "status"), "工具调用次数")
    _def("agent_guard_total", "counter", ("rule",), "规则命中次数")
    _def("agent_interrupt_total", "counter", ("node",), "中断次数")
    _def("agent_structured_degraded_total", "counter", (), "结构化输出降级次数")
    _def("rag_search_seconds", "histogram", (), "检索耗时")
    _def("agent_http_seconds", "histogram", ("method", "path", "status"), "HTTP 耗时")
    _def("agent_mcp_tools", "gauge", (), "已连接 MCP 工具数")
    _def("agent_checkpointer_persistent", "gauge", (), "checkpointer 是否持久化")
    _def("agent_topology_info", "gauge", ("topology", "engine"), "当前拓扑")
    # 【降级可观测】所有"吞异常 + 降级继续"的位置都计数，否则排查只能翻日志
    _def("agent_degraded_total", "counter", ("component", "reason"),
         "降级/兜底触发次数（component 见 note_degradation）")


# ============================================================
# 降级计数（统一入口）
# ============================================================
def note_degradation(component: str, reason: str, detail: Any = "", log: bool = True) -> None:
    """记一次降级：计数 + 一条 warning。永不影响主流程。

    component 是固定枚举（redis/mcp/checkpointer/memory/skill/audit/llm/rag_*），
    reason 用**异常类名**而不是异常正文——否则 label 会随消息内容无限膨胀。
    用法：`except Exception as exc: note_degradation("redis", type(exc).__name__, exc)`
    log=False 用于高频路径（缓存读写）：只计数不刷日志。
    """
    reason = (reason or "unknown")[:40]
    try:
        metric = _REGISTRY.get("agent_degraded_total")
        if metric is not None:
            metric.labels(component=component, reason=reason).inc()
    except Exception:
        pass
    if not log:
        return
    try:
        logger.warning("[degraded] component=%s reason=%s detail=%s",
                       component, reason, str(detail)[:160])
    except Exception:
        pass


def _dispatch(name: str, value: float, labels: Dict[str, Any]) -> None:
    """meter() 的落点：按名字查表，未知名字忽略（前向兼容）。"""
    metric = _REGISTRY.get(name)
    if metric is None:
        return
    kind = _KINDS.get(name)
    allowed = _LABELS.get(name) or ()
    clean = {k: str(v) for k, v in labels.items() if k in allowed}
    # label 缺失会让 prometheus 抛错，这里补 "-" 兜底
    for key in allowed:
        clean.setdefault(key, "-")
    try:
        target = metric.labels(**clean) if allowed else metric
        if kind == "counter":
            target.inc(value)
        elif kind == "histogram":
            target.observe(value)
        else:
            target.set(value)
    except Exception:
        pass


def install() -> bool:
    """注册指标并注入 allagent2 的指标钩子。返回是否可用。"""
    if not _AVAILABLE:
        return False
    try:
        _define_all()
        from allagent2.guance.metrics_hook import set_sink

        set_sink(_dispatch)
        logger.info("[metrics] 指标已注册并注入 allagent2（%d 个）", len(_REGISTRY))
        return True
    except Exception as exc:
        logger.warning("[metrics] 指标注入失败（不影响主链路）：%s", exc)
        return False


# ============================================================
# 进程状态类 Gauge（bootstrap 之后填一次）
# ============================================================
def publish_runtime_state() -> None:
    """把装配结果写进 gauge，重启/换拓扑后一眼能看出来。"""
    if not _AVAILABLE:
        return
    try:
        from agent_bridge import get_status

        status = get_status()
        if "agent_mcp_tools" in _REGISTRY:
            _REGISTRY["agent_mcp_tools"].set(int(status.get("tools_count") or 0))
        if "agent_checkpointer_persistent" in _REGISTRY:
            _REGISTRY["agent_checkpointer_persistent"].set(
                1 if status.get("persistent") else 0
            )
        if "agent_topology_info" in _REGISTRY:
            _REGISTRY["agent_topology_info"].labels(
                topology=str(status.get("topology") or "unknown"),
                engine=str(status.get("engine") or "unknown"),
            ).set(1)
    except Exception as exc:
        logger.warning("[metrics] 运行状态发布失败：%s", exc)


# ============================================================
# HTTP 中间件用
# ============================================================
def observe_http(method: str, path: str, status: int, seconds: float) -> None:
    metric = _REGISTRY.get("agent_http_seconds")
    if metric is None:
        return
    try:
        metric.labels(method=method.upper(), path=path, status=str(status)).observe(seconds)
    except Exception:
        pass


def asgi_app() -> Optional[Any]:
    """给 main.py 挂载用的 /metrics ASGI app；不可用时返回 None。"""
    if not _AVAILABLE:
        return None
    try:
        from prometheus_client import make_asgi_app

        return make_asgi_app()
    except Exception as exc:
        logger.warning("[metrics] 无法创建 /metrics 应用：%s", exc)
        return None
