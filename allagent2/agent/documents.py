"""声明式状态契约（模仿 pi-main `defineDoc`）。

============================================================
为什么要有这个文件
------------------------------------------------------------
改造前：整个智能体状态是一个大 pydantic 模型，字段平铺。
后果（都是真实踩过的）：
  - 结构一改，Redis 里的旧 checkpoint 反序列化就 ValidationError；
  - 会话级（identity/记忆）、轮次级（round/phase）、成本级（usage）、
    可观测级（trace）混在一起，谁也说不清"这个字段该活多久"；
  - 想给 trace 加个字段，不知道会不会把 team 的 TeamState 校验搞炸；
  - 成本只有一个总数，看不出"哪个模型烧的钱、哪个工具调得最多"。

pi-main 的做法是：每个状态域是一个 **document**，声明式定义
    kind / version / scope / history / fork / initial / migrate / checkpointWhen
—— 独立版本、独立迁移、独立快照策略。这里把它搬到 Python。

============================================================
本项目的五个 document
------------------------------------------------------------
  kind                     scope           history      fork      语义
  ----------------------------------------------------------------------
  care.session             session         -            -         进程/部署级：引擎、拓扑、工具清单
  care.conversation        conversation    rewindable   as_of     跨轮存活：消息、记忆、身份、权限
  care.turn                turn            latest       initial   单轮：轮次、阶段、工具黑板、中断
  care.usage               conversation    latest       initial   成本账本：按模型 / 按工具分桶
  care.trace               conversation    rewindable   as_of     事件流：seq/耗时/token/事故

  scope      = 谁拥有它、活多久
  history    = latest（只留当前值）| rewindable（保留可回溯历史）
  fork       = 分叉会话时从哪抄：current（当前值）/ initial（初始值）/ as_of（某个历史点）
  checkpoint = 存完整基线 还是 增量 delta（本项目对应"要不要打快照锚点"）

============================================================
和 LangGraph 的关系
------------------------------------------------------------
LangGraph 一个 StateGraph 只能有一个 state schema，做不到 pi 那种多文档存储。
所以这里的做法是「**语义契约 + 单一扁平状态**」：
  - documents.py 定义契约、版本与迁移（本文件）；
  - CareAgentState 仍是扁平 TypedDict（react_loop / team / bootstrap 都在用），
    但字段按 document 分区注释，且整体带 `schema_version`；
  - 读 checkpoint 时先过 `migrate_state_dict()`，旧结构自动升级，
    不用再靠 try/except 吞 ValidationError。

这样既拿到了"独立版本 + 可迁移"的好处，又不用重写整个图。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

# 当前整体状态版本。每次给 CareAgentState 加/改字段都要 +1 并补迁移分支。
CURRENT_STATE_VERSION = 2

Scope = Literal["session", "conversation", "turn"]
History = Literal["latest", "rewindable"]
Fork = Literal["current", "initial", "as_of"]


# ============================================================
# 文档定义
# ============================================================
@dataclass(frozen=True)
class DocSpec:
    """一个状态域的完整契约。

    kind              稳定持久化标识（写进 checkpoint / 日志 / 指标，不要随便改）
    version           存储形态版本，正整数；改结构就 +1 并写 migrate
    scope             归属与生命周期
    history           latest=只留当前值；rewindable=保留可回溯历史
    fork              分叉时从哪抄
    initial           初始值工厂（必须返回纯 JSON 可序列化的 dict）
    migrate           旧版本 → 当前版本的升级函数；签名 (value, from_version) -> dict
    checkpoint_when   是否打快照锚点：签名 (value, info: dict) -> bool
    """

    kind: str
    version: int
    scope: Scope
    history: History
    fork: Fork
    initial: Callable[[], Dict[str, Any]]
    migrate: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None
    checkpoint_when: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or self.version < 1:
            raise TypeError(f"document {self.kind} 的 version 必须是正整数")
        if self.scope == "session" and (self.history != "latest" or self.fork != "initial"):
            # 与 pi 的校验保持一致：只有 conversation 文档才有历史/分叉语义
            raise TypeError(
                f"document {self.kind}: scope=session 不支持 history/fork 语义"
            )


# ============================================================
# 具体文档
# ============================================================
def _session_initial() -> Dict[str, Any]:
    return {
        "engine": "",          # react | create_agent
        "topology": "",        # single | team
        "tools": [],           # 本次进程装配到的工具名
        "prompt_version": "",  # system prompt 版本（改 prompt 时可追溯）
        "started_at": round(time.time(), 3),
    }


def _conversation_initial() -> Dict[str, Any]:
    return {
        "messages": [],
        "memory_block": "",
        "identity_id": None,
        "permissions": None,
        "current_elder_id": None,
    }


def _turn_initial() -> Dict[str, Any]:
    return {
        "round": 0,
        "phase": None,
        "tool_results": {},
        # 幂等：工具调用台账 + 本轮标识（care.turn 域，每轮重写）
        "tool_ledger": {},
        "turn_id": "",
        "emergency_flag": False,
        "identity_blocked": False,
        "interrupt": None,
        "resumed_from": None,
    }


def _usage_initial() -> Dict[str, Any]:
    """成本账本：按模型、按工具分桶（照 pi 的 UsageState 语义）。

    为什么不只留一个总数：真实排障时的问题是
      "这轮为什么贵" → 要看是哪个模型（主模型/摘要模型/Judge）烧的；
      "哪个工具最慢最贵" → 工具桶没有模型身份，必须单列。
    """
    return {
        "models": {},   # key = "provider/modelId"
        "tools": {},    # key = tool_name
        "turns": 0,     # 累计轮次数（分母，算单轮均价用）
    }


def _trace_initial() -> Dict[str, Any]:
    return {
        "events": [],
        "seq": 0,           # seq 水位线，续跑时从这里接着编号
        "incidents": [],    # 旁路事故（error / interrupted），不进 state 也能查
        "snapshots": [],    # 快照锚点（node / seq / ts / trace_len）
    }


def _usage_checkpoint_always(value: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """成本每次都存基线 —— 金额必须随时可精确读取，不能靠 delta 累加。"""
    return True


def _trace_checkpoint_when(value: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """事件流每 20 条、或出现事故时打一个快照锚点。

    理由：trace 是最容易无限膨胀的域。全量落盘没意义，但完全不留锚点
    又没法做"回到第 N 步"的复盘。20 是经验值（一轮对话约 15~25 条事件）。
    """
    events = value.get("events") or []
    if info.get("has_incident"):
        return True
    return len(events) % 20 == 0


def _migrate_trace(value: Dict[str, Any], from_version: int) -> Dict[str, Any]:
    """trace 老结构：直接是 list（没有 events/seq/incidents 外壳）。"""
    if isinstance(value, list):
        events = value
        seq = 0
        for event in events:
            if isinstance(event, dict):
                seq = max(seq, int(event.get("seq") or 0))
        return {"events": events, "seq": seq, "incidents": [], "snapshots": []}
    return _trace_initial()


DOCS: Dict[str, DocSpec] = {
    "care.session": DocSpec(
        kind="care.session",
        version=1,
        scope="session",
        history="latest",
        fork="initial",
        initial=_session_initial,
        description="进程/部署级装配信息：引擎、拓扑、工具清单、prompt 版本",
    ),
    "care.conversation": DocSpec(
        kind="care.conversation",
        version=1,
        scope="conversation",
        history="rewindable",
        fork="as_of",
        initial=_conversation_initial,
        description="跨轮存活：消息、长期记忆块、可信身份、权限上下文",
    ),
    "care.turn": DocSpec(
        kind="care.turn",
        version=1,
        scope="turn",
        history="latest",
        fork="initial",
        initial=_turn_initial,
        description="单轮：轮次、阶段、工具黑板、中断标记、续跑来源",
    ),
    "care.usage": DocSpec(
        kind="care.usage",
        version=1,
        scope="conversation",
        history="latest",
        fork="initial",
        initial=_usage_initial,
        checkpoint_when=_usage_checkpoint_always,
        description="成本账本：按 provider/model 与 tool 分桶",
    ),
    "care.trace": DocSpec(
        kind="care.trace",
        version=2,
        scope="conversation",
        history="rewindable",
        fork="as_of",
        initial=_trace_initial,
        migrate=_migrate_trace,
        checkpoint_when=_trace_checkpoint_when,
        description="事件流：seq 水位、耗时、token、事故、快照锚点",
    ),
}


def get_doc(kind: str) -> DocSpec:
    spec = DOCS.get(kind)
    if spec is None:
        raise KeyError(f"未注册的 document: {kind}（已注册：{sorted(DOCS)}）")
    return spec


def initial_doc(kind: str) -> Dict[str, Any]:
    return get_doc(kind).initial()


def should_checkpoint(kind: str, value: Dict[str, Any], info: Optional[Dict[str, Any]] = None) -> bool:
    """按文档策略判断"这次要不要打快照锚点"。未定义策略 = 从不主动打。"""
    spec = get_doc(kind)
    if spec.checkpoint_when is None:
        return False
    try:
        return bool(spec.checkpoint_when(value, info or {}))
    except Exception:  # 策略函数失败 → 打锚点（宁可多存，不可丢）
        return True


# ============================================================
# 整体状态迁移
# ============================================================
# 每个字段属于哪个 document —— 迁移时用来判断"这个键还认不认"
_FIELD_OWNER: Dict[str, str] = {
    # care.conversation
    "messages": "care.conversation",
    "memory_block": "care.conversation",
    "identity_id": "care.conversation",
    "permissions": "care.conversation",
    # care.turn
    "round": "care.turn",
    "phase": "care.turn",
    "tool_results": "care.turn",
    "tool_ledger": "care.turn",
    "turn_id": "care.turn",
    "emergency_flag": "care.turn",
    "identity_blocked": "care.turn",
    "interrupt": "care.turn",
    "resumed_from": "care.turn",
    # care.usage
    "usage_models": "care.usage",
    "usage_tools": "care.usage",
    "usage_turns": "care.usage",
    # care.trace
    "trace": "care.trace",
    "snapshot": "care.trace",
    "context_stats": "care.trace",
    # 其它（业务/元数据）
    "intent": "care.turn",
    "rag_used": "care.turn",
    "question": "care.conversation",
    "structured_response": "care.turn",
    "final": "care.turn",
}

# 明确不再使用、读到就丢的键（历史遗留）
_DROPPED_FIELDS = {"session", "diagnostic", "query_time", "update_time"}


def _v1_to_v2(raw: Dict[str, Any]) -> Dict[str, Any]:
    """v1 → v2：补 usage 分桶、快照/上下文统计、中断与续跑字段。"""
    out = dict(raw)
    # ① 旧的单一 usage 累计（如果有）拆成空分桶 —— 历史数据无法还原到模型粒度
    if "usage" in out and not out.get("usage_models"):
        out["usage_models"] = {}
        out["usage_tools"] = {}
    out.setdefault("usage_models", {})
    out.setdefault("usage_tools", {})
    out.setdefault("usage_turns", 0)
    # ② 快照 / 上下文统计 / 中断 / 续跑来源
    out.setdefault("snapshot", None)
    out.setdefault("context_stats", None)
    out.setdefault("interrupt", None)
    out.setdefault("resumed_from", None)
    # ③ trace 老结构（纯 list）包一层
    trace = out.get("trace")
    if isinstance(trace, list):
        out["trace"] = trace  # state 里 trace 仍是 list（operator.add 归约），保持不变
    elif trace is None:
        out["trace"] = []
    # ④ 丢历史垃圾字段（不报错，静默清理）
    for dead in _DROPPED_FIELDS:
        out.pop(dead, None)
    return out


_MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    1: _v1_to_v2,
}


def migrate_state_dict(raw: Any) -> Tuple[Dict[str, Any], int, List[str]]:
    """把任意版本的 checkpoint state 升到当前版本。

    返回 (new_state, from_version, notes)。
    设计原则：**任何异常都不抛出** —— 读旧 checkpoint 是运维路径，
    宁可丢掉无法识别的字段并记 note，也不能让一次会话 500。
    """
    notes: List[str] = []
    if not isinstance(raw, dict):
        return {}, CURRENT_STATE_VERSION, ["输入不是 dict，已重置为空状态"]

    version = raw.get("schema_version")
    if not isinstance(version, int) or version < 1:
        version = 1
        notes.append("缺少 schema_version，按 v1 处理")

    state = dict(raw)
    while version < CURRENT_STATE_VERSION:
        fn = _MIGRATIONS.get(version)
        if fn is None:
            notes.append(f"没有 v{version} 的迁移函数，停在原样")
            break
        try:
            state = fn(state)
            notes.append(f"v{version}→v{version + 1} 迁移完成")
        except Exception as exc:  # 迁移失败 → 保持原样 + 记录
            notes.append(f"v{version}→v{version + 1} 迁移失败：{exc}")
            break
        version += 1

    state["schema_version"] = CURRENT_STATE_VERSION
    state.setdefault("trace", [])
    if not isinstance(state.get("trace"), list):
        notes.append("trace 不是 list，已重置")
        state["trace"] = []
    return state, int(raw.get("schema_version") or 1), notes


# ============================================================
# 快照锚点
# ============================================================
def make_snapshot(
    *,
    node: str,
    batch_seq: int,
    trace_len: int,
    status: str = "ok",
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """打一个快照锚点（记录"第几步、走到哪、事件到第几条"）。

    与 pi 的 checkpoint 语义对应：这里不需要存完整值（LangGraph 的
    checkpoint 已经存了），只需要一个**可定位的锚点**，
    让 /agent/trace 能回答"想回到第 3 轮该从哪个 seq 开始"。
    """
    snap: Dict[str, Any] = {
        "node": node,
        "seq": batch_seq,
        "trace_len": trace_len,
        "status": status,
        "ts": round(time.time(), 3),
    }
    if detail:
        snap["detail"] = {k: v for k, v in detail.items() if v not in (None, "", [], {})}
    return snap
