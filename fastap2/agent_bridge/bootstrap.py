"""集成引导层（胶水层）核心。

职责：
1. 定位并加载两个项目的 .env 配置；
2. 在 sys.path / sys.modules 中把 allagent2 目录以包名 `allagent2` 注册，
   使其内部所有的 `from allagent2.xxx import ...` 全部生效（与目录名一致，
   IDE/Pylance 也能直接解析这些导入）；
3. 在导入 graph 之前，把后端只读能力（查老人档案 / 查健康记录）动态注入
   allagent2 的工具列表（graph 顶层 create_tool_model 会 bind_tools）；
4. 编译 LangGraph，并提供与原 graph.py 交互入口一致的 run_agent 调用。
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import threading  # 【健壮性】用于首次引导的线程锁
import types
from pathlib import Path
from typing import Any, Dict

# 【安全】链路过程日志：整个 Agent 数据链路（节点/消息/工具调用/元数据）只允许出现在日志里，
# 绝不回传前端；前端只能看到智能体的最终回答。
# 【健壮性】agent_bridge 自带控制台 handler（与 allagent2 的 logging_helper 对齐），
# 不依赖 root logger 配置，确保 [chain] 链路日志在服务进程里始终可见。
logger = logging.getLogger("agent_bridge")
if not logger.handlers:
    _chain_handler = logging.StreamHandler()
    _chain_handler.setLevel(logging.INFO)
    _chain_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_chain_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

def _note_degraded(component: str, exc: BaseException, detail: str = "") -> None:
    """【降级计数】统一的降级埋点：Prometheus 计数 + 日志（失败静默）。

    之前所有降级都是"except → logger.warning"，出事只能翻日志，
    没法回答"这次到底是 MCP 挂了还是 Redis 挂了、多久一次"。
    """
    try:
        from observability.metrics import note_degradation

        note_degradation(component, type(exc).__name__, detail or exc)
    except Exception:
        pass


# ---- 路径定位 ----
# 本文件：fastap2/agent_bridge/bootstrap.py
FASTAP2_DIR = Path(__file__).resolve().parent.parent
# 两个项目平级：练手agent/fastap2 与 练手agent/allagent2
DEFAULT_AGENT_DIR = FASTAP2_DIR.parent / "allagent2"
AGENT_DIR = Path(os.environ.get("ALLAGENT_DIR", str(DEFAULT_AGENT_DIR))).resolve()

# 结构化字段清单统一以 allagent2.agent.state.PARSED_FIELDS 为准
# （仅养生养老字段，见 state.py；原旅游字段已删除）
PARSED_FIELDS = None  # 占位：在 bootstrap() 后从 state 模块导入真实清单

_BOOTED_FIELDS = []  # 真实字段清单，bootstrap 时填充


class _State:
    ready: bool = False
    agent: Any = None
    graph_mod: Any = None
    tools_count: int = 0
    persistent: bool = False  # checkpointer 是否走 Redis（否则重启丢会话）
    topology: str = "single"  # 【多智能体】single=单 agent | team=黑板多智能体
    error: BaseException | None = None


_state = _State()

# 【健壮性】首次引导锁：服务启动或首批并发请求可能同时触发 bootstrap，
# 不加锁会重复加载模型、重复编译 graph，甚至产生相互覆盖的中间状态。
_BOOT_LOCK = threading.Lock()


def _load_envs() -> None:
    from dotenv import load_dotenv

    # Agent：LLM key/base_url、天气 key（allagent2/core/key.env）
    load_dotenv(AGENT_DIR / "core" / "key.env")
    # 后端：MySQL 与 JWT（fastap2）
    load_dotenv(FASTAP2_DIR / "database" / "mysql.env")
    load_dotenv(FASTAP2_DIR / "core" / "key.env")


def _register_packages() -> None:
    agent_path = str(AGENT_DIR)
    fastap2_path = str(FASTAP2_DIR)
    # 先清理本引导可能已加入的条目，保证最终顺序确定
    for p in (fastap2_path, agent_path):
        while p in sys.path:
            sys.path.remove(p)
    # 最终顺序：[fastap2_path, agent_path, ...]
    # 两个项目都含顶层 core/database 包，必须让 fastap2 的优先；
    # 而 `session` 仅 allagent2 提供，fastap2 没有，仍会在其后的 agent_path 找到。
    sys.path.insert(0, agent_path)
    sys.path.insert(0, fastap2_path)
    # 把两项目的共同父目录（练手agent）也加入 sys.path，
    # 使 `import allagent2.xxx` 既走显式注册的包，也能按普通包解析。
    parent_dir = FASTAP2_DIR.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    # allagent2 目录以包名 allagent2 注册（其内部导入均为 allagent2.xxx，
    # 与目录名一致，Pylance 等静态检查器也能直接解析）
    if "allagent2" not in sys.modules:
        pkg = types.ModuleType("allagent2")
        pkg.__path__ = [agent_path]  # type: ignore[attr-defined]
        pkg.__package__ = "allagent2"  # type: ignore[attr-defined]
        sys.modules["allagent2"] = pkg


def _inject_mcp_tools() -> int:
    """【MCP 改造】后端工具不再本地注入,改为连接 MCP Server 拉取工具清单。

    必须在 import allagent2.agent.graph 之前完成:make_graph→build_tool_list
    会把当时的 tools 列表 bind 给模型。
    同名本地工具(calculator/get_real_weather)由 MCP 版本替代(去重,避免双份);
    原本地薄壳 backend_tools.py 不再使用(数据面网关已迁入 mcp_servers/data_gateway.py)。
    """
    from agent_bridge.mcp_runtime import connect_servers, get_mcp_tools

    urls: Dict[str, str] = {}
    # knowledge(8003) 不在此列：team.py 的 knowledge 专员直接 import 进程内
    # allagent2.tools.knowledge_tool，即使连上 8003 也不会被调用，只会白等一次超时。
    for name, env in (("elder", "MCP_ELDER_URL"),
                      ("life", "MCP_LIFE_URL")):
        value = os.getenv(env)
        if value:
            urls[name] = value
    if urls:
        connect_servers(urls)

    mcp_tools = get_mcp_tools()
    tool_mod = importlib.import_module("allagent2.tools.tool")
    mcp_names = {t.name for t in mcp_tools}
    # 同名本地工具(calculator/get_real_weather)移除,替换为 MCP 版本;
    # 【关键】MCP 工具必须 append 进 tool_mod.tools —— graph.build_tool_list()
    # 读的是 tools 列表(不只是 tools_name),漏 append 会导致编译时工具没绑定
    tool_mod.tools[:] = [t for t in tool_mod.tools if t.name not in mcp_names]
    tool_mod.tools.extend(mcp_tools)
    tool_mod.tools_name = {t.name: t for t in tool_mod.tools}
    _state.tools_count = len(tool_mod.tools_name)
    if not mcp_tools:
        logger.warning("未连接到任何 MCP Server —— 数据库/生活类工具不可用(fail-closed)")
    return len(mcp_tools)


def _make_checkpointer():
    """【多智能体·地基】构造会话 checkpointer:优先 Redis(持久化),失败降级 InMemory。"""
    _state.persistent = False
    from langgraph.checkpoint.memory import InMemorySaver

    try:
        from urllib.parse import quote

        from core.redis_client import REDIS_DB, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
        pwd = f":{quote(REDIS_PASSWORD)}@" if REDIS_PASSWORD else ""
        redis_url = f"redis://{pwd}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    except Exception as exc:  # Redis 配置缺失
        _note_degraded("checkpointer", exc, "redis 配置不可读")
        logger.warning("Redis 配置不可读,checkpointer 降级 InMemorySaver:%s", exc)
        return InMemorySaver()

    try:
        from langgraph.checkpoint.redis import RedisSaver

        saver = RedisSaver(redis_url)
        saver.setup()  # 创建 redis 索引(key 前缀 checkpoint:*)
        # 【多轮修复】这个版本的 langgraph-checkpoint-redis 把 checkpoint 存成
        # RedisJSON（不走 msgpack ext 钩子）。默认 JSON 复原只放行 langchain 白名单
        # 模块 → 黑板里的自定义 pydantic 模型(Task/PermissionContext/ParsedOutput)
        # 复原成裸 lc 字典 → 第二轮续读 TeamState 校验必炸(500)。
        # 用官方 allowed_json_modules 白名单放行本项目的包,续读时正常还原对象。
        # 注意:白名单不支持前缀匹配,必须精确到"符号路径"——
        # TeamState 里会进 checkpoint 的自定义 pydantic 模型逐一登记。
        # 必须用 Redis 专用子类 JsonPlusRedisSerializer(带 _preprocess_redis_json,
        # 读路径会调用),不能用基类 JsonPlusSerializer。
        from langgraph.checkpoint.redis.base import JsonPlusRedisSerializer

        saver.serde = JsonPlusRedisSerializer(allowed_json_modules=[
            ("allagent2", "agent", "team", "Task"),
            ("allagent2", "agent", "state", "PermissionContext"),
            ("allagent2", "agent", "state", "ParsedOutput"),
        ])
        logger.info("checkpointer=RedisSaver(%s:%s/%s) — 会话状态持久化已启用",
                    REDIS_HOST, REDIS_PORT, REDIS_DB)
        _state.persistent = True
        return saver
    except Exception as exc:  # Redis 连不上
        _note_degraded("checkpointer", exc, "redis 连接失败→内存会话")
        logger.warning("Redis checkpointer 连接失败,降级 InMemorySaver(重启丢会话):%s", exc)
        return InMemorySaver()


def _inject_zero_trust() -> None:
    """【零信任·S3】向 allagent2 注入身份校验器与智能体权限档案（解耦：allagent2 不反向依赖 fastap2）。

    - validator:identity_guard 每次模型调用前回调，复验 Redis 里的可信身份
      （未过期 + perm_version 未变 + 属主匹配）；失效返回 False → 链路内拦截。
    - AGENT_PROFILES:team supervisor 判断"任务可派性"的判据。
    sys.path 顺序保证 `core` 解析到 fastap2/core（_register_packages 已定序）。
    """
    from core.identity import AGENT_PROFILES, get_identity

    from allagent2.agent import middleware as agent_mw

    agent_mw.set_identity_validator(
        lambda iid, uid: get_identity(iid, uid if uid else None) is not None
    )
    logger.info("[zero-trust] identity validator + AGENT_PROFILES 已注入 allagent2")
    return AGENT_PROFILES


def bootstrap() -> _State:
    """执行一次完整接线；重复调用直接返回（幂等，且线程安全）。"""
    if _state.ready:
        return _state
    # 【健壮性】加锁 + 双检：第一个发现未就绪的线程进入临界区完成初始化，
    # 其余线程在锁外/锁内第二次检查到 ready 后直接复用，不再重复接线。
    with _BOOT_LOCK:
        if _state.ready:
            return _state
        try:
            _load_envs()
            _register_packages()
            _inject_mcp_tools()
            # 【改进】业务字段清单以 state.py 的 PARSED_FIELDS 为准（养生养老字段已扩充）
            from allagent2.agent.state import PARSED_FIELDS as _STATE_FIELDS
            global PARSED_FIELDS
            PARSED_FIELDS = list(_STATE_FIELDS)

            # 【改造】make_graph() 直接返回「已编译的 agent」（内部用官方 create_agent）;
            # checkpointer 由 bootstrap 传入(Redis 持久化,失败降级 InMemory)
            checkpointer = _make_checkpointer()
            # 【多智能体】拓扑开关:AGENT_TOPOLOGY=team 切到黑板多智能体
            # (supervisor 并行扇出三专员 → aggregate 汇总);默认 single 单 agent。
            topology = (os.getenv("AGENT_TOPOLOGY") or "single").strip().lower()
            agent_profiles = _inject_zero_trust()
            _inject_abort_check()
            _inject_user_provider()
            if topology == "team":
                from allagent2.agent import team as team_mod
                agent = team_mod.make_team_graph(
                    checkpointer=checkpointer, agent_profiles=agent_profiles
                )
                _state.graph_mod = team_mod
                logger.info("拓扑=team 黑板多智能体已接线(supervisor→3专员→aggregate)")
            else:
                graph_mod = importlib.import_module("allagent2.agent.graph")
                agent = graph_mod.make_graph(checkpointer=checkpointer)
                _state.graph_mod = graph_mod
            _state.topology = topology
            _state.agent = agent
            _state.ready = True
        except BaseException as exc:  # 记录引导失败，便于 /agent/status 诊断
            _state.error = exc
            raise
    return _state



# ============================================================
# 【P1-11】中断标记 + 断点续跑
# ------------------------------------------------------------
# 中断不是杀线程，而是往 Redis 打标记：图在下一个节点入口检查到即抛
# AgentInterrupted，停在上一 superstep，checkpoint 保留，可 resume 续跑。
# ============================================================
# 中断标记 + 断点续跑
# ------------------------------------------------------------
# 中断不是杀线程，而是往 Redis 打标记：图在下一个节点入口检查到即抛
# AgentInterrupted，停在上一 superstep，checkpoint 保留，可 resume 续跑。
# ============================================================
ABORT_KEY_PREFIX = "agent:abort:"
_RESUME_LOCK_PREFIX = "agent:resume-lock:"


def _abort_ttl() -> int:
    """中断标记存活秒数（配置可覆盖，失败回落到 600）。"""
    try:
        from allagent2.config import get_settings

        return int(get_settings().settings.agent.interrupt_ttl)
    except Exception:
        return 600


def _inject_user_provider() -> None:
    """把「当前请求的 user_id」注入知识工具，供个人文档检索用。

    读取器走 contextvars（core.agent_context），所以在线程池/异步生成器里
    都能拿到正确的用户；未注入时知识工具只查静态知识库。
    """
    from core.agent_context import get_access_context

    from allagent2.tools import knowledge_tool

    knowledge_tool.set_user_provider(lambda: get_access_context().user_id)
    logger.info("[knowledge] user provider 已注入（个人文档检索启用）")


def _inject_abort_check() -> None:
    """把「该会话是否已被要求中断」的判定注入 allagent2（读 Redis 标记）。

    未注入时中断能力自然关闭 —— allagent2 不反向依赖 Redis / fastap2，
    与 set_identity_validator 同一套注入式解耦。
    """
    from allagent2.agent import observe as observe_mod

    from core.redis_client import get_redis_sync

    def _is_aborted(thread_id: str, node: str | None = None) -> bool:
        if not thread_id or thread_id == "-":
            return False
        try:
            return bool(get_redis_sync().exists(f"{ABORT_KEY_PREFIX}{thread_id}"))
        except Exception:
            return False  # Redis 不可用 → 不中断（可用性优先）

    observe_mod.set_abort_check(_is_aborted)
    logger.info("[interrupt] abort check 已注入 allagent2（前缀 %s，TTL≈%ss）",
                ABORT_KEY_PREFIX, _abort_ttl())


def set_abort(thread_id: str, ttl: int | None = None) -> bool:
    """给会话打中断标记；返回是否写入成功（Redis 不可用返回 False）。"""
    if not thread_id:
        return False
    try:
        from core.redis_client import get_redis_sync

        return bool(get_redis_sync().setex(
            f"{ABORT_KEY_PREFIX}{thread_id}", ttl or _abort_ttl(), "1"))
    except Exception as exc:
        logger.warning("[interrupt] 写入中断标记失败：%s", exc)
        return False


def clear_abort(thread_id: str) -> bool:
    """清除中断标记（续跑前必须清，否则下一个节点立刻又被停）。"""
    try:
        from core.redis_client import get_redis_sync

        return bool(get_redis_sync().delete(f"{ABORT_KEY_PREFIX}{thread_id}"))
    except Exception:
        return False


def _enter_permissions_from_state(values: Any) -> None:
    """把 checkpoint 里的 PermissionContext 重新放进 contextvars，供 MCP 网关签名。"""
    perm = values.get("permissions") if hasattr(values, "get") else None
    if perm is None:
        return
    try:
        from agent_bridge.middleware import AgentSecurityService

        AgentSecurityService.enter_context({
            "user_role": getattr(perm, "user_role", "common"),
            "user_id": getattr(perm, "user_id", "") or "",
            "allow_sensitive": bool(getattr(perm, "allow_sensitive", False)),
            "allowed_elder_ids": getattr(perm, "allowed_elder_ids", None),
            "allowed_tools": getattr(perm, "allowed_tools", None),
        })
    except Exception as exc:
        logger.warning("[resume] 权限上下文恢复失败（工具可能被网关拒绝）：%s", exc)


def resume_agent(thread_id: str) -> Dict[str, Any]:
    """从最后一个 checkpoint 续跑。

    必须 `invoke(None, config)`：传 state 会追加消息，等于新开一轮。
    并发续跑用 Redis 互斥锁（SET NX EX 180）防两个请求从同一 checkpoint 各跑一遍。
    """
    bootstrap()
    config = {"configurable": {"thread_id": thread_id}}
    locked = False
    try:
        from core.redis_client import get_redis_sync

        if not get_redis_sync().set(f"{_RESUME_LOCK_PREFIX}{thread_id}", "1", nx=True, ex=180):
            raise RuntimeError("该会话已有续跑在执行（互斥锁未释放）")
        locked = True
    except RuntimeError:
        raise
    except Exception as exc:  # Redis 不可用 → 不加锁继续（可用性优先）
        logger.warning("[resume] 续跑互斥锁不可用，继续执行：%s", exc)

    try:
        snap = _state.agent.get_state(config)
        _enter_permissions_from_state(getattr(snap, "values", None) or {})
        clear_abort(thread_id)  # 先清标记，否则恢复后立刻又被停
        result = _state.agent.invoke(None, config=config)
        return _finalize(result, thread_id)
    finally:
        if locked:
            try:
                from core.redis_client import get_redis_sync

                get_redis_sync().delete(f"{_RESUME_LOCK_PREFIX}{thread_id}")
            except Exception:
                pass
        try:
            from agent_bridge.middleware import AgentSecurityService

            AgentSecurityService.exit_context()
        except Exception:
            pass




def get_status() -> Dict[str, Any]:
    return {
        "ready": _state.ready,
        "agent_dir": str(AGENT_DIR),
        "tools_count": _state.tools_count,
        "topology": _state.topology,
        "engine": (os.getenv("AGENT_ENGINE") or "react").strip().lower(),
        "persistent": _state.persistent,
        "error": repr(_state.error) if _state.error else None,
    }


def _fields_from_snapshot(hd: Dict[str, Any]) -> Dict[str, Any]:
    """【多智能体】从 checkpoint 快照提取 PARSED_FIELDS。

    single：字段平铺在 state 上；team：业务字段在黑板 final(ParsedOutput) 里。
    """
    final = hd.get("final")
    if final is not None:
        dump = final.model_dump() if hasattr(final, "model_dump") else dict(final)
        return {f: dump.get(f) for f in (PARSED_FIELDS or []) if dump.get(f) is not None}
    return {f: hd.get(f) for f in (PARSED_FIELDS or []) if hd.get(f) is not None}


def _summarize_message(msg: Any) -> Dict[str, Any]:
    """把一条消息压缩成可在 trace 中展示的结构。"""
    mtype = getattr(msg, "type", msg.__class__.__name__)
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        content = str(content)
    item: Dict[str, Any] = {"role": mtype, "text": content[:150]}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        item["tool_calls"] = [
            {"name": c.get("name"), "args": c.get("args")} for c in tool_calls
        ]
    return item


class _Tracer:
    """【改进】trace 标注节点名：优先取 state.phase（intent_plan / rag_retrieval /
    llm_reasoning / tool_calling / finalizing），缺失时按旧拓扑推断。
    """

    def __init__(self) -> None:
        self.step = 0
        self.phase = "intent_plan"
        self.prev_count = 0

    def observe(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        node = chunk.get("phase") or self.phase
        self.step += 1
        messages = chunk.get("messages", [])
        total = len(messages)
        record = {
            "step": self.step,
            "node": node,
            "messages_total": total,
            "new_messages": [
                _summarize_message(m) for m in messages[self.prev_count:]
            ],
            "fields": {
                f: chunk.get(f) for f in PARSED_FIELDS if chunk.get(f) is not None
            },
        }
        self.prev_count = total
        # 无 phase 时的兜底推断（与 graph 拓扑一致）
        if node == "intent_plan":
            self.phase = "first_llm_call"
        elif node == "first_llm_call":
            last = messages[-1] if messages else None
            self.phase = "tool_use" if getattr(last, "tool_calls", None) else "end"
        elif node == "tool_use":
            self.phase = "first_llm_call"
        return record


def _prepare(
    question: str,
    thread_id: str,
    elder_id: str | None,
    permissions: dict | None,
    user_id: str | None = None,
    identity_id: str | None = None,
):
    """run_agent / run_agent_stream 共用的进图前准备。

    返回 (mode, payload)：
      mode="emergency" → payload 为直接可返回的结果 dict（入口急救短路，零 token）
      mode="graph"     → payload 为 (init_state, ctx, config)
    """
    from langchain.messages import HumanMessage
    from allagent2.agent.state import PermissionContext, config_schema
    from allagent2.prompts.prompts import EMERGENCY_KEYWORDS, EMERGENCY_REPLY

    # 【A5】入口急救短路：规则级判定，不进图、不调 LLM
    hits = [kw for kw in EMERGENCY_KEYWORDS if kw in (question or "")]
    if hits:
        logger.warning("[guard] 入口急救短路命中：%s，直接返回急救指引", hits)
        emergency_struct = {field: None for field in (PARSED_FIELDS or [])}
        emergency_struct["emergency_flag"] = True
        return "emergency", {
            "answer": EMERGENCY_REPLY,
            "structured": emergency_struct,
            "thread_id": thread_id,
        }

    # 【ACL】权限上下文：外部注入 > elder_id 推断 > 默认（fail-closed）
    if permissions is not None:
        perm_ctx = PermissionContext(**permissions)
    elif elder_id is not None:
        perm_ctx = PermissionContext(allowed_elder_ids=[elder_id])
    else:
        # 【安全】fail-closed:调用方未注入任何权限上下文时,显式禁止一切老人数据访问
        perm_ctx = PermissionContext(allowed_elder_ids=[])

    # 【多智能体】team 图的 supervisor 需要 question 原文做任务规划
    init_state: Dict[str, Any] = {
        "messages": [HumanMessage(content=question)],
        "permissions": perm_ctx,
    }
    is_team = _state.topology == "team"
    if is_team:
        init_state["question"] = question
    # 【零信任·S3】身份引用随请求进图：identity_guard 每次模型调用前复验
    # （登出/降权/锁定后，正在跑的长链路在下一个模型调用点即被斩断）
    if identity_id:
        init_state["identity_id"] = identity_id

    # 【记忆系统·链路一】读路径接线：召回长期记忆 → 填 init_state["memory_block"]。
    # 消费端：team 拓扑 supervisor 读 state.memory_block（team.py 拼进规划 prompt）；
    #         single 拓扑 dynamic_prompt 中间件读 state["memory_block"]（middleware.py）。
    # recall 内部全异常吞掉、MySQL/Chroma 不可用时 fail-open 返回空块——
    # 记忆是增益，任何故障都退化为"无记忆"，绝不挡回答。
    if user_id:
        try:
            from allagent2.session.recall import recall

            mem = recall(user_id, question, elder_hint=elder_id or "")
            init_state["memory_block"] = mem.context_block or ""
            if mem.context_block:
                logger.info("[memory] 读路径注入完成 thread=%s stats=%s", thread_id, mem.stats)
        except Exception as exc:
            _note_degraded("memory", exc, "recall")
            logger.warning("[memory] recall 接线失败（按无记忆继续）：%s", exc)

    # 【技能系统】本轮启用的技能 → 渲染成 prompt 块（与 memory_block 同级）。
    # 装配规则在 core.skill_service（显式 > 常开 > 规则命中，上限 3 个）；
    # 技能带工具声明时只与授权取交集，不能扩权。
    if user_id:
        try:
            from core.skill_service import assemble_sync

            picked_skills, skill_trace = assemble_sync(user_id, question)
            if picked_skills:
                from allagent2.skills.registry import render_block

                init_state["skills_block"] = render_block(picked_skills)
                logger.info("[skill] 技能已注入 thread=%s skills=%s", thread_id,
                            [s.id for s in picked_skills])
        except Exception as exc:
            _note_degraded("skill", exc, "assemble")
            logger.warning("[skill] 技能装配失败（按无技能继续）：%s", exc)

    # 运行时上下文走官方 context_schema（create_agent 的 context 参数）
    ctx = config_schema(
        thread_id=thread_id,
        current_elder_id=elder_id,
        permissions=perm_ctx,
        locale="zh-CN",
    )
    config = {"configurable": {"thread_id": thread_id}}
    return "graph", (init_state, ctx, config)


def _with_usage_callback(config: Dict[str, Any]):
    """【token修复】给请求 config 挂上本轮用量回调，返回 (round_acc, new_config)。

    为什么需要：team 拓扑下专员的子图消息不出黑板（messages 通道只有 aggregate
    手工构造的裸 AIMessage），supervisor/aggregate 的 structured 调用更不产生消息
    → 事后扫 messages 的 collect_usage 在 team 模式恒为 0。
    改为回调计量：LangGraph 会把 config.callbacks 传播进所有节点（含嵌套子
    agent），每个 LLM 调用结束(on_llm_end)即累加——single/team、invoke/stream 通吃。
    """
    try:
        from allagent2.guance.token_accounting import (
            TokenAccountant,
            TokenAccountingCallback,
        )

        acc = TokenAccountant()
        cb = TokenAccountingCallback(acc)
        cfg = dict(config or {})
        callbacks = [cb]

        # team 拓扑的专员是嵌套 create_agent，外层黑板只看到"进场/出场"，
        # 看不到它内部调了哪个工具、模型花了多久 —— 用 TraceCallback 单点补齐。
        if _state.topology == "team":
            try:
                from allagent2.agent.observe import TraceCallback

                tid = (cfg.get("configurable") or {}).get("thread_id")
                if tid:
                    callbacks.append(TraceCallback(str(tid)))
            except Exception as exc:
                logger.warning("[trace] TraceCallback 挂载失败（链路事件会缺 LLM/工具明细）：%s", exc)

        cfg["callbacks"] = callbacks
        return acc, cfg
    except Exception as exc:
        logger.warning("[token] 回调计量不可用，回退消息扫描：%s", exc)
        return None, config


def _finalize(result: Dict[str, Any], thread_id: str, debug: bool = False,
              round_acc: Any = None, user_id: str | None = None,
              question: str = "") -> Dict[str, Any]:
    """run_agent / run_agent_stream 共用的出图后处理：结构化提取、兜底、账单。

    输入是图执行完的最终 state（invoke 返回值或 astream 累积的合并 state），
    输出是对外响应 dict（answer/structured/thread_id/usage）——链路过程只进日志。
    """
    result = result if isinstance(result, dict) else {}
    messages = result.get("messages", []) or []

    # 【A3】优先取官方结构化输出：
    #   single → structured_response（response_format=ParsedOutput）
    #   team   → final（aggregate 节点写入黑板的 ParsedOutput）
    structured: Dict[str, Any] = {}
    answer = None
    sr = result.get("final") or result.get("structured_response")
    if sr is not None:
        dump = sr.model_dump() if hasattr(sr, "model_dump") else dict(sr)
        structured = {field: dump.get(field) for field in (PARSED_FIELDS or [])}
        answer = dump.get("answer")

    # 兜底：结构化未产出时，只回退"最后一条 AI 消息"（绝不能取 system 消息冒充答案）
    if not answer:
        for m in reversed(messages):
            if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
                answer = m.content
                break
    if not answer:
        answer = "（智能体已完成本轮解析，但未生成可展示的最终回复，请换个问法重试）"
    if not structured:
        structured = {field: None for field in (PARSED_FIELDS or [])}

    # 急救标记（链路内 emergency_triage 中间件短路时会置位）
    if result.get("emergency_flag"):
        structured["emergency_flag"] = True

    logger.info(
        "[chain] 完整链路结果: emergency=%s 消息总数=%d answer长度=%d",
        result.get("emergency_flag"),
        len(messages),
        len(answer or ""),
    )
    if debug:
        # 【安全】debug 不返回 trace，只把逐步执行轨迹打到日志。
        # 【健壮性修复】整段 try/except：team 拓扑的 TeamState 是严格 pydantic 模型，
        # get_state_history 回放旧 checkpoint 时会对 Redis 里未完全反序列化的
        # lc 载荷做强制校验（曾必现 ValidationError → 整个 /chat 500）。
        # 链路诊断是可观测能力，失败绝不能拖垮用户的正常回答。
        try:
            step = 0
            prev_count = 0
            for h in reversed(list(_state.agent.get_state_history({"configurable": {"thread_id": thread_id}}))):
                hv = h.values
                hd = hv.model_dump() if hasattr(hv, "model_dump") else dict(hv)
                step += 1
                msgs = hd.get("messages", [])
                total = len(msgs)
                logger.info(
                    "[chain] step%d messages_total=%d new_messages=%s fields=%s",
                    step,
                    total,
                    [_summarize_message(m) for m in msgs[prev_count:]],
                    _fields_from_snapshot(hd),
                )
                prev_count = total
        except Exception as exc:
            logger.warning("[chain] debug 轨迹回放失败（不影响回答）：%s", exc)

    # 【token可观测】本轮新增模型调用的用量汇总 → 全局累计 + JSONL 账本。
    # 回调计量优先（见 _with_usage_callback 注释）；无回调时回退扫描 messages
    # （collect_usage 按 AIMessage.id 去重，invoke 返回的 messages 含历史全部消息）。
    try:
        from allagent2.guance.token_accounting import (
            append_usage_log,
            get_global_accountant,
        )

        if round_acc is None:
            from allagent2.guance.token_accounting import collect_usage

            round_acc = collect_usage(messages)
        get_global_accountant().merge(round_acc)
        append_usage_log(round_acc, thread_id=thread_id, tag="chat")
        usage = round_acc.to_dict()
        logger.info(
            "[token] 本轮汇总：calls=%d input=%d(hit %.0f%%) output=%d cost=%.6f元",
            usage["calls"],
            usage["input_tokens"],
            usage["cache_hit_rate"] * 100,
            usage["output_tokens"],
            usage["cost_yuan"],
        )
    except Exception as exc:
        usage = {"error": str(exc)}
        logger.warning("[token] 用量汇总失败（不影响回答）：%s", exc)

    # 【记忆系统·链路二】写路径接线：登记轮次 → 达门槛则后台线程池异步反思沉淀
    # （情景/语义/技能/画像四层）。on_turn_finished 是 fire-and-forget：
    # 登记是快 SQL（同步），反思是慢 LLM 调用（submit 到专用线程池），
    # 内部全异常自吞——沉淀失败绝不影响本响应。
    if user_id:
        try:
            from allagent2.session.reflect import on_turn_finished

            on_turn_finished(user_id, thread_id, messages,
                             question=question, answer=answer or "")
        except Exception as exc:
            logger.warning("[memory] 写路径接线失败（跳过本轮沉淀）：%s", exc)

    # 【安全】对外只返回最终回答 + 结构化字段；链路过程一律不进响应
    return {
        "answer": answer,
        "structured": structured,
        "thread_id": thread_id,
        "usage": usage,
    }


def run_agent(
    question: str,
    thread_id: str = "default",
    debug: bool = False,
    elder_id: str | None = None,
    permissions: dict | None = None,
    user_id: str | None = None,
    identity_id: str | None = None,
) -> Dict[str, Any]:
    """同步执行一次完整 Agent 流程（create_agent / team 通用）。

    结构：_prepare（急救短路/权限/记忆召回/初始 state）→ invoke → _finalize（结构化/兜底/账单/沉淀）。
    流式版本见 run_agent_stream：同一段准备与后处理，进度事件经 SSE 推给前端。
    identity_id：可信身份引用，随请求进图供 identity_guard 复验（零信任·S3）。
    """
    bootstrap()
    mode, payload = _prepare(question, thread_id, elder_id, permissions,
                             user_id=user_id, identity_id=identity_id)
    if mode == "emergency":
        return payload
    init_state, ctx, config = payload
    # 【token】请求级回调计量（team 模式消息扫描拿不到 usage，见 _with_usage_callback）
    round_acc, config = _with_usage_callback(config)

    if _state.topology == "team":
        result = _state.agent.invoke(init_state, config=config)
    else:
        result = _state.agent.invoke(init_state, config=config, context=ctx)
    return _finalize(result, thread_id, debug, round_acc=round_acc,
                     user_id=user_id, question=question)


# ============================================================
# 思考过程可见性
# ------------------------------------------------------------
# AGENT_TRACE_VISIBLE: all（默认，登录用户都能看自己的）| admin | off
# AGENT_TRACE_MAX_CHARS: 单个字段最多展示多少字符（防一次工具结果撑爆前端）
# ============================================================
TRACE_MAX_CHARS = int(os.getenv("AGENT_TRACE_MAX_CHARS", "600"))


def trace_visible_for(user_role: str) -> bool:
    """当前角色能否看到思考过程。"""
    mode = (os.getenv("AGENT_TRACE_VISIBLE") or "all").strip().lower()
    if mode == "off":
        return False
    if mode == "admin":
        return str(user_role) == "admin"
    return True


def _cap(value, limit: int = 0):
    """把长文本截断，超长加省略号。"""
    if not isinstance(value, str):
        return value
    limit = limit or TRACE_MAX_CHARS
    return value if len(value) <= limit else value[:limit] + "…"


def _cap_deep(value, depth: int = 0, limit: int = 0):
    """递归截断 dict/list 里的字符串，避免嵌套结构把整条事件撑大。"""
    if depth > 3:
        return "…"
    if isinstance(value, str):
        return _cap(value, limit)
    if isinstance(value, dict):
        return {k: _cap_deep(v, depth + 1, limit) for k, v in list(value.items())[:30]}
    if isinstance(value, (list, tuple)):
        return [_cap_deep(v, depth + 1, limit) for v in list(value)[:30]]
    return value


def _mask(text):
    """工具结果/报告里的手机号、身份证打码（复用图内同一份规则）。"""
    if not isinstance(text, str) or not text:
        return text
    try:
        from allagent2.agent.middleware import mask_pii

        return mask_pii(text)
    except Exception:
        return text


def sanitize_event(event: Dict[str, Any], raw: bool = False) -> Dict[str, Any]:
    """把一条原始事件变成可安全展示给用户的版本。

    raw=True 只给管理员用：保留完整内容，但单字段仍受 AGENT_TRACE_MAX_CHARS*4 限制。
    """
    if not isinstance(event, dict):
        return {"type": "event", "value": str(event)[:200]}
    limit = TRACE_MAX_CHARS * (4 if raw else 1)
    out = dict(event)
    for key in ("input", "content", "text", "thinking", "summary", "error"):
        if key in out:
            out[key] = _cap(_mask(out[key]) if key in ("content", "input") else out[key], limit)
    if "result" in out:
        out["result"] = _cap_deep(out["result"], 0, limit)
    if "tool_calls" in out and isinstance(out["tool_calls"], list):
        out["tool_calls"] = _cap_deep(out["tool_calls"], 0, limit)
    if "args" in out:
        out["args"] = _cap_deep(out["args"], 0, limit)
    out.pop("thread_id", None)
    return out


# SSE 进度事件：节点名 → 面向用户的阶段文案（不暴露内部链路细节）
_PHASE_LABELS = {
    # team 拓扑
    "supervisor": "正在分析问题、安排任务…",
    "health": "健康专员正在查询老人数据…",
    "knowledge": "知识专员正在检索养生知识…",
    "life": "生活助手正在处理…",
    "aggregate": "正在汇总各方结论…",
    # single 拓扑（create_agent 的节点名）
    "model": "思考中…",
    "tools": "正在调用工具…",
}


async def run_agent_stream(
    question: str,
    thread_id: str = "default",
    debug: bool = False,
    elder_id: str | None = None,
    permissions: dict | None = None,
    user_id: str | None = None,
    identity_id: str | None = None,
):
    """流式执行 Agent（SSE 用）：异步产出 (event, data) 事件序列。

    事件序列：
      ("phase", {"node": "supervisor", "label": "正在分析问题…"})  # 每个节点开始时
      ("final", {answer/structured/thread_id/usage})               # 结束（含急救短路）
    链路过程只以 phase 节点名形式出现，工具参数/检索内容等细节仍只进日志。

    【修复】原来直接 astream()：langgraph-checkpoint-redis 只实现了同步接口
    （aget_tuple 未实现）→ 一跑就 NotImplementedError。
    现改为「同步 graph.stream() 丢进线程池 + asyncio.Queue 桥接」：
      - producer 线程：同步流式产出 chunk，经 call_soon_threadsafe 塞进队列；
      - 消费侧（事件循环）：await queue.get() 取事件，逐个 yield 给 SSE。
    权限上下文用 contextvars.copy_context() 显式带进 producer 线程
    （loop.run_in_executor 不传播 contextvars，MCP 网关取不到令牌会全线 403）。
    """
    bootstrap()
    mode, payload = _prepare(question, thread_id, elder_id, permissions,
                             user_id=user_id, identity_id=identity_id)
    if mode == "emergency":
        yield "final", payload
        return
    init_state, ctx, config = payload
    is_team = _state.topology == "team"
    stream_kwargs = {} if is_team else {"context": ctx}
    # 【token】请求级回调计量（team 模式消息扫描拿不到 usage，见 _with_usage_callback）
    round_acc, config = _with_usage_callback(config)

    import asyncio
    import contextvars

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    from allagent2.agent import observe as _observe

    def _on_event(ev: Dict[str, Any]) -> None:
        """订阅回调跑在 producer 线程里，必须经 call_soon_threadsafe 回主循环。"""
        try:
            loop.call_soon_threadsafe(queue.put_nowait, ("event", ev))
        except Exception:
            pass

    def _produce():
        """在线程池里跑同步流（RedisSaver 的同步接口）；结果回填队列。"""
        _observe.register_event_sink(thread_id, _on_event)
        try:
            for chunk in _state.agent.stream(
                init_state, config=config, stream_mode="updates", **stream_kwargs
            ):
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
            # 最终 state 从 checkpoint 取：updates 流只含各节点增量
            # （messages 是增量列表，手动 merge 会互相覆盖），get_state 拿全量。
            snap = _state.agent.get_state(config)
            values = snap.values if snap is not None else {}
            final_state = values.model_dump() if hasattr(values, "model_dump") else dict(values)
            loop.call_soon_threadsafe(queue.put_nowait, ("done", (final_state, round_acc)))
        except BaseException as exc:  # producer 在线程里，异常只能经队列带回
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            _observe.unregister_event_sink(thread_id)

    # copy_context：捕获当前请求的权限 contextvars（含 MCP 令牌所需上下文），
    # producer 线程内 ctx.run(...) 使其在整个同步执行期间可见。
    request_ctx = contextvars.copy_context()
    loop.run_in_executor(None, request_ctx.run, _produce)

    while True:
        kind, item = await queue.get()
        if kind == "chunk":
            for node in (item or {}):
                if node.startswith("__"):
                    continue  # __start__/__end__ 等框架节点不推给前端
                yield "phase", {"node": node, "label": _PHASE_LABELS.get(node, "处理中…")}
        elif kind == "event":
            yield "event", sanitize_event(item)
        elif kind == "done":
            final_state, final_round_acc = item
            break
        else:  # error
            exc = item
            logger.error("[stream] 图执行失败：%s", exc, exc_info=debug)
            yield "final", {
                "answer": "智能体处理本轮请求时出现异常，请稍后重试；如涉及紧急健康状况请直接拨打 120。",
                "structured": {field: None for field in (PARSED_FIELDS or [])},
                "thread_id": thread_id,
            }
            return

    yield "final", _finalize(final_state, thread_id, debug, round_acc=final_round_acc,
                             user_id=user_id, question=question)


def _node_name(snapshot, index: int) -> str:
    """从 checkpoint 快照推断节点名。

    【改造】原来靠 state 里手写的 phase 字符串标注阶段；create_agent 版节点
    由框架管理（model / tools），改用快照自带的 next / tasks 标注更准确。
    """
    nxt = getattr(snapshot, "next", None) or ()
    if nxt:
        return ",".join(nxt)
    tasks = getattr(snapshot, "tasks", None) or ()
    if tasks:
        return getattr(tasks[0], "name", "unknown") or "unknown"
    return "start" if index == 0 else "model"


def get_trace(thread_id: str, raw: bool = False) -> Dict[str, Any]:
    """【管理员监控】导出某会话的完整思考链路(仅供 /agent/trace 管理端点调用)。

    从 LangGraph checkpoint 历史逐步回放:
    每一步包含 node(节点名)、新增消息(role+文本摘要)、急救标记、结构化字段变化。
    普通用户接口(run_agent)绝不返回这些内容 —— 链路过程只进日志和本管理端点。
    """
    bootstrap()
    config = {"configurable": {"thread_id": thread_id}}
    steps: list = []
    prev_count = 0
    # get_state_history 由新到旧,reverse 成执行顺序;每步只取"新增"消息避免重复
    # 【改造】原来用 state 里的 phase 字段标注阶段；create_agent 版不再维护
    # 手写 phase（节点由框架管理），改用 checkpoint 自带的节点名标注每一步。
    # 【健壮性】回放失败（如 team 的 TeamState 对旧 checkpoint 校验失败）返回空
    # 轨迹而非 500，错误原因写日志。
    # 【team 拓扑】事件流走旁路通道（TeamState 没有 trace 字段），
    # 这里把旁路事件一并带回；否则 /agent/trace 在 team 下只有消息摘要，
    # 看不到 node_start / llm_end / tool_end。
    team_events: list = []
    if _state.topology == "team":
        try:
            from allagent2.agent.observe import read_team_trace

            team_events = read_team_trace(thread_id)
        except Exception as exc:
            logger.warning("[trace] team 旁路事件读取失败：%s", exc)

    try:
        history = list(_state.agent.get_state_history(config))
    except Exception as exc:
        logger.warning("[trace] 链路回放失败（返回空轨迹）：%s", exc)
        events = team_events if raw else [sanitize_event(e) for e in team_events]
        return {"thread_id": thread_id, "steps": [], "events": events,
                "error": str(exc)}
    for i, h in enumerate(reversed(history)):
        hv = h.values
        hd = hv.model_dump() if hasattr(hv, "model_dump") else dict(hv)
        msgs = hd.get("messages", [])
        step = {
            "step": i + 1,
            "node": _node_name(h, i),
            "emergency_flag": hd.get("emergency_flag"),
            "messages_total": len(msgs),
            "new_messages": [_summarize_message(m) for m in msgs[prev_count:]],
            "fields": _fields_from_snapshot(hd),
        }
        prev_count = len(msgs)
        steps.append(step)
    events = team_events if raw else [sanitize_event(e) for e in team_events]
    return {"thread_id": thread_id, "steps": steps, "events": events}
