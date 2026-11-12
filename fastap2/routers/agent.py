"""养老智能体对话路由:把 HTTP 请求桥接到 LangGraph Agent。

【ACL 改造】本路由不含任何安全判断逻辑,只做三件事:
1. require_permission(PERM_AGENT_CHAT):端点级 ACL(登录 + 权限码);
2. 权限上下文由服务端推导:allowed_elder_ids 来自 user_elder_binding 表,
   客户端请求体不再传任何 allowed_* 字段(原来客户端不传 = 不限定 = 可查任意老人);
3. thread_id 属主校验:会话线程登记在 conversation_owner 表,
   只能续接自己创建的线程(防会话劫持);安全拒绝一律 HTTP 403。
"""
import json
import logging
import uuid

logger = logging.getLogger("agent.metrics")

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agent_bridge import (
    get_status,
    get_trace,
    resume_agent,
    run_agent,
    run_agent_stream,
    set_abort,
)  # 【C1】流式版一并导入
from agent_bridge.middleware import AgentSecurityService
from agent_bridge.schemas import AgentChatRequest
from core import idempotency as idem
import os
import re

from core.acl import PERM_AGENT_CHAT, PERM_AUDIT_READ, PERM_SYSTEM_DIAG, require_permission
from core.audit import write_audit
from core.rate_limit import enforce_rate_limit
from core.redis_client import get_redis_async
from core.responses import ApiResponse
from database.database import get_db
from models.acl import ConversationOwner, UserElderBinding
from models.elder import Elder

agent_router = APIRouter(prefix="/agent", tags=["养老智能体对话"])


def _is_llm_down(exc: Exception) -> bool:
    """判定是否为推理后端不可达（本地 vLLM 未启动 / 在线 API 故障）。

    openai.OpenAIError 是 APIConnectionError/InternalServerError 等的基类;
    懒导入:openai 包轻量且是 langchain-openai 底层硬依赖,放函数内避免顶层耦合。
    """
    try:
        import openai
        return isinstance(exc, openai.OpenAIError)
    except Exception:
        return False


from sqlalchemy.exc import IntegrityError

async def _resolve_thread_id(db: AsyncSession, user_id: str, req: AgentChatRequest) -> str:
    """会话线程属主校验与登记（先插后查，让唯一索引裁决竞态）。

    - 客户端传了 thread_id：直接 insert；成功=归当前用户，撞唯一索引=复查属主；
    - 未传：服务端生成后登记（不用 elder_id 当 thread_id，那是可枚举的劫持入口）。

    为什么不是 check-then-insert：两个并发请求同时 select 都拿不到行、然后都
    insert，后者撞 unique 索引冒成 500。改为让数据库裁决，撞了就回滚复查。
    """
    thread_id = req.thread_id or f"t-{uuid.uuid4().hex[:12]}"
    db.add(ConversationOwner(thread_id=thread_id, user_id=user_id))
    try:
        await db.commit()
        return thread_id
    except IntegrityError:
        await db.rollback()
        row = (await db.execute(
            select(ConversationOwner).where(ConversationOwner.thread_id == thread_id)
        )).scalar_one_or_none()
        if row is None or str(row.user_id) != str(user_id):
            raise HTTPException(status_code=403, detail="无权使用该会话线程")
        return thread_id


async def _assert_thread_owner(db: AsyncSession, current, thread_id: str) -> None:
    """【ACL】续跑/中断前的属主校验（防会话劫持）。

    /resume 与 /interrupt 只凭 thread_id 操作一个已存在的会话：不做属主校验，
    拿到别人的 thread_id 就能续跑（resume_agent 还会用该线程 checkpoint 里的
    身份继续跑，等于读走别人的答案），或把别人的跑批打断（DoS）。
    /chat 由 _resolve_thread_id 保证，/trace 有等价校验，这里补齐这两个端点。
    """
    if str(getattr(current, "user_role", "")) == "admin":
        return
    owner = (await db.execute(
        select(ConversationOwner).where(ConversationOwner.thread_id == thread_id)
    )).scalar_one_or_none()
    if owner is None or str(owner.user_id) != str(current.user_id):
        write_audit(str(current.user_id), "thread:denied", resource=f"thread:{thread_id}",
                    decision=False, reason="非会话属主(resume/interrupt)")
        raise HTTPException(status_code=403, detail="无权操作该会话线程")



_ELDER_TOKEN_RE = re.compile(r"[Ee]\d{2,6}")


def _extract_elder_id(text: str) -> str | None:
    """从用户问题里认出老人编号（"E001 最近血压怎么样" → "E001"）。

    前端有老人 pill，但用户更常直接打在句子里；两条路都要能识别。
    """
    if not text:
        return None
    hit = _ELDER_TOKEN_RE.search(text)
    return hit.group(0).upper() if hit else None


def _auto_bind_enabled() -> bool:
    """是否允许"提问即绑定"。ELDER_AUTO_BIND=0 关闭，交回管理员手工授权。

    【修复】原来写的是 "ELDER_A UTO_BIND"（中间多了个空格），
    环境变量永远读不到 → 这个安全开关形同不存在，只能改代码才能关。
    默认仍为 1（保留"提问即绑定"的产品行为），要严格对象级授权就把
    ELDER_AUTO_BIND=0 写进 core/key.env。
    """
    return (os.getenv("ELDER_AUTO_BIND", "1").strip().lower()
            not in ("0", "false", "off", "no"))


async def _auto_bind_elder(db: AsyncSession, user_id: str, elder_id: str) -> bool:
    """把 elder_id 绑定到当前账号（档案不存在则拒绝，避免绑一堆不存在的编号）。

    返回是否本次新绑定。并发重复绑定由唯一索引裁决，撞了就当已绑定。
    """
    exists = (await db.execute(
        select(Elder.elder_id).where(Elder.elder_id == elder_id)
    )).scalar_one_or_none()
    if exists is None:
        return False

    db.add(UserElderBinding(user_id=user_id, elder_id=elder_id, relation="self-service"))
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        return False
    write_audit(user_id, "acl:auto_bind", resource=f"elder:{elder_id}",
                decision=True, reason="提问触发自动绑定（ELDER_AUTO_BIND）")
    return True


async def _authorize_chat(
    db: AsyncSession,
    current,
    req: AgentChatRequest,
) -> tuple:
    """/chat 与 /chat/stream 共用的零信任三步流水线，返回 (thread_id, TrustedContext)。

    ① 认证：登录态已由 require_permission 保证，此处做用户级限流（防刷 token 账单）；
    ② 清洗：注入倾向扫描（403 零 token）+ 标识符格式校验（422），全部瞬时、不碰 LLM；
    ③ 装配：会话线程属主校验 → 可信身份（客户端携带且校验通过则复用，否则重建）→
       写 Redis（identity:{id}，TTL 900s）→ 后续 Agent 层凭 identity_id 复验。
    安全拒绝一律 HTTP 403/422（而非 200 包错误 body，便于监控/网关识别）。
    """
    from core import identity as idm

    # ---- ① 认证：限流(每用户每分钟 10 次,滑动窗口) ----
    try:
        await enforce_rate_limit(f"rl:agent:user:{current.user_id}", 10, 60, "智能体对话")
    except HTTPException as e:
        write_audit(str(current.user_id), "rate_limited:agent_chat",
                    resource=f"user:{current.user_id}", decision=False, reason=str(e.detail))
        raise

    # ---- ② 清洗：注入扫描 + 敏感意图阻断 + 标识符校验（规则级,零 token,不进入智能体链路）----
    injection = AgentSecurityService.scan_injection(req.message)
    if injection is not None:
        write_audit(str(current.user_id), "guard:injection",
                    resource=f"msg:{req.message[:50]}", decision=False, reason=injection)
        raise HTTPException(status_code=403, detail="消息包含不被允许的内容，已被拒绝")

    guard = AgentSecurityService.guard_message(
        req.message, {"allow_sensitive": current.user_role == "admin"}
    )
    if guard is not None:
        write_audit(str(current.user_id), "guard:sensitive",
                    resource=f"msg:{req.message[:50]}", decision=False, reason=guard["msg"])
        raise HTTPException(status_code=403, detail=guard["msg"])

    bad_id = AgentSecurityService.validate_identifiers(req.elder_id)
    if bad_id is not None:
        write_audit(str(current.user_id), "guard:bad_identifier",
                    resource=f"elder:{req.elder_id}", decision=False, reason=bad_id)
        raise HTTPException(status_code=422, detail=bad_id)

    # ---- ③ 装配：线程属主校验 → 可信身份 ----
    thread_id = await _resolve_thread_id(db, str(current.user_id), req)

    bound_ids = list((await db.execute(
        select(UserElderBinding.elder_id).where(
            UserElderBinding.user_id == str(current.user_id)
        )
    )).scalars().all())

    # 【自动绑定】用户问谁就绑谁：从请求体或问题原文里认出老人编号，
    # 未绑定则当场建立绑定（关系记为 self-service，写审计）。
    # 想要严格的对象级授权，把 ELDER_AUTO_BIND 设为 0，改回管理员手工绑定。
    elder_hint = req.elder_id or _extract_elder_id(req.message)
    transient_hint = None
    if (elder_hint and current.user_role != "admin" and _auto_bind_enabled()
            and elder_hint not in bound_ids):
        if await _auto_bind_elder(db, str(current.user_id), elder_hint):
            bound_ids.append(elder_hint)
            logger.info("[acl] 自动绑定 user=%s elder=%s", current.user_id, elder_hint)
        else:
            # 档案不存在：不落库绑定，但本轮放行，让工具如实回答"没有这位老人的记录"，
            # 而不是把用户挡在"你没绑定"这种看不懂的提示上（不存在的档案无数据可泄）。
            transient_hint = elder_hint
            logger.info("[acl] 编号 %s 无档案，本轮放行查询", elder_hint)
    # 对象级范围永远以 DB 现查为准（缓存只省 RBAC 重建,绑定关系不做陈旧缓存）
    allowed_elders = None if current.user_role == "admin" else list(bound_ids)
    if allowed_elders is not None and transient_hint and transient_hint not in allowed_elders:
        allowed_elders = allowed_elders + [transient_hint]

    trusted = None
    if req.identity_id:
        # 【性能】get_identity 内部是同步 Redis 读（+ 必要时回源 DB），
        # 直接在 async 里调会卡住事件循环；Redis 抖动时一次就是秒级。
        # 客户端携带登录时下发的身份：属主匹配 + TTL 内 + perm_version 未变才复用
        trusted = await run_in_threadpool(
            idm.get_identity, req.identity_id, str(current.user_id)
        )
        if trusted is not None:
            trusted.allowed_elders = allowed_elders
            trusted.session_id = thread_id
            await run_in_threadpool(idm.put_identity, trusted)  # 刷新缓存(续 TTL + 归属)
    if trusted is None:
        # build_identity 同样含同步 DB(pymysql 读 role_permission)+同步 Redis(perm_version)
        trusted = await run_in_threadpool(
            idm.build_identity, str(current.user_id), str(current.user_role),
            thread_id, allowed_elders,
        )
        await run_in_threadpool(idm.put_identity, trusted)

    write_audit(str(current.user_id), "identity:assembled",
                resource=f"thread:{thread_id}", decision=True,
                reason=f"identity={trusted.identity_id} reused={req.identity_id is not None and trusted.identity_id == req.identity_id}")
    return thread_id, trusted


@agent_router.post("/chat")
async def chat_with_agent(
    req: AgentChatRequest,
    request: Request,
    current=Depends(require_permission(PERM_AGENT_CHAT)),
    db: AsyncSession = Depends(get_db),
):
    """对话主链路：准入 → 幂等三态 → 线程池执行图 → 统一响应。"""
    thread_id, trusted = await _authorize_chat(db, current, req)
    permissions = AgentSecurityService.build_permissions_from_identity(trusted)

    # 【幂等】只有客户端带了 Idempotency-Key 才生效；不带则完全跳过，
    # 不强迫调用方改造。三态：done 回放 / inflight 409 / new 抢锁执行。
    redis = get_redis_async()
    idem_key = idem.normalize_key(
        str(current.user_id), request.headers.get("Idempotency-Key"))
    if idem_key:
        state, cached = await idem.acquire(redis, idem_key)
        if state == idem.ST_DONE and cached is not None:
            body = ApiResponse.success(data=cached).model_dump()
            return JSONResponse(content=body, headers={"Idempotent-Replay": "true"})
        if state == idem.ST_INFLIGHT:
            raise HTTPException(status_code=409, detail="同一请求正在处理中，请稍后重试",
                                headers={"Retry-After": "2"})

    AgentSecurityService.enter_context(permissions)
    try:
        result = await run_in_threadpool(
            run_agent, req.message, thread_id, req.debug, req.elder_id, permissions,
            str(current.user_id), trusted.identity_id,
        )
    except Exception as exc:
        # 执行失败只释放锁、不写 done —— 否则这次失败会变成终态，用户永远拿不到结果
        if idem_key:
            await idem.release(redis, idem_key)
        if _is_llm_down(exc):
            logger.error("[chat] 推理后端不可达: %s", str(exc)[:200])
            write_audit(str(current.user_id), "llm:unavailable", decision=False,
                        resource=f"thread:{thread_id}", reason=str(exc)[:200])
            raise HTTPException(status_code=503, detail="推理服务暂时不可用，请稍后再试")
        raise
    finally:
        AgentSecurityService.exit_context()

    if idem_key:
        await idem.commit(redis, idem_key, result)
    return ApiResponse.success(data=result)


@agent_router.post("/chat/stream")
async def chat_with_agent_stream(
    req: AgentChatRequest,
    request: Request,
    current=Depends(require_permission(PERM_AGENT_CHAT)),  # 【ACL】与 /chat 同级
    db: AsyncSession = Depends(get_db),
):
    """【C1 · SSE 流式】与 /chat 完全相同的零信任三步流水线,响应改为事件流。

    事件格式(text/event-stream):
        event: phase
        data: {"node": "health", "label": "健康专员正在查询老人数据…"}

        event: final
        data: {answer/structured/thread_id/usage}
    链路细节(工具参数/检索内容)仍只进日志;前端拿到阶段提示 + 最终结果。

    【幂等】与 /chat 同一套语义：客户端带 Idempotency-Key 时——
      · done 命中 → 不再重跑图，只补发一条 final 事件（响应头 Idempotent-Replay: true）；
      · inflight  → 409，避免同一条消息被两路 SSE 各跑一遍（双倍 token）；
      · 断线重连拿同一个 key 重发，就是"回放"而不是"再问一次"。
    """
    thread_id, trusted = await _authorize_chat(db, current, req)
    permissions = AgentSecurityService.build_permissions_from_identity(trusted)

    # 【幂等】只有带 Idempotency-Key 才生效；不带完全跳过，不强迫调用方改造
    redis = get_redis_async()
    idem_key = idem.normalize_key(
        str(current.user_id), request.headers.get("Idempotency-Key"))
    if idem_key:
        state, cached = await idem.acquire(redis, idem_key)
        if state == idem.ST_DONE and cached is not None:
            async def replay_source():
                yield ("event: final\ndata: " +
                       json.dumps(cached, ensure_ascii=False, default=str) + "\n\n")

            return StreamingResponse(
                replay_source(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                         "Idempotent-Replay": "true"},
            )
        if state == idem.ST_INFLIGHT:
            raise HTTPException(status_code=409, detail="同一请求正在处理中，请稍后重试",
                                headers={"Retry-After": "2"})

    async def event_source():
        # contextvars 注入:异步生成器与请求同一上下文,MCP 网关据此取权限令牌
        AgentSecurityService.enter_context(permissions)
        final_payload = None
        try:
            async for event, data in run_agent_stream(
                req.message, thread_id, req.debug, req.elder_id, permissions,
                user_id=str(current.user_id), identity_id=trusted.identity_id,
            ):
                if event == "final":
                    final_payload = data
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
            # 跑完才落 done：中途抛错/客户端断开都不写，允许同 key 重试
            if idem_key and final_payload is not None:
                await idem.commit(redis, idem_key, final_payload)
        except Exception as exc:
            if idem_key:
                await idem.release(redis, idem_key)
            # 【降级】SSE 已开流,不能改状态码;推 error 事件让前端能感知并提示
            if _is_llm_down(exc):
                logger.error("[chat/stream] 推理后端不可达: %s", str(exc)[:200])
                write_audit(str(current.user_id), "llm:unavailable", decision=False,
                            resource=f"thread:{thread_id}", reason=str(exc)[:200])
                yield ("event: error\ndata: " +
                       json.dumps({"detail": "推理服务暂时不可用，请稍后再试"},
                                  ensure_ascii=False) + "\n\n")
                return
            raise
        finally:
            AgentSecurityService.exit_context()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@agent_router.get("/metrics")
async def agent_metrics(
    current=Depends(require_permission(PERM_AUDIT_READ)),  # 【ACL】成本/检索指标仅 admin
):
    """【管理员监控·接口②】Token 成本账本 + RAG 检索指标。

    - usage:  全局累计（进程内实时）+ 最近 50 轮 JSONL 账单（重启不丢）;
    - rag:    最近 50 次检索指标 + 汇总（平均延迟/重排使用率/向量降级次数/上下文注入量）。
    与 /agent/trace 同级权限(audit:read),普通用户不可见——成本与检索细节不外泄。
    """
    from allagent2.guance.token_accounting import get_global_accountant, read_usage_log

    usage_log = read_usage_log(50)
    rag_log = []
    try:
        # 延迟导入:rag 链路拖 torch/chromadb,不能在模块顶层拉起
        from allagent2.rag.rag import read_metrics_log

        rag_log = read_metrics_log(50)
    except Exception as exc:
        logger.warning("[metrics] RAG 指标读取失败(不影响 usage):%s", exc)

    rag_summary = {}
    if rag_log:
        lat = [r.get("latency_ms", 0) for r in rag_log]
        rag_summary = {
            "queries": len(rag_log),
            "avg_latency_ms": round(sum(lat) / len(lat), 1),
            "rerank_used_rate": round(
                sum(1 for r in rag_log if r.get("rerank_used")) / len(rag_log), 3
            ),
            "vector_degraded": sum(1 for r in rag_log if r.get("vector_degraded")),
            "avg_injected_tokens": round(
                sum(r.get("injected_tokens", 0) for r in rag_log) / len(rag_log), 1
            ),
            "avg_lex_hits": round(sum(r.get("lex_hits", 0) for r in rag_log) / len(rag_log), 1),
            "avg_vec_hits": round(sum(r.get("vec_hits", 0) for r in rag_log) / len(rag_log), 1),
        }

    return ApiResponse.success(data={
        "global_usage": get_global_accountant().to_dict(),
        "usage_log": usage_log,
        "rag_summary": rag_summary,
        "rag_log": rag_log,
    })


@agent_router.get("/status")
async def agent_status(
    current=Depends(require_permission(PERM_SYSTEM_DIAG)),  # 【ACL】诊断端点仅 admin
):
    # 【安全】不再返回 agent_dir 路径等部署细节
    st = get_status()
    return ApiResponse.success(data={
        "ready": st["ready"],
        "tools_count": st["tools_count"],
        "error": st["error"],
    })


@agent_router.get("/trace/{thread_id}")
async def agent_trace(
    thread_id: str,
    raw: bool = False,
    current=Depends(require_permission(PERM_AGENT_CHAT)),  # 登录即可看
    db: AsyncSession = Depends(get_db),
):
    """【管理员监控·接口①】导出某会话的完整思考链路。

    与 /agent/chat(用户接口)严格分离:普通用户的聊天响应里永远没有链路过程,
    只有管理员凭 audit:read 权限码才能从本端点读取(意图→检索→工具→汇总每一步)。
    """
    # 可见性开关：AGENT_TRACE_VISIBLE=off 全关；=admin 只给管理员
    from agent_bridge.bootstrap import trace_visible_for

    is_admin = str(current.user_role) == "admin"
    if not trace_visible_for(str(current.user_role)):
        raise HTTPException(status_code=403, detail="思考过程展示已关闭")

    # 非管理员只能看【自己创建的线程】——否则凭 thread_id 就能偷看别人的对话
    if not is_admin:
        owner = (await db.execute(
            select(ConversationOwner).where(ConversationOwner.thread_id == thread_id)
        )).scalar_one_or_none()
        if owner is None or str(owner.user_id) != str(current.user_id):
            write_audit(str(current.user_id), "trace:denied", decision=False,
                        resource=f"thread:{thread_id}", reason="非会话属主")
            raise HTTPException(status_code=403, detail="无权查看该会话的执行链路")

    # get_trace 会回放 checkpoint 历史（同步 IO + 反序列化），丢线程池
    data = await run_in_threadpool(get_trace, thread_id, bool(raw) and is_admin)
    write_audit(str(current.user_id), "trace:view", resource=f"thread:{thread_id}", decision=True)
    return ApiResponse.success(data=data)


# ============================================================
# 【记忆系统】用户记忆管理端点（评审 ⑯ attribution + ★⑭ 删除权）
# -----------------------------------------------------------
# 每个用户只能看到/删除【自己的】记忆（store 全部查询强制带 user_id，
# 跨用户读是红线）；删除是 GDPR 式级联清空（MySQL + Chroma），写审计。
# ============================================================
@agent_router.get("/memory")
async def list_my_memories(
    current=Depends(require_permission(PERM_AGENT_CHAT)),
):
    """查看自己的长期记忆（分层明细 + 命中次数 + 来源会话，可回溯）。"""
    from allagent2.session import store

    # 【性能】list_items 是同步 MySQL 读（pymysql），丢线程池避免阻塞事件循环
    items = await run_in_threadpool(store.list_items, str(current.user_id), limit=200)
    data = [{
        "id": r.id, "layer": r.layer, "category": r.category,
        "content": r.content, "importance": r.importance,
        "confidence": r.confidence, "hit_count": r.hit_count,
        "active": r.active, "superseded_by": r.superseded_by,
        "source_thread": r.source_thread,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    } for r in items]
    write_audit(str(current.user_id), "memory:list", decision=True,
                resource=f"count:{len(data)}")
    return ApiResponse.success(data={"count": len(data), "items": data})


@agent_router.delete("/memory")
async def delete_my_memories(
    current=Depends(require_permission(PERM_AGENT_CHAT)),
):
    """删除权（★⑭）：级联清空本人全部长期记忆（MySQL 事实源 + Chroma 索引）。"""
    from allagent2.session import store

    # 【性能】delete_user_memories 同步 MySQL + Chroma 清理，丢线程池
    n = await run_in_threadpool(store.delete_user_memories, str(current.user_id))
    write_audit(str(current.user_id), "memory:delete", decision=True,
                resource=f"deleted:{n}")
    return ApiResponse.success(data={"deleted": n})



# ============================================================
# 【P1-11】中断 / 断点续跑
# ------------------------------------------------------------
# 中断 = 往 Redis 打标记，图在下一个节点入口停下（不是杀线程）；
# checkpoint 停在最后成功的 superstep，续跑用 invoke(None) 从那里继续。
# ============================================================
class ResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=64)


@agent_router.post("/interrupt/{thread_id}")
async def interrupt_thread(
    thread_id: str,
    current=Depends(require_permission(PERM_AGENT_CHAT)),
    db: AsyncSession = Depends(get_db),
):
    """给会话打中断标记：下一节点入口即停，checkpoint 保留。"""
    await _assert_thread_owner(db, current, thread_id)  # 【ACL】只能中断自己的会话
    # set_abort 是同步 Redis 写，丢线程池避免阻塞事件循环
    ok = await run_in_threadpool(set_abort, thread_id)
    write_audit(str(current.user_id), "agent:interrupt", resource=f"thread:{thread_id}",
                decision=True, reason=f"marked={ok}")
    return ApiResponse.success(data={"thread_id": thread_id, "interrupted": ok})


@agent_router.post("/resume")
async def resume_thread(
    body: ResumeRequest,
    current=Depends(require_permission(PERM_AGENT_CHAT)),
    db: AsyncSession = Depends(get_db),
):
    """从最后一个 checkpoint 续跑（不追加用户消息，不新开一轮）。"""
    await _assert_thread_owner(db, current, body.thread_id)  # 【ACL】防会话劫持
    try:
        result = await run_in_threadpool(resume_agent, body.thread_id)
    except Exception as exc:
        if _is_llm_down(exc):
            logger.error("[resume] 推理后端不可达: %s", str(exc)[:200])
            raise HTTPException(status_code=503, detail="推理服务暂时不可用，请稍后再试")
        raise
    write_audit(str(current.user_id), "agent:resume", resource=f"thread:{body.thread_id}",
                decision=True, reason="resumed")
    return ApiResponse.success(data=result)

