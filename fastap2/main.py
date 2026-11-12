# main.py
from contextlib import asynccontextmanager

import time

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from fastapi.responses import FileResponse
from database.database import Base, engine
from routers.health import router
from routers.elder import elder_router
from models.elder import Elder
import uvicorn
from models.health import HealthRecord,HealthRisk
from routers.users import user_router 
from routers.agent import agent_router
from routers.acl import acl_router  # 【ACL】授权管理端点
from routers.audit import audit_router
from routers.files import files_router  # 文件上传（解析成文本供对话引用）
from routers.skills import skill_router  # 技能启停/反馈
from observability import metrics  # Prometheus（软依赖，未安装则自动跳过）  # 【审计】管理员访问日志端点
from models.acl import UserElderBinding, ConversationOwner  # 【ACL】注册新表进 metadata,create_all 才会建
from models.audit import AuditLog  # 【审计】注册审计表进 metadata
from core.request_id import (  # 【可观测】请求级追踪 ID（原模块写好但从未挂载）
    install_logging,
    new_request_id,
    reset_request_id,
    set_request_id,
)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 启动自动创建不存在的表；已经存在不会覆盖
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 【修复】原 init_db 定义了但从未被调用:ACL 新表(user_elder_binding 等)不会创建
    await init_db()
    # metrics.install() 要 import allagent2；先把 allagent2 挂进 sys.path，
    # 否则启动期会报 "No module named 'allagent2'"，指标 sink 永远装不上。
    try:
        from agent_bridge.bootstrap import _register_packages

        _register_packages()
    except Exception as exc:
        print(f"[startup] 注册 allagent2 包失败：{exc}")
    # 指标注册 + 把 sink 注入 allagent2（单向：fastap2 -> allagent2）
    if metrics.install():
        metrics.publish_runtime_state()
    # 【可观测】给所有日志 handler 挂 request_id Filter（幂等）
    try:
        install_logging()
    except Exception as exc:
        print(f"[startup] request_id 日志注入失败：{exc}")
    yield

app = FastAPI(title="养老智能体", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.include_router(user_router,prefix="/api/v1")
app.include_router(router, prefix="/api/v1")
app.include_router(elder_router,prefix="/api/v1")
app.include_router(agent_router,prefix="/api/v1")
app.include_router(acl_router,prefix="/api/v1")  # 【ACL】
app.include_router(audit_router,prefix="/api/v1")
app.include_router(files_router,prefix="/api/v1")
app.include_router(skill_router,prefix="/api/v1")  # 【审计】

@app.get("/")
async def index_page():
    return FileResponse(BASE_DIR / "chat.html")

@app.get("/dashboard")
async def dashboard_page():
    return FileResponse(BASE_DIR / "chat.html")

@app.get("/auth")
async def auth_page():
    return FileResponse(BASE_DIR / "login.html")

@app.get("/admin")
async def admin_page():
    """管理后台（前端再校验一次角色，非 admin 跳回对话页）。"""
    return FileResponse(BASE_DIR / "admin.html")

@app.middleware("http")
async def http_request_id(request: Request, call_next):
    """【可观测】每请求一个 rid：入口生成（或复用网关传来的 X-Request-ID），
    写进 contextvars → 所有日志自动带 rid=...；同时回写响应头便于前端/网关串联。

    contextvars 会随 run_in_threadpool / LangGraph 节点自动复制，
    所以 agent 线程池里打的日志也带同一个 rid（后台线程无请求上下文时是 "-"）。
    """
    rid = request.headers.get("X-Request-ID") or new_request_id()
    token = set_request_id(rid)
    try:
        response = await call_next(request)
    finally:
        reset_request_id(token)
    response.headers["X-Request-ID"] = rid
    return response


@app.middleware("http")
async def http_metrics(request: Request, call_next):
    """HTTP 耗时指标：用路由模板做 label，避免每个 thread_id 生成一个 label。"""
    t0 = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", None) or request.url.path
    try:
        metrics.observe_http(request.method, path, response.status_code,
                             time.perf_counter() - t0)
    except Exception:  # 指标失败绝不影响业务响应
        pass
    return response


_metrics_app = metrics.asgi_app()
if _metrics_app is not None:
    app.mount("/metrics", _metrics_app)



if __name__=="__main__":
    uvicorn.run("main:app",
                host="127.0.0.1",
                port=8000,
                reload=True
                )#uvicorn ft:app --reload命令行启动
