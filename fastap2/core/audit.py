"""审计写入器(同步):给鉴权/授权/ACL 管理等关键路径落审计日志。

设计要点:
- 独立小同步引擎:审计不与业务 engine(aiomysql 异步)混用,跑在线程池里也安全;
- **审计失败绝不影响主流程**:write_audit 内部吞异常只打日志——
  不能因为审计库抖一下,把用户的正常请求/正常的拒绝响应搞挂;
- 每次调用独立 Session:审计 QPS 低(只在鉴权决策点),开销可接受。
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from models.audit import AuditLog

logger = logging.getLogger("audit")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "database" / "mysql.env")

_engine = None


def _get_engine():
    """懒加载审计专用同步引擎(pymysql)。"""
    global _engine
    if _engine is None:
        url = (
            f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
            f"@{os.getenv('MYSQL_HOST', '127.0.0.1')}:{os.getenv('MYSQL_PORT', '3306')}"
            f"/{os.getenv('MYSQL_DATABASE')}?charset=utf8mb4"
        )
        _engine = create_engine(url, pool_pre_ping=True, pool_size=5)
    return _engine


def write_audit(user_id: str, action: str, resource: str, decision: bool, reason: str = "") -> None:
    """落一条审计。同步函数:在 async 代码里调用时建议丢线程池(量小直接调也可)。

    同时打一行结构化日志——即使审计表写失败,日志里也有记录。
    """
    try:
        with Session(_get_engine()) as session:
            session.add(AuditLog(
                user_id=str(user_id or "anonymous"),
                action=action,
                resource=resource,
                decision=1 if decision else 0,
                reason=reason,
            ))
            session.commit()
        logger.info("[audit] user=%s action=%s resource=%s decision=%s reason=%s",
                    user_id, action, resource, decision, reason)
    except Exception as e:  # 审计失败不影响业务
        logger.error("[audit] 写入失败 user=%s action=%s err=%s", user_id, action, e)
