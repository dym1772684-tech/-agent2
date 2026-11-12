"""审计日志表:谁(user_id)在何时对什么资源做了什么动作,允许还是拒绝。

医疗/养老场景的硬性要求,也是发现"被人试探权限"的唯一窗口:
- 鉴权通过(decision=True) → 留痕"谁看了哪个老人的数据"
- 鉴权拒绝(decision=False)→ 留痕"谁在越权,试了什么"
"""
import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from database.database import Base


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)   # 操作者(未登录=anonymous)
    action = Column(String(64), nullable=False)                # 动作:gateway:query / acl:bind / ...
    resource = Column(String(128), nullable=True)              # 对象:elder:E0001 / thread:t-xxx
    decision = Column(Integer, nullable=False, default=1)      # 1=允许 0=拒绝
    reason = Column(Text, nullable=True)                       # 拒绝原因/备注
    create_time = Column(DateTime, default=datetime.datetime.now, index=True)
