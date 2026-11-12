"""ACL 数据模型：RBAC + 对象级授权。

role_permission    :功能级 RBAC —— 角色 → 权限码（core/identity.PERM_REGISTRY 是全集）
user_elder_binding :用户 ↔ 老人 绑定关系 —— 决定"谁能访问哪个老人的数据行"
conversation_owner :会话线程属主     —— 决定"谁能续接哪个 thread_id"(防会话劫持)
"""
import datetime

from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint

from database.database import Base


class RolePermission(Base):
    """角色-权限码映射（RBAC）。

    identity.load_role_permissions 读取；表为空/缺失时回退
    ROLE_DEFAULT_PERMS 硬编码（平滑迁移，不阻塞现有链路）。
    """
    __tablename__ = "role_permission"

    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String(32), nullable=False, index=True)
    perm_code = Column(String(64), nullable=False, index=True)  # 见 identity.PERM_REGISTRY
    active = Column(Integer, nullable=False, default=1)         # 软删
    create_time = Column(DateTime, default=datetime.datetime.now)

    __table_args__ = (
        UniqueConstraint("role", "perm_code", name="uq_role_perm"),
    )


class UserElderBinding(Base):
    """用户-老人绑定:对象级 ACL 的授权记录(护工/家属 ↔ 老人)。"""
    __tablename__ = "user_elder_binding"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    elder_id = Column(String(64), nullable=False, index=True)
    relation = Column(String(32), default="caregiver")  # 关系:caregiver / family
    create_time = Column(DateTime, default=datetime.datetime.now)

    __table_args__ = (
        UniqueConstraint("user_id", "elder_id", name="uq_user_elder"),
    )


class ConversationOwner(Base):
    """会话线程属主:thread_id 一律服务端登记,续接他人线程 → 403。"""
    __tablename__ = "conversation_owner"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(64), nullable=False, unique=True, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    create_time = Column(DateTime, default=datetime.datetime.now)
