from sqlalchemy import Column, String, Text, DateTime, Integer
from sqlalchemy.orm import Session
from sqlalchemy import UniqueConstraint
from database.database import Base
import datetime
class User_table(Base):
    # 【修复】原来写 "User"：Windows MySQL 表名大小写不敏感看着没事，
    # 换到 Linux（lower_case_table_names=0）会按 "User" 再建一张空表，
    # 而真实数据在 "user" 里 → 登录永远查不到人。库里的实际表名是小写。
    __tablename__="user"
    id=Column(Integer,primary_key=True,autoincrement=True)
    username=Column(String(64), unique=True, nullable=False)
    password=Column(String(64), nullable=False)
    create_time=Column(DateTime,default=datetime.datetime.now)
    user_id=Column(String(64), unique=True, nullable=False)
    user_role=Column(String(64), default="common")
    # 【零信任·第①步】账户实时状态：active 正常 / locked 登录锁定 / disabled 已禁用。
    # 变更时须同步 identity.revoke_user(user_id)（perm_version+1，在线身份即时失效）
    status=Column(String(16), nullable=False, default="active")

    __table_args__ = (
        UniqueConstraint('username', name='uq_user_username'),
        UniqueConstraint('user_id', name='uq_user_user_id'),
    )
