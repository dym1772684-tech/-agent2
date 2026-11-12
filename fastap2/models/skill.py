"""用户-技能启用状态。

技能定义本身是文件（allagent2/skills/builtin/<id>/SKILL.md），不需要建表；
这张表只存"某个用户启用了哪些技能、用了多少次、反馈如何"。

设计取舍：
- 不在 user 表加字段（技能增删要改表结构，不可接受）；
- 不塞进 memory_items（技能是系统能力，不是用户记忆，混在一起权限与生命周期都乱）；
- `enabled` 与 `auto` 分开：enabled=用户允许这个技能存在，auto=默认每轮都带上。
  手动选一次不等于常开，所以这两个语义必须分开。
"""
import datetime

from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint

from database.database import Base


class UserSkillState(Base):
    __tablename__ = "user_skill_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    skill_id = Column(String(64), nullable=False, index=True)
    enabled = Column(Integer, nullable=False, default=1)      # 用户是否允许使用
    auto = Column(Integer, nullable=False, default=0)         # 是否默认每轮注入（常开）
    use_count = Column(Integer, nullable=False, default=0)    # 被注入次数
    success_count = Column(Integer, nullable=False, default=0)  # 用户反馈"有用"
    fail_count = Column(Integer, nullable=False, default=0)     # 用户反馈"没用"
    last_used_at = Column(DateTime, nullable=True)
    create_time = Column(DateTime, default=datetime.datetime.now)

    __table_args__ = (
        UniqueConstraint("user_id", "skill_id", name="uq_user_skill"),
    )
