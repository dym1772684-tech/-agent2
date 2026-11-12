"""管理员种子脚本：创建或提升 admin 账号。

背景：/api/v1/register 出于安全考虑把角色写死为 common
（防止客户端传 role=admin 提权），所以 admin 账号只能在服务端
用本脚本创建/提升 —— 这也是 core/acl.py 注释里约定的方式。

用法（在 fastap2 目录下，需 MySQL 已启动）：
    E:/python/python.exe create_admin.py 用户名 密码

行为：
- 用户已存在 → 把该用户 user_role 提升为 admin
- 用户不存在 → 直接创建 admin 账号
- 密码用与注册接口完全一致的 bcrypt 逻辑哈希，登录接口通用
"""
import asyncio
import sys
import uuid

from sqlalchemy import select

from core.settings import hash_password
from database.database import AsyncSessionLocal
from models.users import User_table
from schemas.user import User


async def main(username: str, password: str) -> None:
    async with AsyncSessionLocal() as db:
        row = await db.execute(select(User_table).where(User_table.username == username))
        user = row.scalar_one_or_none()
        if user is not None:
            user.user_role = "admin"
            action = f"已存在用户 {username}，角色提升为 admin"
        else:
            pwd = User(username=username, password=password)
            user = User_table(
                username=username,
                password=hash_password(pwd),
                user_id=str(uuid.uuid4()),
                user_role="admin",
            )
            db.add(user)
            action = f"已创建 admin 账号 {username}"
        # 【修复】commit 后属性会过期，隐式刷新在 async 上下文外触发
        # MissingGreenlet；故在 commit 前先取好要打印的值。
        uid = user.user_id
        await db.commit()
        print(f"[OK] {action} (user_id={uid})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python create_admin.py <用户名> <密码>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
