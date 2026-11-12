"""S2 零信任体系数据库迁移（幂等，可重复执行）。

1. User 表加 status 列（active/locked/disabled，存量行回填 active）
2. 建 role_permission 表（RBAC；空表 → identity 回退硬编码 ROLE_DEFAULT_PERMS）

用法：cd fastap2 && python migrate_identity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathlib import Path as P

from dotenv import load_dotenv

load_dotenv(P(__file__).resolve().parent / "database" / "mysql.env")
load_dotenv(P(__file__).resolve().parent / "core" / "key.env")

import os

from sqlalchemy import create_engine, text

url = (f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
       f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DATABASE')}"
       "?charset=utf8mb4")
eng = create_engine(url)

with eng.begin() as c:
    # ① User.status（已存在则跳过）
    cols = [r[0] for r in c.execute(text("SHOW COLUMNS FROM `User`"))]
    if "status" not in cols:
        c.execute(text("ALTER TABLE `User` ADD COLUMN status VARCHAR(16) "
                       "NOT NULL DEFAULT 'active'"))
        print("已添加 User.status 列（存量行=active）")
    else:
        print("User.status 已存在，跳过")

    # ② role_permission 表
    c.execute(text("""
        CREATE TABLE IF NOT EXISTS role_permission (
          id INT AUTO_INCREMENT PRIMARY KEY,
          role VARCHAR(32) NOT NULL,
          perm_code VARCHAR(64) NOT NULL,
          active INT NOT NULL DEFAULT 1,
          create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uq_role_perm (role, perm_code),
          INDEX idx_role (role)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """))
    n = c.execute(text("SELECT COUNT(*) FROM role_permission")).scalar()
    print(f"role_permission 表就绪（当前 {n} 行；0 行=回退硬编码映射，属正常）")
print("迁移完成")
