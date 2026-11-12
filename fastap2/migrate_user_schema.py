"""user 表结构补齐脚本（唯一约束 + NOT NULL）——幂等，可反复执行。

================================================================
为什么需要这个脚本
----------------------------------------------------------------
`Base.metadata.create_all`（main.py 的 init_db）**只建表、不改表**：
models/users.py 里声明的 `unique=True` 与 `nullable=False`，
对"表已经存在"的库一律不生效。

实测本机 agent_demo.user 的现状（脚本运行前）：
    username/password/user_id/user_role 全部 DEFAULT NULL，且只有 PRIMARY KEY。
后果：
    · 并发注册能写出同名账号（check_password 取 .order_by(id).first() 就是为它打的补丁）；
    · 空用户名/空密码行能被写入，登录逻辑要靠运气；
    · 换到 Linux MySQL（表名大小写敏感）还会因为模型写 "User"、库里是 "user"
      再建出一张空表。

用法：
    E:/python/python.exe migrate_user_schema.py --check    # 只体检，不改
    E:/python/python.exe migrate_user_schema.py            # 执行（幂等）
================================================================
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "database", "mysql.env"))

TABLE = "user"
UK_USERNAME = "uq_user_username"
UK_USERID = "uq_user_user_id"
NOT_NULL_COLS = ("username", "password", "user_id", "user_role")


def _engine():
    url = (
        f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
        f"@{os.getenv('MYSQL_HOST', '127.0.0.1')}:{os.getenv('MYSQL_PORT', '3306')}"
        f"/{os.getenv('MYSQL_DATABASE')}?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True)


def _columns(conn) -> dict:
    rows = conn.execute(
        text("""SELECT COLUMN_NAME, IS_NULLABLE, COLUMN_TYPE, COLUMN_DEFAULT
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"""),
        {"t": TABLE},
    ).all()
    return {r[0]: {"nullable": r[1] == "YES", "type": r[2], "default": r[3]} for r in rows}


def _indexes(conn) -> set:
    rows = conn.execute(
        text("""SELECT DISTINCT INDEX_NAME FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"""),
        {"t": TABLE},
    ).all()
    return {r[0] for r in rows}


def _dupes(conn, col: str) -> list:
    return conn.execute(text(
        f"SELECT {col}, COUNT(*) c FROM `{TABLE}` WHERE {col} IS NOT NULL "
        f"GROUP BY {col} HAVING c > 1"
    )).all()


def _nulls(conn, col: str) -> int:
    return conn.execute(text(f"SELECT COUNT(*) FROM `{TABLE}` WHERE {col} IS NULL")).scalar() or 0


def main() -> int:
    ap = argparse.ArgumentParser(description="user 表唯一约束 / NOT NULL 补齐")
    ap.add_argument("--check", action="store_true", help="只体检不修改")
    args = ap.parse_args()

    eng = _engine()
    with eng.begin() as conn:
        cols = _columns(conn)
        if not cols:
            print(f"[!] 表 `{TABLE}` 不存在（先启动一次服务让 create_all 建表）")
            return 1
        idx = _indexes(conn)
        total = conn.execute(text(f"SELECT COUNT(*) FROM `{TABLE}`")).scalar()

        print(f"[体检] {TABLE}: {total} 行, 索引={sorted(idx)}")
        problems = []
        for c in NOT_NULL_COLS:
            info = cols.get(c)
            if info is None:
                print(f"  [!] 缺列 {c}")
                continue
            if info["nullable"]:
                problems.append(f"{c} 可空")
        if UK_USERNAME not in idx:
            problems.append("缺唯一索引 " + UK_USERNAME)
        if UK_USERID not in idx:
            problems.append("缺唯一索引 " + UK_USERID)

        if not problems:
            print("[OK] 结构已符合模型声明，无需变更")
            return 0
        print("[问题] " + "; ".join(problems))

        # ---- 数据前置校验：有重复/NULL 就不能加约束 ----
        blockers = []
        for col, uk in (("username", UK_USERNAME), ("user_id", UK_USERID)):
            d = _dupes(conn, col)
            if d:
                blockers.append(f"{col} 有重复值: {d[:5]}")
        for c in NOT_NULL_COLS:
            n = _nulls(conn, c)
            if n and c in ("username", "password", "user_id"):
                blockers.append(f"{c} 有 {n} 行 NULL（空值行无法自动补齐，请先清理）")
        if blockers:
            print("[中止] 存在脏数据，先修数据再迁移：")
            for b in blockers:
                print("   -", b)
            return 2

        if args.check:
            print("[check] 未做修改（去掉 --check 执行）")
            return 0

        # ---- 可安全补齐的可空列：user_role 给默认值 ----
        fixed = conn.execute(text(
            f"UPDATE `{TABLE}` SET user_role='common' WHERE user_role IS NULL"
        )).rowcount
        if fixed:
            print(f"[补齐] user_role NULL → 'common'：{fixed} 行")

        # ---- 加唯一索引（幂等：先看索引名）----
        for uk, col in ((UK_USERNAME, "username"), (UK_USERID, "user_id")):
            if uk in idx:
                continue
            conn.execute(text(f"ALTER TABLE `{TABLE}` ADD UNIQUE KEY `{uk}` (`{col}`)"))
            print(f"[DDL] + UNIQUE {uk}({col})")

        # ---- 改 NOT NULL（保留原类型，避免改到长度/字符集）----
        for c in NOT_NULL_COLS:
            info = cols.get(c)
            if info is None or not info["nullable"]:
                continue
            extra = ""
            if c == "user_role":
                extra = " DEFAULT 'common'"
            conn.execute(text(
                f"ALTER TABLE `{TABLE}` MODIFY `{c}` {info['type']} NOT NULL{extra}"
            ))
            print(f"[DDL] {c} → NOT NULL{extra}")

        cols2 = _columns(conn)
        idx2 = _indexes(conn)
        ok = (all(not cols2[c]["nullable"] for c in NOT_NULL_COLS if c in cols2)
              and UK_USERNAME in idx2 and UK_USERID in idx2)
        print("[完成] 结构校验:", "通过" if ok else "仍未达标，请检查")
        return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
