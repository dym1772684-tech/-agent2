import asyncio
from logging.config import fileConfig
import os

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine # 注意这里改成 create_async_engine
from alembic import context
from dotenv import load_dotenv, find_dotenv

# 1. 导入你的 Base
from database.database import Base

# 2. 加载环境变量文件
# 如果你的文件叫 mysql.env 且在 database 目录下，直接指定相对路径
# 使用 find_dotenv 的好处是它在任何终端目录下运行都能自动找到
load_dotenv("database/mysql.env") 

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 3. 读取环境变量（匹配你提供的 MYSQL_ 前缀）
db_user = os.getenv("MYSQL_USER", "root")
db_pass = os.getenv("MYSQL_PASSWORD", "你的密码")
db_host = os.getenv("MYSQL_HOST", "localhost")
db_port = os.getenv("MYSQL_PORT", "3306")
db_name = os.getenv("MYSQL_DATABASE", "agent_demo")

# 4. 拼装成 aiomysql 的异步驱动 URL
async_url = f"mysql+aiomysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# 把拼好的 URL 注入到 config（虽然我们下面直接用了，但留着有益无害）
config.set_main_option("sqlalchemy.url", async_url)

# 5. 【极其重要】必须是 Base.metadata
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    sync_url = url.replace("mysql+aiomysql", "mysql+pymysql") if url else None
    
    context.configure(
        url=sync_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode."""
    # 直接用 create_async_engine 硬传 async_url，彻底避免读取 ini 报错
    connectable = create_async_engine(
        async_url,
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()