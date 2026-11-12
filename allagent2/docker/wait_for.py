"""容器启动前依赖就绪探测（被 entrypoint.sh 调用）。

退出码 0 = 就绪；非 0 = 未就绪（entrypoint 会重试）。
用法： python wait_for.py mysql <host> <port>
       python wait_for.py redis <host> <port>
"""
import os
import sys


def wait_mysql(host: str, port: str) -> bool:
    import pymysql

    try:
        conn = pymysql.connect(
            host=host,
            port=int(port),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "agent_demo"),
            connect_timeout=3,
            read_timeout=3,
        )
        conn.close()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[wait_for] mysql not ready: {exc}")
        return False


def wait_redis(host: str, port: str) -> bool:
    import redis

    try:
        client = redis.Redis(
            host=host,
            port=int(port),
            password=os.getenv("REDIS_PASSWORD") or None,
            socket_connect_timeout=3,
        )
        return bool(client.ping())
    except Exception as exc:  # noqa: BLE001
        print(f"[wait_for] redis not ready: {exc}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: wait_for.py <mysql|redis> <host> <port>")
        sys.exit(2)
    kind, host, port = sys.argv[1], sys.argv[2], sys.argv[3]
    ok = wait_mysql(host, port) if kind == "mysql" else wait_redis(host, port)
    sys.exit(0 if ok else 1)
