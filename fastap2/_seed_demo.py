
import sys, asyncio
sys.path.insert(0, "."); sys.path.insert(0, "..")

async def main():
    from datetime import datetime, timedelta
    from sqlalchemy import text
    from database.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        await db.execute(text("""
            INSERT INTO elder (elder_id, name, age, gender, health_tag, chronic_disease, phone, create_time)
            VALUES ('E001', :n, 78, :g, :t, :c, :p, :ct)
            ON DUPLICATE KEY UPDATE name=VALUES(name), age=VALUES(age)
        """), {"n": "\u5f20\u6842\u5170", "g": "\u5973", "t": "\u81ea\u7406",
                "c": "\u9ad8\u8840\u538b\uff1b2\u578b\u7cd6\u5c3f\u75c5", "p": "13800000001",
                "ct": datetime.now()})
        await db.commit()

        rows = (await db.execute(text(
            "SELECT COUNT(*) FROM health_record WHERE elder_id='E001'"))).scalar()
        if rows:
            print("health_record 已有", rows, "条，跳过造数")
        else:
            data = [("\u8840\u538b", 135, 6), ("\u8840\u538b", 142, 5), ("\u8840\u538b", 148, 4),
                    ("\u8840\u538b", 150, 3), ("\u8840\u538b", 146, 2), ("\u8840\u538b", 139, 1),
                    ("\u8840\u7cd6", 6.1, 4), ("\u8840\u7cd6", 5.8, 3), ("\u8840\u7cd6", 6.5, 2),
                    ("\u8840\u7cd6", 7.2, 1), ("\u5fc3\u7387", 78, 2), ("\u5fc3\u7387", 82, 1)]
            for metric, value, ago in data:
                await db.execute(text("""
                    INSERT INTO health_record
                        (elder_id, device_id, metric_type, value, risk_content, extra, record_time, create_time)
                    VALUES ('E001', 'DEV-DEMO', :m, :v, '{}', '{}', :rt, :ct)
                """), {"m": metric, "v": float(value),
                        "rt": datetime.now() - timedelta(days=ago), "ct": datetime.now()})
            await db.commit()
            print("已写入 E001 老人档案 + 12 条健康记录")

        e = (await db.execute(text("SELECT elder_id, name, age FROM elder WHERE elder_id='E001'"))).all()
        h = (await db.execute(text("SELECT COUNT(*) FROM health_record WHERE elder_id='E001'"))).scalar()
        print("elder:", e, "| health_record:", h)

asyncio.run(main())
