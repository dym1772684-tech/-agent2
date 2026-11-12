from typing import Optional

from pydantic import BaseModel, Field
from datetime import datetime


class HealthRecord_create(BaseModel):
    # 主键改为可选：留空交给数据库自增（原来强制客户端给 id，前端无从得知下一个号）
    health_id: Optional[int] = Field(default=None, description="记录主键；留空由数据库自增")
    elder_id: str = Field(..., max_length=64)
    device_id: str = Field(..., max_length=64)
    metric_type: str = Field(..., description="指标名，如 血压 / 血糖 / 心率 / 体温 / 血氧")
    risk_content: dict = Field(default_factory=dict)
    value: float
    extra: dict = Field(default_factory=dict)
    record_time: datetime = Field(default_factory=datetime.now)


