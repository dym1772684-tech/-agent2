"""Agent 对话接口的请求 / 响应 schema。

【安全改造】对外响应只包含智能体的最终回答 + 结构化业务字段;
query / intent / phase / next_need / tool_results / trace 等链路过程
一律不进 HTTP 响应,只写入服务日志(见 bootstrap.run_agent)。

【ACL 改造】删除 allowed_elder_ids / allowed_tools 两个客户端字段:
权限范围是服务端资源,由 user_elder_binding 表推导,客户端无权指定
(原来客户端不传这两个字段 = 不限定 = 可查任意老人,是越权漏洞)。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    message: str = Field(..., description="向养老智能体提出的问题")
    elder_id: Optional[str] = Field(
        default=None, description="可选,老人编号,用于定位档案;能否访问由服务端 ACL 绑定关系决定"
    )
    thread_id: Optional[str] = Field(
        default=None, description="可选,会话线程ID;必须是本人创建的线程,不传则服务端自动生成并登记"
    )
    identity_id: Optional[str] = Field(
        default=None,
        description="可选,登录时下发的可信身份引用;携带且校验通过则复用,否则服务端重新装配(客户端伪造无效)"
    )
    debug: bool = Field(
        default=False, description="为true时在服务日志中输出逐步执行轨迹(不返回前端)"
    )


class AgentChatData(BaseModel):
    answer: str = Field(..., description="智能体最终自然语言回复(前端唯一可见的链路产物)")
    structured: Dict[str, Any] = Field(
        default_factory=dict,
        description="结构化输出字段(养生养老维度:health_advice/diet_advice/exercise_advice/medication_advice/risk_level/emergency_flag,见 state.PARSED_FIELDS)",
    )
    thread_id: str = Field(..., description="本次会话线程ID,可用于多轮对话")
