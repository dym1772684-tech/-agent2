from pydantic import BaseModel
from typing import Optional


class User(BaseModel):
    username: str
    password: str
    # 【安全/ACL】删除 role 字段:角色是服务端分配的资源,绝不能由注册请求指定
    # (原来客户端可传 "role":"admin" 直接提权)。注册一律由服务端写死 common,
    # admin 账号由种子脚本/管理端创建。
    email: Optional[str] = None
    full_nmae: Optional[str] = None
    disabled: bool = False  # 注册时默认未禁用
    user_id: Optional[str] = None
