"""
MindBridge 认证与授权模块

使用 HTTP Basic Auth 实现用户认证：
- hash_password(): SHA-256 哈希密码（明文密码不存储）
- verify_password(): 使用 hmac.compare_digest 防时序攻击
- _credentials(): 从请求头解析 Basic Auth 凭据
- current_user(): FastAPI 依赖注入，验证凭据并返回用户对象
- require_admin(): FastAPI 依赖注入，要求 ROLE_ADMIN 角色

认证流程：
1. 客户端在请求头携带 Authorization: Basic base64(username:password)
2. _credentials() 解码并提取用户名和密码
3. current_user() 查询数据库验证凭据
4. require_admin() 在 current_user() 基础上检查管理员角色
"""
import base64
import hashlib
import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import UserAccount


def hash_password(password: str) -> str:
    """
    将明文密码哈希为 SHA-256 十六进制字符串。

    注意：生产环境应使用 bcrypt/argon2 等慢哈希算法。
    当前方案仅适用于演示和原型。
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """
    验证明文密码是否匹配哈希值。

    使用 hmac.compare_digest 进行常量时间比较，
    防止通过响应时间差异推断密码哈希的逐字符匹配情况。
    """
    return hmac.compare_digest(hash_password(password), hashed)


def _credentials(request: Request) -> tuple[str, str]:
    """
    从 HTTP 请求头解析 Basic Auth 凭据。

    返回 (username, password) 元组。
    缺少或格式错误时抛出 401 异常。
    """
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing Basic authorization")
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        username, password = decoded.split(":", 1)
        return username, password
    except Exception as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Basic authorization") from exc


def current_user(request: Request, db: Annotated[Session, Depends(get_db)]) -> UserAccount:
    """
    FastAPI 依赖注入函数：验证 Basic Auth 凭据并返回当前用户。

    使用方式：user: UserAccount = Depends(current_user)

    流程：
    1. 从请求头解析用户名和密码
    2. 查询数据库获取用户记录
    3. 验证密码哈希
    4. 任一步骤失败则抛出 401
    """
    username, password = _credentials(request)
    user = db.query(UserAccount).filter(UserAccount.username == username).first()
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bad credentials")
    return user


def require_admin(user: Annotated[UserAccount, Depends(current_user)]) -> UserAccount:
    """
    FastAPI 依赖注入函数：要求当前用户具有管理员角色。

    使用方式：admin: UserAccount = Depends(require_admin)

    在 current_user() 基础上增加角色检查。
    非管理员用户访问时抛出 403。
    """
    if "ROLE_ADMIN" not in user.roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return user
