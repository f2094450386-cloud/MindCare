"""
应用启动初始化模块

职责：
- create_schema(): 根据 ORM 模型创建所有数据库表（幂等操作）
- seed_data(): 初始化默认数据：
  1. 如果 user_accounts 表为空，创建 admin 和 student 两个默认账号
  2. 同步 app/knowledge/*.md 内置知识库到数据库（按 source 去重，内容变化时刷新）

调用时机：在 main.py 的 startup 事件中调用，确保每次启动时数据库就绪。
"""
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import Base, engine
from app.core.security import hash_password
from app.models.entities import UserAccount
from app.services.knowledge import KnowledgeService


def create_schema() -> None:
    """
    根据所有 ORM 模型创建数据库表。

    幂等操作：已存在的表不会被重建。
    在 FastAPI startup 事件中调用。
    """
    Base.metadata.create_all(bind=engine)


def seed_data(db: Session) -> None:
    """
    初始化默认用户和内置知识库。

    1. 用户初始化：仅在 user_accounts 表为空时创建：
       - admin: 管理员账号，拥有 ROLE_ADMIN + ROLE_USER 角色
       - student: 演示学生账号，只有 ROLE_USER 角色

    2. 知识库同步：扫描 app/knowledge/*.md 文件，
       通过 KnowledgeService.ensure_source() 按 source 名去重。
       如果 md 文件内容发生变化（切块结果不同），会重新入库。
    """
    # ── 默认用户初始化 ────────────────────────────────────────────
    if db.query(UserAccount).count() == 0:
        admin = UserAccount(
            username="admin",
            display_name="Counselor Admin",
            password_hash=hash_password("admin123"),
        )
        admin.roles = {"ROLE_ADMIN", "ROLE_USER"}
        student = UserAccount(
            username="student",
            display_name="Demo Student",
            password_hash=hash_password("student123"),
        )
        student.roles = {"ROLE_USER"}
        db.add_all([admin, student])
        db.commit()

    # ── 内置知识库同步 ────────────────────────────────────────────
    # 遍历 app/knowledge/ 下所有 .md 文件，按文件名作为 source 标识
    # ensure_source 会比较现有切块和新切块，内容一致则跳过
    service = KnowledgeService(db, get_settings())
    root = Path(__file__).resolve().parents[1]
    for file in sorted((root / "knowledge").glob("*.md")):
        service.ensure_source(file.name, file.read_text(encoding="utf-8"))
