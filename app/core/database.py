"""
SQLAlchemy 数据库引擎与会话管理模块

职责：
- 创建 SQLAlchemy 引擎（支持 MySQL 和 SQLite）
- 提供 SessionLocal 工厂函数
- 提供 get_db() 依赖注入函数（FastAPI Depends 用）
- 提供 session_scope() 用于非请求上下文（如 harness 测试）

注意：
- MySQL 使用 pymysql 驱动，连接串格式：mysql+pymysql://user:pass@host:port/db
- SQLite 用于 harness 测试，需要 check_same_thread=False
- pool_pre_ping=True 防止连接池中的死连接
- pool_recycle=3600 每小时回收连接，防止 MySQL 超时断开
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """SQLAlchemy ORM 声明式基类，所有实体模型继承此类。"""
    pass


settings = get_settings()

# 引擎参数：连接池保活和回收
engine_kwargs = {
    "pool_pre_ping": True,    # 每次取连接前先 ping，丢弃死连接
    "pool_recycle": 3600,     # 连接最大存活时间（秒），防止 MySQL wait_timeout 超时
}

# SQLite 特殊处理：多线程访问需要关闭线程检查
if settings.database_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    **engine_kwargs,
)

# 会话工厂
# autoflush=False: 不在查询前自动 flush 未提交的修改，避免意外副作用
# autocommit=False: 显式控制事务边界
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """
    FastAPI 依赖注入函数，为每个请求提供一个数据库会话。

    使用 yield 模式确保请求结束后会话被正确关闭。
    典型用法：db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def session_scope() -> Session:
    """
    非请求上下文的会话获取函数。

    用于 harness 测试、后台任务等不在 FastAPI 请求链路中的场景。
    调用方需自行管理会话生命周期（try/finally db.close()）。
    """
    return SessionLocal()
