"""
MindBridge FastAPI 应用入口

创建和配置 FastAPI 应用实例。

启动流程：
1. create_app() 创建 FastAPI 实例
2. 注册 HTTP 中间件（前端资源禁用缓存）
3. 注册 startup 事件：
   - 创建数据库表（create_schema）
   - 初始化默认数据（seed_data）
   - 启动工具队列后台调度器
4. 注册 shutdown 事件：停止工具队列调度器
5. 注册 API 路由
6. 挂载静态文件目录（前端 HTML/CSS/JS）

使用方式：
  uvicorn app.main:app --host 127.0.0.1 --port 8080
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.bootstrap import create_schema, seed_data
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.services.tool_queue import get_tool_queue_worker


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例。

    包含所有中间件、事件处理器和路由注册。
    """
    app = FastAPI(title="MindBridge Python", version="0.1.0")

    # ── HTTP 中间件：前端资源禁用缓存 ──────────────────────────────
    # 开发阶段确保每次请求都获取最新的前端资源
    @app.middleware("http")
    async def no_cache_frontend_assets(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store"
        return response

    # ── 启动事件 ────────────────────────────────────────────────────
    @app.on_event("startup")
    def startup() -> None:
        # 创建数据库表
        create_schema()
        # 初始化默认数据（用户、知识库）
        db = SessionLocal()
        try:
            seed_data(db)
        finally:
            db.close()
        # 启动工具队列后台调度器
        worker = get_tool_queue_worker(get_settings())
        worker.start()
        app.state.tool_queue_worker = worker

    # ── 关闭事件 ────────────────────────────────────────────────────
    @app.on_event("shutdown")
    def shutdown() -> None:
        worker = getattr(app.state, "tool_queue_worker", None)
        if worker is not None:
            worker.stop()

    # ── 注册路由和静态文件 ──────────────────────────────────────────
    app.include_router(router)
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


# 全局应用实例（uvicorn 入口）
app = create_app()
