"""
MindBridge 全局配置模块

使用 pydantic-settings 从 .env 文件和环境变量加载所有运行时配置。
所有配置项都有合理默认值，支持通过 .env 文件覆盖。

配置分组：
- AI 模型相关：provider、模型名、temperature、max_tokens
- 数据库与缓存：MySQL、Redis 连接参数
- RAG 知识库：切块大小、向量/关键词融合权重、reranker 开关
- Chroma 向量库：持久化目录、快照策略
- 邮件预警：SMTP 配置、收件人、投递模式
- 工具队列：异步任务轮询间隔、重试策略、限流参数
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    MindBridge 应用配置类。

    继承 pydantic-settings 的 BaseSettings，自动从 .env 文件读取配置。
    所有字段都有默认值，未配置时使用默认值运行（mock 模式）。

    属性说明见各字段注释。
    """

    # ── Agent 编排框架 ──────────────────────────────────────────────
    # "langgraph" = 使用 LangGraph 构建多 Agent 有向图（需要安装 langgraph）
    # "custom"    = 使用自研有限循环 runtime（无外部依赖兜底方案）
    agent_framework: str = "langgraph"

    # ── AI 模型配置 ────────────────────────────────────────────────
    # ai_provider: "ollama" / "openai" / "mock"
    #   - ollama: 本地 Ollama 服务，使用本地微调 GGUF 模型
    #   - openai: OpenAI-compatible API（支持 GPT-4o-mini 等）
    #   - mock:   离线模拟，用于开发和测试
    ai_provider: str = "ollama"
    ai_temperature: float = 0.35       # 生成温度，越低越确定性
    ai_max_tokens: int = 512           # 单次生成最大 token 数

    # Ollama 本地模型配置
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mindbridge-qwen2.5-7b-ft:latest"

    # 微调模型资产路径（用于 /api/agent/status 检查 GGUF 和 Modelfile 是否就绪）
    finetuned_model_name: str = "mindbridge-qwen2.5-7b-ft:latest"
    finetuned_model_dir: str = "models/mindbridge-qwen2.5-7b-ft"
    finetuned_model_file: str = "mindbridge-qwen2.5-7b-ft-q4_k_m.gguf"

    # OpenAI-compatible API 配置（也可用于 embeddings）
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── 数据库配置 ─────────────────────────────────────────────────
    # 默认 MySQL；harness 测试时会切换为 SQLite
    database_url: str = "mysql+pymysql://mindbridge:mindbridge@127.0.0.1:3306/mindbridge?charset=utf8mb4"
    chat_history_limit: int = 10       # 注入 LLM 的最近对话轮数（每轮=user+assistant）

    # ── RAG 知识库配置 ─────────────────────────────────────────────
    knowledge_top_k: int = 4           # 最终返回给 LLM 的知识条数
    knowledge_candidate_k: int = 16    # 向量+BM25 融合前的候选条数
    knowledge_chunk_size: int = 512    # 知识文档切块字符数
    knowledge_chunk_overlap: int = 64  # 相邻切块重叠字符数（防止语义断裂）

    # 混合检索权重：向量语义召回 vs BM25 关键词召回
    knowledge_hybrid_vector_weight: float = 0.65
    knowledge_hybrid_bm25_weight: float = 0.35

    # reranker 开关：融合后是否再做一次本地 rerank 打分
    knowledge_rerank_enabled: bool = True

    # 向量检索开关和强制要求
    # vector_enabled=false → 完全不初始化 Chroma
    # vector_required=false → Chroma 不可用时静默降级到 BM25
    # vector_required=true  → Chroma 不可用时抛出异常
    knowledge_vector_enabled: bool = True
    knowledge_vector_required: bool = False

    # ── Chroma 向量库配置 ──────────────────────────────────────────
    chroma_persist_dir: str = "data/chroma"          # Chroma 本地持久化目录
    chroma_collection_name: str = "mindbridge_knowledge"  # Chroma collection 名称
    chroma_snapshot_dir: str = "data/chroma-snapshots"    # 快照备份目录
    chroma_snapshot_keep: int = 5                    # 保留最近 N 个快照
    embedding_timeout_seconds: float = 30.0          # embedding API 超时

    # ── RAG 评测配置 ──────────────────────────────────────────────
    rag_eval_dataset: str = "app/rag_eval/mindbridge-rag-eval.json"
    rag_eval_output: str = "target/rag-eval-report.json"
    rag_eval_enabled: bool = False
    rag_eval_exit_after_run: bool = False

    # ── Excel 台账配置 ────────────────────────────────────────────
    excel_path: str = "data/mindbridge-risk-ledger.xlsx"

    # ── Redis 短期记忆配置 ────────────────────────────────────────
    # Redis 仅存储每个会话最近 N 条消息作为短期上下文
    # 完整聊天记录始终写入 MySQL
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_memory_ttl_seconds: int = 86400    # 短期记忆过期时间（秒），默认 24 小时
    redis_memory_max_messages: int = 40      # 每个会话最多保留消息数
    redis_socket_timeout_seconds: float = 2.0

    # 历史压缩：MemoryAgent 注入 prompt 前，把较早的消息折叠为一条系统摘要
    # 只保留最近 N 条原始消息，避免上下文过长；摘要文本同时用于 memory_brief
    memory_compaction_enabled: bool = True
    memory_compaction_recent_messages: int = 8
    memory_summary_max_chars: int = 500

    # ── SMTP 邮件预警配置 ─────────────────────────────────────────
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_timeout_seconds: float = 10.0

    # 预警投递模式：
    #   "log"  → 只写数据库日志，不发邮件（本地演示用）
    #   "smtp" → 通过 SMTP 真实发送邮件
    alert_email_delivery_mode: str = "log"
    alert_email_from: str = ""
    alert_email_to: str = ""    # 多个收件人用逗号分隔
    alert_email_subject_prefix: str = "[MindBridge 高风险预警]"

    # ── 异步工具队列配置 ──────────────────────────────────────────
    # 工具队列用于异步执行 Excel 写入、个案创建、邮件预警等后处理任务
    # 不阻塞学生端的 SSE 流式回复
    tool_queue_enabled: bool = True
    tool_queue_poll_interval_seconds: float = 1.0   # 调度器轮询间隔
    tool_queue_batch_size: int = 10                 # 每次最多取多少个待处理任务
    tool_queue_max_attempts: int = 3                # 最大重试次数，超过后进入死信
    tool_queue_retry_delay_seconds: float = 15.0    # 重试延迟基数（乘以 attempts 次数）
    tool_queue_excel_workers: int = 1               # Excel 写入线程数（进程内锁串行化）
    tool_queue_email_workers: int = 2               # 邮件发送线程数
    alert_email_rate_limit_per_minute: int = 30     # 每分钟最多发送预警邮件数

    # pydantic-settings 配置：从 .env 文件加载，忽略未知字段
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def project_root(self) -> Path:
        """返回项目根目录（app/core/ 的上两级）"""
        return Path(__file__).resolve().parents[2]


@lru_cache
def get_settings() -> Settings:
    """
    获取全局配置单例。

    使用 lru_cache 保证整个应用生命周期内只创建一次 Settings 实例。
    harness 测试时可通过 cache_clear() 重新加载配置。
    """
    return Settings()
