"""集中配置管理。

所有配置项在此定义，业务代码通过 ``get_settings()`` 读取。
禁止在业务模块中散落硬编码配置。
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置（读取 .env / 环境变量）。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------- 应用 ----------
    app_env: str = "development"
    app_name: str = "TeleComOps Agent"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # ---------- LLM（DeepSeek） ----------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    # deepseek-chat：稳定别名，支持 JSON mode / 结构化输出。
    # deepseek-reasoner 不支持 function calling 与 JSON 输出，勿用于 Agent 主链路。
    deepseek_model: str = "deepseek-chat"
    mock_llm: bool = False
    llm_timeout: int = 30
    llm_max_retries: int = 3

    # ---------- Embedding / Reranker（BGE 系列） ----------
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_device: str = "cpu"
    # 向量维度不从配置读取，而是从模型配置动态获取，避免手写错误维度。

    # ---------- PostgreSQL ----------
    postgres_url: str = "postgresql+psycopg://telecomops:telecomops@localhost:5433/telecomops"

    # ---------- Milvus ----------
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "knowledge_base"
    milvus_index_type: str = "IVF_FLAT"
    milvus_index_nlist: int = 128
    milvus_nprobe: int = 16
    milvus_metric_type: str = "COSINE"

    # ---------- RAG ----------
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    # 融合权重不是"最佳参数"，需通过 evaluation 数据调优（见 docs/README 评测章节）。
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    min_relevance_score: float = 0.35
    max_context_chars: int = 4000

    # ---------- Workflow 循环边界 ----------
    max_retries: int = 2
    max_replans: int = 1
    max_tool_executions: int = 10
    validator_use_llm: bool = True

    # ---------- Tool ----------
    tool_timeout: int = 5
    max_rows: int = 20
    max_cell_length: int = 200

    # ---------- Guardrails ----------
    max_input_length: int = 2000

    # ---------- Observability ----------
    otel_enabled: bool = True
    otel_service_name: str = "telecomops-agent"
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例。"""
    return Settings()


settings = get_settings()
