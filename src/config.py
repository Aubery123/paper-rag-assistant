"""集中配置：模型、阈值、路径、检索参数。

检索链路统一为 hybrid → rerank → answer，所有可调参数（k、阈值、权重）集中于此，
其他模块一律从这里取值，勿散落硬编码。

通过环境变量 / .env 覆盖（见 .env.example）。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（src/config.py → 上两级）
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"


class Settings(BaseSettings):
    """全局配置，可被环境变量 / .env 覆盖。"""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM（默认通义千问 qwen-plus，OpenAI 兼容接口）----
    dashscope_api_key: str = Field(default="", description="通义千问 API key")
    llm_model: str = Field(default="qwen-plus")
    llm_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    llm_temperature: float = Field(default=0.1)

    # ---- Embedding / Reranker（开源，本地）----
    embedding_model: str = Field(default="BAAI/bge-m3")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    embedding_dim: int = Field(default=1024)  # bge-m3 稠密维度

    # ---- Qdrant ----
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="papers")

    # ---- 切分 chunking ----
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=120)

    # ---- 检索 hybrid → rerank ----
    retrieve_top_k: int = Field(default=20)   # 每路召回数量
    rrf_k: int = Field(default=60)            # RRF 融合常数
    rerank_top_k: int = Field(default=5)      # 重排后送入 LLM 的数量


settings = Settings()
