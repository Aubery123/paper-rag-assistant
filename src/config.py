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

    # ---- Embedding / Reranker（阿里百炼 API，复用 dashscope key + base_url）----
    embedding_model: str = Field(default="text-embedding-v3")
    embedding_dim: int = Field(default=1024)   # text-embedding-v3 默认维度
    embedding_batch: int = Field(default=10)   # 百炼兼容接口单次最多 10 条
    reranker_model: str = Field(default="gte-rerank-v2")  # 百炼重排 API（P2 用）

    # ---- Qdrant ----
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="papers")

    # ---- 切分 chunking ----
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=120)

    # ---- 检索 hybrid → rerank ----
    retrieve_mode: str = Field(default="hybrid")  # dense | hybrid（P3 消融用）
    use_rerank: bool = Field(default=True)        # 是否重排（P3 消融用）
    retrieve_top_k: int = Field(default=20)   # 每路召回 / 送入重排的候选数
    rrf_k: int = Field(default=60)            # RRF 融合常数
    rerank_top_k: int = Field(default=5)      # 重排后送入 LLM 的数量
    rerank_url: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )


settings = Settings()
