"""P0 冒烟测试：确认配置可正常加载，给 CI 一个最小可跑的测试。"""

from src.config import settings


def test_settings_defaults():
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.llm_model == "qwen-plus"


def test_retrieve_params_sane():
    # 重排后数量不应超过单路召回数量
    assert settings.rerank_top_k <= settings.retrieve_top_k
    assert settings.chunk_overlap < settings.chunk_size
