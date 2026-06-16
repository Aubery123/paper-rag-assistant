"""Streamlit 聊天界面（带引用展示 + 多篇对比）。

运行：streamlit run app.py
依赖知识库已入库（python main.py ingest ./data/papers/）。
"""

import streamlit as st

from src.config import settings
from src.generate.answerer import answer_stream
from src.retrieve.store import VectorStore

st.set_page_config(page_title="论文研读 RAG 助手", page_icon="📚", layout="wide")


@st.cache_data(show_spinner=False)
def list_papers() -> list[str]:
    """从向量库取所有 paper_id（用于多篇对比筛选）。"""
    try:
        payloads = VectorStore().scroll_all()
        return sorted({p.get("paper_id") for p in payloads if p.get("paper_id")})
    except Exception:
        return []


def render_sources(sources: list) -> None:
    """渲染引用来源清单。"""
    if not sources:
        return
    with st.expander(f"📎 引用来源（{len(sources)} 条）", expanded=True):
        for s in sources:
            p = s.payload
            st.markdown(
                f"**[{s._idx}]** 《{p.get('title','')}》 · "
                f"{p.get('section','') or '正文'} · 第 {p.get('page','?')} 页 "
                f"· `score={s.score:.3f}`"
            )


# ---- 侧边栏：检索设置 ----
st.sidebar.title("⚙️ 检索设置")
papers = list_papers()
selected = st.sidebar.multiselect(
    "限定论文（多篇对比 / 留空=全部）", options=papers, default=[]
)
mode = st.sidebar.radio("检索方式", ["hybrid", "dense"], index=0)
use_rerank = st.sidebar.checkbox("启用重排", value=settings.use_rerank)
st.sidebar.caption(f"知识库：{len(papers)} 篇论文")

# ---- 主区：聊天 ----
st.title("📚 论文研读 RAG 助手")
st.caption("带精确引用溯源（哪篇 / 哪节 / 哪页）的论文问答")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 回放历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

if prompt := st.chat_input("就这些论文提问，例如：FLASH 在服务器端如何做漂移感知优化？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream, hits = answer_stream(
            prompt, paper_ids=selected or None, mode=mode, use_rerank=use_rerank
        )
        # 给每条来源附上展示序号
        for i, h in enumerate(hits, start=1):
            h._idx = i
        answer_text = st.write_stream(stream)
        render_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": hits}
    )
