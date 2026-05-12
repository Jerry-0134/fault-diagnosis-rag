"""Streamlit chat interface for the RAG fault diagnosis system."""

import os
import streamlit as st
from config import DATA_FILE, TOP_K
from data_loader import parse_fault_cases, get_case_title
from vector_store import add_fault_cases, get_collection
from rag_engine import query

st.set_page_config(page_title="工业故障诊断助手", page_icon="🔧", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("🔧 故障诊断助手")
    st.markdown("基于 RAG 的工业故障智能问答系统")

    st.divider()

    # Data status
    st.subheader("数据状态")
    if os.path.exists(DATA_FILE):
        st.success(f"✅ data.txt 已找到")
    else:
        st.error(f"❌ 未找到 data.txt")
        st.code(f"请将 data.txt 放入:\n{os.path.dirname(DATA_FILE)}")

    # Vector store status
    collection = get_collection()
    doc_count = collection.count()
    if doc_count > 0:
        st.success(f"✅ 向量库已就绪 ({doc_count} 条)")
    else:
        st.warning("⚠️ 向量库为空，将从 data.txt 导入")

    st.divider()

    # API key status (已跳过检查，API Key 已在 rag_engine.py 中写死)
# api_key = os.getenv("DEEPSEEK_API_KEY", "")
# if api_key:
#     st.success("✅ DeepSeek API Key 已设置")
# else:
#     st.error("❌ DeepSeek API Key 未设置")
#     st.caption("请在 .env 文件中添加 DEEPSEEK_API_KEY=your_key")

st.divider()
st.caption(f"检索数量: Top-{TOP_K} | 模型: deepseek-chat")


# --- Init vector store ---
@st.cache_resource
def init_vector_store():
    """Initialize vector store from data.txt (cached, runs once)."""
    if not os.path.exists(DATA_FILE):
        return False, "data.txt 文件不存在"

    cases = parse_fault_cases(DATA_FILE)
    if not cases:
        return False, "data.txt 中未解析到故障案例"

    count = add_fault_cases(cases)
    return True, f"已加载 {count} 条故障案例"


init_ok, init_msg = init_vector_store()

# --- Main UI ---
st.title("工业故障诊断 RAG 问答系统")
st.caption("输入故障现象或问题，系统将检索相关案例并结合 AI 生成诊断建议")

# Init message
if not init_ok:
    st.error(f"⚠️ 初始化失败: {init_msg}")
    st.stop()

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📋 查看参考案例"):
                for src in msg["sources"]:
                    title = get_case_title(src)
                    st.caption(f"**{title}** (匹配度: {1 - src.get('distance', 0):.2f})")
                    st.text(src.get("content", ""))

# Chat input
if prompt := st.chat_input("请输入故障现象或问题..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("正在检索相关案例并生成回答..."):
            answer, hits = query(prompt)

        st.markdown(answer)

        if hits:
            with st.expander("📋 查看参考案例"):
                for hit in hits:
                    title = get_case_title(hit)
                    st.caption(f"**{title}** (匹配度: {1 - hit.get('distance', 0):.2f})")
                    st.text(hit.get("content", ""))

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": hits,
    })
