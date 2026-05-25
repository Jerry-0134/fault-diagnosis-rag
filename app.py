"""Streamlit chat interface for the RAG fault diagnosis system with multimodal support."""

import os
import streamlit as st
from config import DATA_FILE, TOP_K
from data_loader import parse_fault_cases, get_case_title
from vector_store import add_fault_cases, get_collection
from rag_engine import query
from monitor import get_stats, response_times
from cache_manager import cache
from ocr_simulator import OCREngine
from vlm_simulator import get_vlm_engine
import tempfile
from pathlib import Path

st.set_page_config(page_title="工业故障诊断助手", page_icon="🔧", layout="wide")

# --- Init vector store (必须在最前面) ---
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

# 立即执行初始化
init_ok, init_msg = init_vector_store()

# --- Sidebar ---
with st.sidebar:
    st.title("🔧 故障诊断助手")
    st.markdown("基于 RAG 的工业故障智能问答系统")

    st.divider()
    
    # ========== 性能监控面板 ==========
    st.subheader("⚡ 性能监控")
    
    stats = get_stats()
    if stats and len(response_times) > 0:
        col1, col2 = st.columns(2)
        col1.metric("平均响应", f"{stats['avg_ms']:.0f}ms")
        col2.metric("P95响应", f"{stats['p95_ms']:.0f}ms")
        
        cache_stats = cache.stats()
        st.caption(f"📦 缓存命中率: {cache_stats['hit_rate']}")
        st.caption(f"💾 缓存条目: {cache_stats['total_entries']}")
        st.caption(f"📊 总请求数: {stats['total']}")
    else:
        st.info("等待首次请求...")
    # ==================================
    
    st.divider()

    # Data status
    st.subheader("数据状态")
    if os.path.exists(DATA_FILE):
        st.success(f"✅ data.txt 已找到")
    else:
        st.error(f"❌ 未找到 data.txt")
        st.code(f"请将 data.txt 放入:\n{os.path.dirname(DATA_FILE)}")

    # Vector store status
    if init_ok:
        collection = get_collection()
        doc_count = collection.count()
        if doc_count > 0:
            st.success(f"✅ 向量库已就绪 ({doc_count} 条)")
        else:
            st.warning("⚠️ 向量库为空，将从 data.txt 导入")
    else:
        st.error(f"❌ 向量库初始化失败: {init_msg}")

    st.divider()
    
    # ========== 多模态图片上传 ==========
    st.subheader("📤 上传设备图片")
    st.caption("支持从文件名识别故障关键词")
    
    uploaded_file = st.file_uploader(
        "选择图片 (JPG/PNG)",
        type=['jpg', 'jpeg', 'png'],
        key="multimodal_upload"
    )
    
    if uploaded_file:
        # 保存临时文件
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / uploaded_file.name
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.image_path = str(temp_path)
        st.image(uploaded_file, caption="已上传图片", use_container_width=True)
        st.caption(f"📄 文件名: {uploaded_file.name}")
        
        # 分析图片按钮
        if st.button("🔍 分析图片并诊断", key="analyze_btn", type="primary"):
            with st.spinner("正在分析图片..."):
                ocr = OCREngine()
                vlm = get_vlm_engine()
                
                ocr_text = ocr.extract_text(str(temp_path))
                vlm_result = vlm.analyze_device_photo(str(temp_path))
                
                st.session_state.ocr_result = ocr_text
                st.session_state.vlm_result = vlm_result
                
                # 构建图片分析查询
                fault_type = vlm_result.get('fault_type', [])
                if fault_type and fault_type != ['normal']:
                    image_query = f"图片分析检测到故障: {', '.join(fault_type)}。请给出诊断建议。"
                else:
                    image_query = "请根据图片信息给出一般性设备检查建议。"
                
                # 调用诊断
                with st.spinner("正在生成诊断建议..."):
                    answer, hits = query(image_query)
                    
                    # 保存到对话历史
                    st.session_state.messages.append({"role": "user", "content": f"📷 [图片分析] {image_query}"})
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits})
                    
                    # 强制刷新页面显示新消息
                    st.rerun()
    # ==================================
    
    st.divider()
    
    st.caption(f"🔍 检索数量: Top-{TOP_K} | 🤖 模型: deepseek-chat")

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

    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回复
    with st.chat_message("assistant"):
        with st.spinner("正在检索相关案例并生成回答..."):
            
            answer, hits = query(prompt)
            
            st.markdown(answer)
            
            if hits:
                with st.expander("📚 查看参考案例"):
                    for hit in hits:
                        title = get_case_title(hit)
                        st.caption(f"**{title}** (匹配度: {1 - hit.get('distance', 0):.2f})")
                        st.text(hit.get("content", ""))
                
                with st.expander("🔧 调试信息（混合检索详情）", expanded=False):
                    st.markdown("**混合检索结果详情：**")
                    for i, hit in enumerate(hits, 1):
                        score = hit.get('score', 1 - hit.get('distance', 0))
                        st.markdown(f"""
                        **{i}. {get_case_title(hit)}** (综合分: {score:.3f})
                        - 设备: {hit.get('设备名称', '未知')}
                        - 故障: {hit.get('故障现象', '未知')[:80]}
                        - 向量分: {hit.get('vector_score', 0):.3f} | 关键词分: {hit.get('keyword_score', 0):.3f}
                        """)
                        st.divider()

    # 保存助手消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": hits,
    })
