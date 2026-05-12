from config import TOP_K
from vector_store import search_similar
from openai import OpenAI

# 直接写死 API Key（请替换成你的完整 Key）
import os
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

def build_prompt(question: str, contexts: list) -> str:
    """构建提示词"""
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        case = ctx['case']
        context_text += f"\n--- 案例 {i} ---\n"
        context_text += f"设备名称：{case.get('设备名称', '')}\n"
        context_text += f"故障现象：{case.get('故障现象', '')}\n"
        context_text += f"可能原因：{case.get('可能原因', '')}\n"
        context_text += f"解决步骤：{case.get('解决步骤', '')}\n"
        if case.get('参考参数'):
            context_text += f"参考参数：{case.get('参考参数')}\n"
    
    prompt = f"""你是一个专业的工业设备故障诊断专家。请根据以下参考案例，回答用户的问题。

参考案例：
{context_text}

用户问题：{question}

请回答：
1. 如果参考案例中有相关信息，请基于案例给出具体的故障原因和解决步骤
2. 如果信息不足，请说明需要进一步检查哪些方面
3. 回答要专业、清晰、可操作
"""
    return prompt

def query(question: str):
    """返回 (回答文本, 检索到的案例列表)"""
    # 1. 检索相似案例
    similar_cases = search_similar(question, TOP_K)
    
    if not similar_cases or len(similar_cases) == 0:
        return "抱歉，知识库中没有找到相关的故障案例。请检查设备或咨询专业工程师。", []
    
    # 2. 构建提示词
    prompt = build_prompt(question, similar_cases)
    
    # 3. 调用 DeepSeek API
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content
        return answer, similar_cases
    except Exception as e:
        return f"调用 AI 服务时出错：{str(e)}", []