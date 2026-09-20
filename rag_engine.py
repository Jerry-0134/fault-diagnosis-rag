"""RAG engine with hybrid retrieval (keyword + vector + rerank)."""

from cache_manager import cache
from monitor import timer
import os
import requests
import json
from config import DEEPSEEK_API_KEY

# 强制离线模式，使用本地缓存
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from hybrid_retriever import HybridRetriever

# 全局单例，避免重复加载
_retriever = None

def get_retriever():
    """获取混合检索器单例"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(vector_weight=0.7)
    return _retriever


@timer
def query(question: str, top_k: int = 5):
    """混合检索 + LLM生成答案（带缓存）"""
    
    # 1. 检查缓存
    cached_result = cache.get(question)
    if cached_result:
        print(f"📦 缓存命中，跳过检索和LLM")
        return cached_result['answer'], cached_result['hits']
    
    # 2. 执行检索（原有逻辑）
    retriever = get_retriever()
    results = retriever.search(question, top_k=top_k)
    
    # 构建 hits（保持原有格式）
    hits = []
    for r in results:
        case = r['case']
        content = f"设备：{case.get('设备名称', '未知')}\n故障：{case.get('故障现象', '未知')}"
        hit = {
            'content': content,
            '设备名称': case.get('设备名称', '未知'),
            '故障现象': case.get('故障现象', '未知'),
            'score': r['score'],
            'vector_score': r.get('vector_score', 0),
            'keyword_score': r.get('keyword_score', 0),
        }
        hits.append(hit)
    
    # 3. 生成回答
    if not hits:
        answer = generate_fallback_answer(question)
    else:
        prompt = generate_prompt(question, hits)
        answer = call_deepseek_api(prompt)
    
    # 4. 存入缓存
    cache.set(question, {'answer': answer, 'hits': hits})
    
    return answer, hits

def generate_prompt(question: str, hits: list) -> str:
    """生成 LLM prompt"""
    # 构建上下文
    context = "以下是检索到的相关故障案例（按相关度排序）：\n\n"
    for i, hit in enumerate(hits, 1):
        context += f"【案例{i}】相关度: {hit['score']:.3f}\n"
        context += f"设备：{hit.get('设备名称', '未知')}\n"
        context += f"故障现象：{hit.get('故障现象', '未知')}\n"
        context += f"详细内容：\n{hit['content']}\n\n"
        context += "-" * 40 + "\n\n"
    
    prompt = f"""你是一个专业的工业设备故障诊断专家。请根据以下参考案例，回答用户的问题。

{context}

用户问题：{question}

要求：
1. 如果参考案例与问题高度相关（相关度 > 0.6），请结合案例给出具体的诊断建议
2. 如果参考案例相关度一般（0.3-0.6），请说明参考价值有限，结合专业知识给出通用排查思路
3. 如果参考案例相关度较低（< 0.3），请说明未找到高度匹配的案例，给出通用故障排查指南
4. 回答要专业、具体、可操作
5. 建议用户补充必要信息（设备型号、故障代码、具体现象等）

请用中文回答："""
    
    return prompt


def generate_fallback_answer(question: str) -> str:
    """无检索结果时的降级回答"""
    return f"""未在知识库中找到与您问题直接相关的故障案例。

您的问题：{question}

建议：
1. 请提供更详细的故障信息，如设备型号、故障代码、具体现象
2. 检查设备供电、连接线缆等基础问题
3. 查阅设备手册中的故障排除章节
4. 如需进一步帮助，请补充信息后重新提问

当前知识库涵盖：变频器、PLC、电机、传感器、CNC系统、液压气动系统等工业设备。"""


def call_deepseek_api(prompt: str) -> str:
    """调用 DeepSeek API"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是工业设备故障诊断专家，回答专业、准确、实用。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"API调用失败（状态码：{response.status_code}）。请检查网络或API Key。"
            
    except requests.exceptions.Timeout:
        return "API调用超时。请检查网络连接后重试。"
    except Exception as e:
        return f"API调用出错：{str(e)}"


# 测试代码
if __name__ == "__main__":
    # 测试查询
    test_questions = [
        "电机振动很大",
        "变频器过流怎么办",
    ]
    
    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"问题：{q}")
        print('='*50)
        answer, hits = query(q)
        print(f"\n检索到 {len(hits)} 条案例")
        for i, h in enumerate(hits, 1):
            print(f"{i}. {h['设备名称']} (综合分: {h['score']:.3f})")
        print(f"\n回答：\n{answer[:200]}...")
