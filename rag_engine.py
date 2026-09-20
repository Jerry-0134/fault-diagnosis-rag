# -*- coding: utf-8 -*-
"""RAG engine with hybrid retrieval, TTL caching and response-time monitoring."""

import os

import requests

from cache_manager import cache
from config import DEEPSEEK_API_KEY, TOP_K
from monitor import timer


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from hybrid_retriever import HybridRetriever


_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(vector_weight=0.7)
    return _retriever


def _build_hit(result: dict) -> dict:
    case = result["case"]
    content = case.get("完整内容", "").strip()
    if not content:
        content = "\n".join(
            [
                f"设备名称：{case.get('设备名称', '')}",
                f"故障代码：{case.get('故障代码', '')}",
                f"故障现象：{case.get('故障现象', '')}",
                f"可能原因：{case.get('可能原因', '')}",
                f"解决步骤：{case.get('解决步骤', '')}",
                f"紧急程度：{case.get('紧急程度', '')}",
                f"参考参数：{case.get('参考参数', '')}",
                f"数据来源：{case.get('数据来源', '')}",
            ]
        )

    return {
        "案例ID": case.get("案例ID", ""),
        "content": content,
        "设备名称": case.get("设备名称", "未知"),
        "故障代码": case.get("故障代码", ""),
        "故障现象": case.get("故障现象", "未知"),
        "紧急程度": case.get("紧急程度", ""),
        "数据来源": case.get("数据来源", ""),
        "score": float(result["score"]),
        "vector_score": float(result.get("vector_score", 0)),
        "keyword_score": float(result.get("keyword_score", 0)),
    }


@timer
def query(question: str, top_k: int = TOP_K):
    question = " ".join(question.strip().split())
    if not question:
        return "请输入具体的设备名称、故障代码或故障现象。", []

    cached_result = cache.get(question)
    if cached_result:
        return cached_result["answer"], cached_result["hits"]

    results = get_retriever().search(question, top_k=top_k)
    hits = [_build_hit(result) for result in results]

    if not hits:
        answer = generate_fallback_answer(question)
    elif not DEEPSEEK_API_KEY:
        answer = (
            "已检索到相关案例，但当前未配置 DeepSeek API Key。"
            "请在本地 .env 文件中设置 DEEPSEEK_API_KEY 后重试。"
        )
    else:
        answer = call_deepseek_api(generate_prompt(question, hits))

    cache.set(question, {"answer": answer, "hits": hits})
    return answer, hits


def generate_prompt(question: str, hits: list) -> str:
    context_parts = []
    for index, hit in enumerate(hits, start=1):
        context_parts.append(
            f"【案例{index}】综合相关度：{hit['score']:.3f}\n"
            f"{hit['content']}"
        )
    context = "\n\n".join(context_parts)

    return f"""你是工业设备故障诊断辅助系统。请严格依据参考案例回答，不得编造案例中不存在的型号、阈值或操作结果。

参考案例：
{context}

用户描述：{question}

输出要求：
1. 先概括最可能的故障方向，再按安全优先级给出排查步骤。
2. 具体参数必须与对应设备型号和参考案例绑定，不得推广为所有设备的通用值。
3. 证据不足时明确说明不确定，并列出需要补充的设备型号、报警代码、运行工况和近期变更。
4. 涉及断电、拆线、参数修改或机械操作时，先提示由具备权限的人员遵守现场安全规程。
5. 结尾列出采用的案例编号与数据来源。

请使用中文回答。"""


def generate_fallback_answer(question: str) -> str:
    return f"""知识库中没有找到足够相关的故障案例。

当前描述：{question}

请补充设备型号、故障代码、具体现象、运行工况和近期改动。在确认安全状态后，可先检查供电、连接线缆、机械负载和基础参数。高风险操作应由具备权限的人员按照设备手册和现场安全规程执行。"""


def call_deepseek_api(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是谨慎的工业设备故障诊断助手，必须优先保证安全并依据提供的案例作答。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1000,
    }

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "AI服务响应超时，请稍后重试。"
    except requests.exceptions.RequestException:
        return "AI服务暂时不可用，请检查网络和API配置后重试。"
    except (KeyError, ValueError, TypeError):
        return "AI服务返回格式异常，请稍后重试。"


if __name__ == "__main__":
    for sample in ("电机振动很大", "变频器过流"):
        answer, retrieved = query(sample)
        print(sample, len(retrieved), answer[:200])
