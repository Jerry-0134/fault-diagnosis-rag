import re
from typing import List, Dict, Any

# 全局变量
_cases = []
_initialized = False

def add_fault_cases(cases: List[Dict[str, Any]]):
    """添加故障案例"""
    global _cases, _initialized
    _cases = cases
    _initialized = True
    print(f"已加载 {len(cases)} 个故障案例")

def search_similar(query_text: str, top_k: int = 2):
    """用关键词匹配搜索最相似的故障案例"""
    global _cases, _initialized
    
    if not _initialized or len(_cases) == 0:
        return []
    
    # 提取查询中的关键词
    query_keywords = set()
    # 常见故障词
    keywords_list = ['电机', '振动', '温度', 'PLC', '变频器', '传感器', '过电流', '短路', '绝缘', 
                     '输出', '信号', '连接', '电源', '程序', '轴承', '螺栓', '负载', '校准', '参数']
    
    for kw in keywords_list:
        if kw in query_text:
            query_keywords.add(kw)
    
    # 如果没有匹配到关键词，用原始查询词
    if not query_keywords:
        query_keywords = set(query_text)
    
    # 计算每个案例的匹配分数
    scores = []
    for idx, case in enumerate(_cases):
        score = 0
        case_text = f"{case.get('设备名称', '')} {case.get('故障现象', '')} {case.get('可能原因', '')}"
        
        for kw in query_keywords:
            if kw in case_text:
                score += 1
        
        # 额外加分：故障现象完全匹配
        if case.get('故障现象', '') in query_text:
            score += 3
        
        scores.append((idx, score))
    
    # 按分数排序
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # 返回 top-k
    results = []
    for idx, score in scores[:top_k]:
        if score > 0:
            results.append({
                'case': _cases[idx],
                'similarity': score / max(1, len(query_keywords)),
                'index': idx
            })
    
    return results

def get_collection():
    return MockCollection()

class MockCollection:
    def query(self, query_texts: List[str], n_results: int = 2):
        global _cases, _initialized
        
        if not _initialized:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        
        results = search_similar(query_texts[0], n_results)
        
        ids = [[str(r['index']) for r in results]]
        documents = [[r['case'] for r in results]]
        metadatas = [[{"title": r['case'].get('故障现象', '')} for r in results]]
        
        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas
        }
    
    def count(self):
        global _cases
        return len(_cases)