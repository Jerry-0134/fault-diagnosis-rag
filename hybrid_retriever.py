import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re

class VectorRetriever:
    """向量检索器"""
    def __init__(self, embedding_path='case_embeddings.pkl'):
        with open(embedding_path, 'rb') as f:
            data = pickle.load(f)
        self.texts = data['texts']
        self.cases = data['cases']
        self.embeddings = data['embeddings']
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        
    def search(self, query, top_k=10):
        query_vec = self.model.encode([query])[0]
        similarities = np.dot(self.embeddings, query_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'case': self.cases[idx],
                'vector_score': float(similarities[idx]),
                'index': idx
            })
        return results

class KeywordRetriever:
    """关键词检索器（基于你的原始逻辑）"""
    def __init__(self, data_path='data.txt'):
        self.cases = self._load_cases(data_path)
        
    def _load_cases(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        blocks = content.strip().split('\n\n')
        cases = []
        for block in blocks:
            if not block.strip():
                continue
            case = {}
            for line in block.strip().split('\n'):
                line = line.strip()
                if '：' in line:
                    key, value = line.split('：', 1)
                    case[key.strip()] = value.strip()
                elif ':' in line:
                    key, value = line.split(':', 1)
                    case[key.strip()] = value.strip()
            if case.get('设备名称'):
                cases.append(case)
        return cases
    
    def search(self, query, top_k=10):
        """简单关键词匹配（基于设备名称和故障现象）"""
        query_words = set(query.lower())
        scored_cases = []
        
        for idx, case in enumerate(self.cases):
            text = f"{case.get('设备名称', '')} {case.get('故障现象', '')}".lower()
            # 计算关键词匹配分数
            match_count = sum(1 for word in query_words if word in text)
            keyword_score = match_count / max(len(query_words), 1)
            
            if keyword_score > 0:
                scored_cases.append({
                    'case': case,
                    'keyword_score': keyword_score,
                    'index': idx
                })
        
        scored_cases.sort(key=lambda x: x['keyword_score'], reverse=True)
        return scored_cases[:top_k]

class HybridRetriever:
    """混合检索器（向量 + 关键词）"""
    def __init__(self, vector_weight=0.7):
        print("初始化混合检索器...")
        self.vector_retriever = VectorRetriever()
        self.keyword_retriever = KeywordRetriever()
        self.vector_weight = vector_weight
        self.keyword_weight = 1 - vector_weight
        print(f"权重配置：向量={vector_weight}，关键词={self.keyword_weight}")
        
    def search(self, query, top_k=5, rerank=False):
        """混合检索"""
        # 获取向量检索结果
        vector_results = self.vector_retriever.search(query, top_k=20)
        # 获取关键词检索结果
        keyword_results = self.keyword_retriever.search(query, top_k=20)
        
        # 融合分数（使用索引映射）
        combined = {}
        
        # 添加向量检索结果
        for r in vector_results:
            idx = r['index']
            combined[idx] = {
                'case': r['case'],
                'score': self.vector_weight * r['vector_score'],
                'vector_score': r['vector_score'],
                'keyword_score': 0
            }
        
        # 添加关键词检索结果
        for r in keyword_results:
            idx = r['index']
            keyword_score = r['keyword_score']
            if idx in combined:
                combined[idx]['score'] += self.keyword_weight * keyword_score
                combined[idx]['keyword_score'] = keyword_score
            else:
                combined[idx] = {
                    'case': r['case'],
                    'score': self.keyword_weight * keyword_score,
                    'vector_score': 0,
                    'keyword_score': keyword_score
                }
        
        # 排序并返回Top-K
        sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)[:top_k]
        return sorted_results

# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("混合检索器测试")
    print("="*60)
    
    retriever = HybridRetriever(vector_weight=0.7)
    
    test_queries = [
        "电机振动很大",
        "变频器过流怎么办",
        "PLC通讯中断"
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询：{query}")
        print("-"*50)
        results = retriever.search(query, top_k=3)
        
        for i, r in enumerate(results, 1):
            device_name = r['case'].get('设备名称', '未知')[:35]
            fault = r['case'].get('故障现象', '未知')[:35]
            print(f"{i}. 综合分:{r['score']:.3f} (向量:{r['vector_score']:.3f}, 关键词:{r['keyword_score']:.3f})")
            print(f"   → {device_name} | {fault}")
