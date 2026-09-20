# -*- coding: utf-8 -*-
"""Hybrid retrieval based on BGE semantic vectors and exact token overlap."""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from case_parser import load_fault_cases
from config import DATA_FILE, EMBEDDING_FILE


MODEL_NAME = "BAAI/bge-small-zh-v1.5"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _data_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _query_tokens(text: str) -> set[str]:
    """Keep Chinese characters and complete alphanumeric identifiers such as PLC or F0001."""
    return set(re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text.lower()))


class VectorRetriever:
    def __init__(self, embedding_path: str | Path = EMBEDDING_FILE):
        embedding_path = Path(embedding_path)
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"缺少向量文件 {embedding_path.name}，请先运行 python prepare_vectors.py"
            )

        with embedding_path.open("rb") as handle:
            data = pickle.load(handle)

        self.texts = data["texts"]
        self.cases = data["cases"]
        self.embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        self.model_name = data.get("model_name", MODEL_NAME)

        expected_hash = data.get("data_sha256")
        current_hash = _data_fingerprint(Path(DATA_FILE))
        if expected_hash and expected_hash != current_hash:
            raise RuntimeError("data.txt 已变化，请重新运行 python prepare_vectors.py")
        if len(self.cases) != len(self.embeddings):
            raise RuntimeError("案例数量与向量数量不一致，请重新生成向量")

        model_source = os.getenv("BGE_MODEL_PATH", self.model_name)
        self.model = SentenceTransformer(model_source)

    def search(self, query: str, top_k: int = 10):
        query_vec = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        similarities = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "case": self.cases[int(index)],
                "vector_score": float(similarities[index]),
                "index": int(index),
            }
            for index in top_indices
        ]


class KeywordRetriever:
    def __init__(self, data_path: str | Path = DATA_FILE):
        self.cases = load_fault_cases(data_path)

    def search(self, query: str, top_k: int = 10):
        query_tokens = _query_tokens(query)
        if not query_tokens:
            return []

        scored_cases = []
        for index, case in enumerate(self.cases):
            searchable = " ".join(
                [
                    case.get("设备名称", ""),
                    case.get("故障代码", ""),
                    case.get("故障现象", ""),
                ]
            ).lower()
            match_count = sum(token in searchable for token in query_tokens)
            keyword_score = match_count / len(query_tokens)
            if keyword_score > 0:
                scored_cases.append(
                    {
                        "case": case,
                        "keyword_score": float(keyword_score),
                        "index": index,
                    }
                )

        scored_cases.sort(key=lambda item: item["keyword_score"], reverse=True)
        return scored_cases[:top_k]


class HybridRetriever:
    def __init__(self, vector_weight: float = 0.7):
        if not 0 <= vector_weight <= 1:
            raise ValueError("vector_weight 必须位于 0 到 1 之间")

        self.vector_retriever = VectorRetriever()
        self.keyword_retriever = KeywordRetriever()
        self.vector_weight = vector_weight
        self.keyword_weight = 1 - vector_weight

    def search(self, query: str, top_k: int = 5):
        candidate_k = min(
            max(top_k * 4, top_k), len(self.vector_retriever.cases)
        )
        vector_results = self.vector_retriever.search(query, top_k=candidate_k)
        keyword_results = self.keyword_retriever.search(query, top_k=candidate_k)
        combined = {}

        for result in vector_results:
            index = result["index"]
            combined[index] = {
                "case": result["case"],
                "score": self.vector_weight * result["vector_score"],
                "vector_score": result["vector_score"],
                "keyword_score": 0.0,
            }

        for result in keyword_results:
            index = result["index"]
            keyword_score = result["keyword_score"]
            if index not in combined:
                combined[index] = {
                    "case": result["case"],
                    "score": 0.0,
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                }
            combined[index]["score"] += self.keyword_weight * keyword_score
            combined[index]["keyword_score"] = keyword_score

        return sorted(
            combined.values(), key=lambda item: item["score"], reverse=True
        )[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever(vector_weight=0.7)
    for sample in ("电机振动很大", "变频器过流", "PLC通讯中断"):
        print(f"\n查询：{sample}")
        for rank, result in enumerate(retriever.search(sample, top_k=3), start=1):
            case = result["case"]
            print(
                f"{rank}. {case.get('设备名称')} | {case.get('故障现象')} | "
                f"综合分 {result['score']:.3f}"
            )
