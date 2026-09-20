# -*- coding: utf-8 -*-
"""Offline integrity checks that do not call DeepSeek or load the BGE model."""

import ast
import hashlib
import pickle
from pathlib import Path

from case_parser import load_fault_cases


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.txt"
EMBEDDING_FILE = BASE_DIR / "case_embeddings.pkl"


def main() -> None:
    cases = load_fault_cases(DATA_FILE)
    assert len(cases) == 50, f"预期50条案例，实际为{len(cases)}条"

    required_fields = ("设备名称", "故障现象", "可能原因", "解决步骤", "参考参数", "数据来源")
    for field in required_fields:
        missing = [case["案例ID"] for case in cases if not case.get(field)]
        assert not missing, f"字段 {field} 为空：{missing}"

    for path in BASE_DIR.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    config_text = (BASE_DIR / "config.py").read_text(encoding="utf-8")
    assert "sk-" not in config_text, "config.py 中仍存在硬编码密钥"

    assert EMBEDDING_FILE.exists(), "缺少 case_embeddings.pkl，请先运行 python prepare_vectors.py"
    with EMBEDDING_FILE.open("rb") as handle:
        payload = pickle.load(handle)

    current_hash = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    assert payload.get("data_sha256") == current_hash, "向量文件与当前 data.txt 不一致"
    assert len(payload.get("cases", [])) == 50, "向量文件案例数量不是50"
    assert len(payload.get("texts", [])) == 50, "向量文本数量不是50"
    assert all(case.get("可能原因") for case in payload["cases"]), "向量案例缺少可能原因"
    assert all(case.get("解决步骤") for case in payload["cases"]), "向量案例缺少解决步骤"

    embeddings = payload.get("embeddings")
    assert getattr(embeddings, "shape", None) == (50, 512), "向量矩阵应为50×512"

    print("项目完整性检查通过")
    print("案例数量：50")
    print("可能原因完整：50/50")
    print("解决步骤完整：50/50")
    print("向量矩阵：50×512")


if __name__ == "__main__":
    main()
