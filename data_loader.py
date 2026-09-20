# -*- coding: utf-8 -*-
"""Compatibility helpers for loading and displaying structured fault cases."""

from typing import Dict, List

from case_parser import load_fault_cases


def parse_fault_cases(file_path: str) -> List[Dict]:
    cases = load_fault_cases(file_path)
    return [
        {
            "content": case["完整内容"],
            "metadata": {
                "source": case["案例ID"],
                "设备名称": case.get("设备名称", ""),
                "故障代码": case.get("故障代码", ""),
                "故障现象": case.get("故障现象", ""),
                "紧急程度": case.get("紧急程度", ""),
                "参考参数": case.get("参考参数", ""),
                "数据来源": case.get("数据来源", ""),
            },
        }
        for case in cases
    ]


def get_case_title(case: Dict) -> str:
    metadata = case.get("metadata", {})
    device = case.get("设备名称") or metadata.get("设备名称") or "未知设备"
    symptom = case.get("故障现象") or metadata.get("故障现象") or "未知故障"
    return f"[{device}] {symptom}"
