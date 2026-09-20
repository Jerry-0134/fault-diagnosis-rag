# -*- coding: utf-8 -*-
"""Shared parser for the structured industrial fault-case knowledge base."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List


KNOWN_FIELDS = (
    "设备名称",
    "故障代码",
    "故障现象",
    "现场描述",
    "可能原因",
    "解决步骤",
    "紧急程度",
    "参考参数",
    "数据来源",
    "案例来源",
)


def load_fault_cases(file_path: str | Path) -> List[Dict[str, str]]:
    """Parse blank-line separated cases while retaining multiline fields."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    raw_cases = [block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]

    cases: List[Dict[str, str]] = []
    for index, raw in enumerate(raw_cases, start=1):
        buffers = {field: [] for field in KNOWN_FIELDS}
        current_field = None

        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            match = re.match(r"^([^：:]+)[：:]\s*(.*)$", line)
            if match and match.group(1).strip() in KNOWN_FIELDS:
                current_field = match.group(1).strip()
                value = match.group(2).strip()
                if value:
                    buffers[current_field].append(value)
                continue

            if current_field:
                buffers[current_field].append(line)

        case = {field: "\n".join(buffers[field]).strip() for field in KNOWN_FIELDS}
        if not case["设备名称"]:
            continue

        case["案例ID"] = f"case_{index:03d}"
        case["完整内容"] = raw
        if not case["数据来源"] and case["案例来源"]:
            case["数据来源"] = case["案例来源"]
        cases.append(case)

    return cases
