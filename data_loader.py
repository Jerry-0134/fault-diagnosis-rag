"""Parse data.txt into individual fault case chunks."""

import typing
import re
from typing import List, Dict


def parse_fault_cases(file_path: str) -> List[Dict[str, str]]:
    """
    Parse the structured fault cases from data.txt.
    Each case has: 设备名称, 故障现象, 可能原因, 解决步骤, 紧急程度, 参考参数(optional)
    Returns a list of dicts with 'content' and 'metadata' keys.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into individual cases by blank lines
    raw_cases = re.split(r"\n\s*\n", text.strip())

    cases = []
    for i, raw in enumerate(raw_cases):
        raw = raw.strip()
        if not raw:
            continue

        case = _parse_one_case(raw)
        if case is None:
            # If parsing fails, treat the whole block as plain text
            first_line = raw.split("\n")[0].strip()
            cases.append({
                "content": raw,
                "metadata": {"source": f"case_{i+1}", "title": first_line},
            })
        else:
            case["metadata"]["source"] = f"case_{i+1}"
            cases.append(case)

    return cases


def _parse_one_case(raw: str) -> typing.Optional[Dict[str, str]]:
    """
    Parse a single structured fault case.
    Returns {'content': ..., 'metadata': {...}} or None on failure.
    """
    metadata = {}

    # Extract 设备名称
    m = re.search(r"设备名称[：:]\s*(.+)", raw)
    if m:
        metadata["设备名称"] = m.group(1).strip()

    # Extract 故障现象
    m = re.search(r"故障现象[：:]\s*(.+)", raw)
    if m:
        metadata["故障现象"] = m.group(1).strip()

    # Extract 紧急程度
    m = re.search(r"紧急程度[：:]\s*(.+)", raw)
    if m:
        metadata["紧急程度"] = m.group(1).strip()

    # Extract 参考参数
    m = re.search(r"参考参数[：:]\s*(.+)", raw)
    if m:
        metadata["参考参数"] = m.group(1).strip()

    if not metadata:
        return None

    return {"content": raw, "metadata": metadata}


def get_case_title(case: Dict) -> str:
    """Generate a short display title for a case."""
    meta = case.get("metadata", {})
    device = meta.get("设备名称", "未知设备")
    symptom = meta.get("故障现象", "未知故障")
    return f"[{device}] {symptom}"
