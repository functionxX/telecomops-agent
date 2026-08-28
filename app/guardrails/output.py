"""Output Guardrail：敏感数据脱敏。

对 LLM 输出做最后一道检查：手机号、身份证号、银行卡号等
个人敏感信息脱敏后再返回客户端。宁可多脱敏，不可漏脱敏。
"""

import re

_PATTERNS = [
    # 手机号（1 开头 11 位）
    (re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)"), r"\1****\3"),
    # 身份证号（18 位）
    (
        re.compile(r"(?<!\d)(\d{6})(\d{8})(\d{3}[\dXx])(?!\d)"),
        r"\1********\3",
    ),
    # 银行卡号（16-19 位，宽松匹配）
    (re.compile(r"(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)"), r"\1********\2"),
]


def mask_sensitive(text: str) -> str:
    """把文本中的敏感信息替换为掩码形式。"""
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def contains_sensitive(text: str) -> bool:
    """检查是否包含敏感信息（用于日志前判断）。"""
    return any(p.search(text) for p, _ in _PATTERNS)
