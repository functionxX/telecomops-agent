"""Input Guardrail：异常输入检测。

检测手段（确定性为主，不额外调用 LLM）：
- 超长输入（MAX_INPUT_LENGTH）
- Prompt Injection 特征：越狱指令、角色覆盖、系统提示泄露请求
- 控制字符/可疑编码

注意：正则检测不可能拦截所有注入，它是第一道廉价防线；
真正的防线是「LLM 只做计划与输出、所有工具调用经 ToolExecutor
的程序级校验」（见 app/tools/executor.py 与 ADR-007）。
"""

import re

from app.core.config import settings

# 常见注入特征（中文 + 英文）
INJECTION_PATTERNS = [
    r"忽略(上述|之前|上面|以下)?(所有)?(指令|规则|提示)",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?)",
    r"你现在是.{0,20}(另一个|新的).{0,10}(角色|AI)",
    r"system\s*prompt",
    r"输出.{0,10}(完整)?(系统)?提示词",
    r"jailbreak",
    r"开发者模式",
]

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class InputGuardResult:
    def __init__(self, ok: bool, reason: str = "") -> None:
        self.ok = ok
        self.reason = reason


def check_input(text: str) -> InputGuardResult:
    """输入检查：超长 / 注入特征 / 控制字符。"""
    if not text or not text.strip():
        return InputGuardResult(False, "输入为空")
    if len(text) > settings.max_input_length:
        return InputGuardResult(
            False, f"输入超长（{len(text)} > {settings.max_input_length}）"
        )
    if CONTROL_CHARS.search(text):
        return InputGuardResult(False, "输入包含控制字符")
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return InputGuardResult(False, "输入包含疑似注入内容")
    return InputGuardResult(True)
