"""结构化日志配置。

JSON 格式输出（容器内可被 Loki/EFK 等采集），每条日志自动附带 trace_id。
注意：绝不记录 API Key、密码、完整敏感用户信息。
"""

import json
import logging
import sys
from datetime import UTC, datetime

from app.observability.tracing import get_trace_id


class JsonFormatter(logging.Formatter):
    """把日志记录格式化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_trace_id(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """初始化根日志器（幂等）。"""
    root = logging.getLogger()
    if root.handlers:  # 已配置（如 uvicorn 预配置）
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
