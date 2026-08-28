"""知识库导入入口。

用法：uv run python scripts/ingest_knowledge.py
"""

import sys

from app.core.config import settings
from app.core.logging import setup_logging
from app.rag.ingestion import ingest

setup_logging(settings.log_level)


def main() -> int:
    stats = ingest()
    print(f"导入完成：{stats['total']} 条文档（embedding 维度 {stats.get('dimension')}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
