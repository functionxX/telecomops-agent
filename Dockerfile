# TeleComOps Agent 应用镜像
# 依赖 uv.lock（本机先执行 uv sync 生成）。模型在容器首次启动时从 HuggingFace 下载。

FROM ghcr.io/astral-sh/uv:0.12.5 AS uv

FROM python:3.11-slim

WORKDIR /app

# 安装 uv 二进制
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# 先装依赖（利用 layer cache；README.md 是 hatchling 构建的必需文件）
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# 应用代码
COPY app ./app
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
