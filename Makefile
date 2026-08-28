# TeleComOps Agent 常用命令（Linux/macOS/CI；Windows 用户见 README 的 PowerShell 等价命令）
.PHONY: setup infra-up infra-down app run test test-integration lint typecheck eval-rag eval-agent ingest init-db

setup:
	uv sync

infra-up:
	docker compose up -d postgres etcd minio milvus prometheus grafana otel-collector

infra-down:
	docker compose down

# 完整项目（含 app 容器）
up:
	docker compose up -d

app:
	uv run uvicorn app.main:app --reload

run: app

init-db:
	uv run python scripts/init_db.py

ingest:
	uv run python scripts/ingest_knowledge.py

test:
	uv run pytest tests/unit -q

test-integration:
	uv run pytest tests/integration tests/workflow tests/api -q

test-all:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy app scripts

eval-rag:
	uv run python scripts/evaluate_rag.py

eval-agent:
	uv run python scripts/evaluate_agent.py
