# langgraph 的 Saver 协议参数类型是内部 RunnableConfig/Sequence 等，
# 与我们的 dict/list 签名在 mypy 层面存在系统性摩擦（运行时兼容）。
# 对本文件豁免 override/arg-type 两类错误码，业务逻辑不受影响。
# mypy: disable-error-code="override,arg-type"

"""自实现 PostgresCheckpointer：LangGraph Checkpoint 的 PostgreSQL 持久化。

为什么自实现（共识 Q8 / ADR-006）：
- 拥有自己的 workflow_checkpoints 表结构，中断/恢复机制可亲手解释；
- 序列化用官方 JsonPlusSerializer 的序列化协议，规避格式细节；
- 单元测试可用官方 MemorySaver，本实现只在集成路径生效。

checkpoint 协议要点：
- put：把 (thread_id, checkpoint_id) 的快照落库，parent 指向被取代的版本；
- put_writes：中断/并发任务产生的 pending writes 按 task_id 落库，
  resume 时 get_tuple 必须把它们还给框架重放；
- get_tuple：按 thread_id（+ 可选 checkpoint_id）取最新快照。
"""

import json
from collections.abc import Iterator
from typing import Any

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.config import settings


def _to_json(value: Any) -> str:
    """JSONB 写入：绑定 JSON 字符串（psycopg 对 text() 参数不做 dict→jsonb 推断）。"""
    return json.dumps(value, default=str, ensure_ascii=False)


def _ser_writes_value(serde: SerializerProtocol, value: Any) -> list[str]:
    """pending write 值用 serde 序列化（Interrupt 等对象走 msgpack 协议），
    以 [tag, base64(blob)] 存入 JSONB。"""
    import base64

    tag, blob = serde.dumps_typed(value)
    return [tag, base64.b64encode(blob).decode("ascii")]


def _deser_writes_value(serde: SerializerProtocol, encoded: list[str]) -> Any:
    import base64

    tag, b64 = encoded
    return serde.loads_typed((tag, base64.b64decode(b64)))


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workflow_checkpoints (
    thread_id            TEXT        NOT NULL,
    checkpoint_id        TEXT        NOT NULL,
    parent_checkpoint_id TEXT,
    checkpoint           JSONB       NOT NULL,
    metadata             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    pending_writes       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (thread_id, checkpoint_id)
);
CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_thread
    ON workflow_checkpoints (thread_id, created_at DESC);
"""


class PostgresCheckpointer(BaseCheckpointSaver):
    """workflow_checkpoints 表上的 CheckpointSaver。

    同步方法为真实实现；异步方法（astream 路径需要）通过
    asyncio.to_thread 委托给同步实现（底层 DB 驱动是同步的）。
    """

    def __init__(
        self,
        conn_string: str | None = None,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self._conn_string = conn_string or settings.postgres_url
        self._engine: Engine | None = None

    # ---------- 异步桥接（委托同步实现） ----------

    async def aget_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        import asyncio

        return await asyncio.to_thread(self.get_tuple, config)

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        import asyncio

        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def alist(
        self,
        config: dict[str, Any],
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Any:
        import asyncio

        def _sync() -> list[CheckpointTuple]:
            return list(self.list(config, filter=filter, before=before, limit=limit))

        return await asyncio.to_thread(_sync)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self._conn_string, pool_pre_ping=True)
            with self._engine.begin() as conn:
                conn.execute(text(_SCHEMA_SQL))
        return self._engine

    # ---------- 协议实现 ----------

    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        with self.engine.connect() as conn:
            if checkpoint_id:
                row = conn.execute(
                    text(
                        "SELECT thread_id, checkpoint_id, parent_checkpoint_id, checkpoint,"
                        " metadata, pending_writes FROM workflow_checkpoints"
                        " WHERE thread_id = :t AND checkpoint_id = :c"
                    ),
                    {"t": thread_id, "c": checkpoint_id},
                ).first()
            else:
                row = conn.execute(
                    text(
                        "SELECT thread_id, checkpoint_id, parent_checkpoint_id, checkpoint,"
                        " metadata, pending_writes FROM workflow_checkpoints"
                        " WHERE thread_id = :t ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"t": thread_id},
                ).first()
        if row is None:
            return None
        # JSONB 列由 SQLAlchemy 自动反序列化为 Python 对象，无需 json.loads
        checkpoint = row.checkpoint
        metadata = row.metadata or {}
        pending_writes = [
            (w[0], w[1], _deser_writes_value(self.serde, w[2])) for w in (row.pending_writes or [])
        ]
        parent_config = None
        if row.parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": row.parent_checkpoint_id,
                }
            }
        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_id": row.checkpoint_id}},
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        thread_id = config["configurable"]["thread_id"]
        parent_id = config["configurable"].get("checkpoint_id")
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO workflow_checkpoints"
                    " (thread_id, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)"
                    " VALUES (:t, :c, :p, :cp, :md)"
                    " ON CONFLICT (thread_id, checkpoint_id) DO UPDATE SET"
                    " checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata"
                ),
                {
                    "t": thread_id,
                    "c": checkpoint["id"],
                    "p": parent_id,
                    "cp": _to_json(checkpoint),
                    "md": _to_json(metadata),
                },
            )
        return {
            "configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint["id"]}
        }

    def put_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """pending writes 以 [task_id, channel, value] 三元组数组存储；同 task 覆盖。"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT pending_writes FROM workflow_checkpoints"
                    " WHERE thread_id = :t AND checkpoint_id = :c"
                ),
                {"t": thread_id, "c": checkpoint_id},
            ).first()
            existing: list = list(row.pending_writes or []) if row else []
            existing = [w for w in existing if w[0] != task_id]
            for channel, value in writes:
                existing.append([task_id, channel, _ser_writes_value(self.serde, value)])
            conn.execute(
                text(
                    "UPDATE workflow_checkpoints SET pending_writes = :pw"
                    " WHERE thread_id = :t AND checkpoint_id = :c"
                ),
                {"pw": _to_json(existing), "t": thread_id, "c": checkpoint_id},
            )
            conn.commit()

    def list(
        self,
        config: dict[str, Any],
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        sql = (
            "SELECT thread_id, checkpoint_id, parent_checkpoint_id, checkpoint, metadata,"
            " pending_writes FROM workflow_checkpoints WHERE thread_id = :t ORDER BY created_at DESC"
        )
        params: dict[str, Any] = {"t": thread_id}
        if limit:
            sql += " LIMIT :lim"
            params["lim"] = limit
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        for row in rows:
            parent_config = None
            if row.parent_checkpoint_id:
                parent_config = {
                    "configurable": {"thread_id": thread_id, "checkpoint_id": row.parent_checkpoint_id}
                }
            yield CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "checkpoint_id": row.checkpoint_id}},
                checkpoint=row.checkpoint,
                metadata=row.metadata or {},
                parent_config=parent_config,
                pending_writes=[
                    (w[0], w[1], _deser_writes_value(self.serde, w[2]))
                    for w in (row.pending_writes or [])
                ],
            )
