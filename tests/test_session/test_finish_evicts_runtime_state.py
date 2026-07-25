"""SessionManager.finish drops module-level subagent + routing bookkeeping."""

from __future__ import annotations

import pytest

from opensquilla.engine import cache_break_monitor
from opensquilla.engine.steps.squilla_router import _history_store
from opensquilla.gateway.subagent_announce import _tracker
from opensquilla.provider import ChatConfig, Message
from opensquilla.session.manager import SessionManager
from opensquilla.session.models import SessionStatus


class _MemoryStorage:
    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}

    async def get_session(self, session_key: str):
        return self._sessions.get(session_key)

    async def upsert_session(self, node) -> None:
        self._sessions[node.session_key] = node


@pytest.mark.asyncio
async def test_finish_evicts_spawn_group_tracker_and_routing_history() -> None:
    from opensquilla.session.models import SessionNode

    storage = _MemoryStorage()
    node = SessionNode(
        session_key="agent:main:main",
        session_id="abc",
        agent_id="main",
        created_at=1,
        updated_at=1,
        started_at=1,
        status=SessionStatus.RUNNING,
    )
    await storage.upsert_session(node)

    _tracker.mark_closed("agent:main:main", "task-X")
    _history_store.set("agent:main:main", [{"turn_index": 0}])
    assert _tracker.is_closed("agent:main:main", "task-X")
    assert _history_store.get("agent:main:main") is not None

    mgr = SessionManager(storage)  # type: ignore[arg-type]
    await mgr.finish("agent:main:main", status=SessionStatus.DONE)

    assert not _tracker.is_closed("agent:main:main", "task-X")
    assert _history_store.get("agent:main:main") is None


@pytest.mark.asyncio
async def test_finish_evicts_cache_break_monitor_state() -> None:
    """The prompt-cache baseline is keyed by session and belongs to the same hook."""
    from opensquilla.session.models import SessionNode

    storage = _MemoryStorage()
    session_key = "agent:main:cache-break"
    node = SessionNode(
        session_key=session_key,
        session_id="abc",
        agent_id="main",
        created_at=1,
        updated_at=1,
        started_at=1,
        status=SessionStatus.RUNNING,
    )
    await storage.upsert_session(node)

    monitor = cache_break_monitor.default_cache_break_monitor
    snapshot = monitor.record_prompt_state(
        messages=[Message(role="user", content="old"), Message(role="user", content="now")],
        tools=None,
        config=ChatConfig(system="stable system"),
        model="model-a",
    )
    monitor.check_response_for_cache_break(session_key, snapshot, 5000)
    monitor.notify_compaction(session_key)
    try:
        mgr = SessionManager(storage)  # type: ignore[arg-type]
        await mgr.finish(session_key, status=SessionStatus.DONE)

        assert monitor.evict(session_key) is False
    finally:
        monitor.evict(session_key)
