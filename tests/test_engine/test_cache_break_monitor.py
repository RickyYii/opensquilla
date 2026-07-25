from __future__ import annotations

from opensquilla.engine import cache_break_monitor
from opensquilla.engine.cache_break_monitor import CacheBreakMonitor
from opensquilla.provider import ChatConfig, Message, ToolDefinition, ToolInputSchema


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} tool",
        input_schema=ToolInputSchema(properties={}),
    )


def test_cache_break_monitor_initializes_then_detects_attributed_drop() -> None:
    monitor = CacheBreakMonitor(min_drop_tokens=10, min_drop_ratio=0.05)
    first = monitor.record_prompt_state(
        messages=[
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
        ],
        tools=[_tool("search")],
        config=ChatConfig(
            system="stable system",
            cache_breakpoints=[{"text": "stable system", "cache": "true"}],
            cache_mode="auto",
        ),
        model="anthropic/claude-sonnet-4-6",
    )

    initial = monitor.check_response_for_cache_break("agent:main:s1", first, 5000)

    assert initial.break_detected is False
    assert initial.reason == "baseline_initialized"

    second = monitor.record_prompt_state(
        messages=[
            Message(role="user", content="different old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
        ],
        tools=[_tool("search")],
        config=ChatConfig(
            system="stable system",
            cache_breakpoints=[{"text": "stable system", "cache": "true"}],
            cache_mode="auto",
        ),
        model="anthropic/claude-sonnet-4-6",
    )

    report = monitor.check_response_for_cache_break("agent:main:s1", second, 100)

    assert report.break_detected is True
    assert report.reason == "cache_read_drop"
    assert report.changed_fields == ("messages_prefix_hash",)
    assert report.previous_cache_read_tokens == 5000
    assert report.current_cache_read_tokens == 100
    log_payload = report.to_log_dict()
    assert "forensics" in log_payload
    assert log_payload["forensics"]["previous"]["messages_prefix_item_hashes"]
    assert log_payload["forensics"]["previous"]["messages_prefix_item_kinds"] == ["history"]
    assert log_payload["forensics"]["current"]["cache_control_field_hashes"]


def test_cache_break_monitor_forensics_labels_request_context_prefix_items() -> None:
    monitor = CacheBreakMonitor(min_drop_tokens=10, min_drop_ratio=0.05)
    first = monitor.record_prompt_state(
        messages=[
            Message(role="user", content="[Request context for this turn]\nvolatile one"),
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
        ],
        tools=None,
        config=ChatConfig(system="stable system"),
        model="model-a",
    )
    monitor.check_response_for_cache_break("agent:main:s1", first, 5000)

    second = monitor.record_prompt_state(
        messages=[
            Message(role="user", content="[Request context for this turn]\nvolatile two"),
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
        ],
        tools=None,
        config=ChatConfig(system="stable system"),
        model="model-a",
    )

    report = monitor.check_response_for_cache_break("agent:main:s1", second, 100)

    assert report.break_detected is True
    payload = report.to_log_dict()
    assert payload["forensics"]["previous"]["messages_prefix_item_kinds"][0] == "request_context"
    assert payload["forensics"]["current"]["messages_prefix_item_kinds"][0] == "request_context"


def test_cache_break_monitor_resets_baseline_after_compaction() -> None:
    monitor = CacheBreakMonitor(min_drop_tokens=10, min_drop_ratio=0.05)
    before = monitor.record_prompt_state(
        messages=[Message(role="user", content="old"), Message(role="user", content="now")],
        tools=None,
        config=ChatConfig(system="stable system"),
        model="model-a",
    )
    monitor.check_response_for_cache_break("agent:main:s1", before, 5000)
    monitor.notify_compaction("agent:main:s1")

    after = monitor.record_prompt_state(
        messages=[
            Message(role="assistant", content="kept"),
            Message(role="user", content="now"),
        ],
        tools=None,
        config=ChatConfig(system="stable system"),
        model="model-a",
    )

    report = monitor.check_response_for_cache_break("agent:main:s1", after, 0)

    assert report.break_detected is False
    assert report.reason == "baseline_reset_after_compaction"
    assert report.baseline_reset is True


def test_notify_compaction_notifies_registered_listeners() -> None:
    events: list[tuple[str, dict]] = []

    remove = cache_break_monitor.add_compaction_listener(
        lambda session_key, payload: events.append((session_key, payload))
    )
    try:
        cache_break_monitor.notify_compaction(
            "agent:main:s1",
            source="manual",
            phase="manual",
            tokens_before=100,
            tokens_after=40,
        )
    finally:
        remove()

    assert events == [
        (
            "agent:main:s1",
            {
                "status": "completed",
                "source": "manual",
                "phase": "manual",
                "tokens_before": 100,
                "tokens_after": 40,
            },
        )
    ]


def test_notify_compaction_resets_cache_only_after_completed_status(
    monkeypatch,
) -> None:
    monitor = CacheBreakMonitor(min_drop_tokens=10, min_drop_ratio=0.05)
    monkeypatch.setattr(cache_break_monitor, "default_cache_break_monitor", monitor)
    before = monitor.record_prompt_state(
        messages=[Message(role="user", content="old"), Message(role="user", content="now")],
        tools=None,
        config=ChatConfig(system="stable system"),
        model="model-a",
    )
    monitor.check_response_for_cache_break("agent:main:s1", before, 5000)

    for status in ("started", "observed", "replayed"):
        cache_break_monitor.notify_compaction("agent:main:s1", status=status)
    after_started = monitor.record_prompt_state(
        messages=[
            Message(role="assistant", content="kept"),
            Message(role="user", content="now"),
        ],
        tools=None,
        config=ChatConfig(system="stable system"),
        model="model-a",
    )
    started_report = monitor.check_response_for_cache_break(
        "agent:main:s1", after_started, 0
    )

    assert started_report.reason != "baseline_reset_after_compaction"

    cache_break_monitor.notify_compaction("agent:main:s1", status="completed")
    after_completed = monitor.record_prompt_state(
        messages=[
            Message(role="assistant", content="new baseline"),
            Message(role="user", content="now"),
        ],
        tools=None,
        config=ChatConfig(system="stable system"),
        model="model-a",
    )
    completed_report = monitor.check_response_for_cache_break(
        "agent:main:s1", after_completed, 0
    )

    assert completed_report.reason == "baseline_reset_after_compaction"


def test_notify_compaction_can_reset_cache_without_notifying_listeners(
    monkeypatch,
) -> None:
    monitor = CacheBreakMonitor(min_drop_tokens=10, min_drop_ratio=0.05)
    monkeypatch.setattr(cache_break_monitor, "default_cache_break_monitor", monitor)
    events: list[tuple[str, dict]] = []
    remove = cache_break_monitor.add_compaction_listener(
        lambda session_key, payload: events.append((session_key, payload))
    )
    try:
        before = monitor.record_prompt_state(
            messages=[
                Message(role="user", content="old"),
                Message(role="user", content="now"),
            ],
            tools=None,
            config=ChatConfig(system="stable system"),
            model="model-a",
        )
        monitor.check_response_for_cache_break("agent:main:s1", before, 5000)

        cache_break_monitor.notify_compaction(
            "agent:main:s1",
            status="completed",
            notify_listeners=False,
        )

        after = monitor.record_prompt_state(
            messages=[
                Message(role="assistant", content="new baseline"),
                Message(role="user", content="now"),
            ],
            tools=None,
            config=ChatConfig(system="stable system"),
            model="model-a",
        )
        report = monitor.check_response_for_cache_break("agent:main:s1", after, 0)
    finally:
        remove()

    assert events == []
    assert report.reason == "baseline_reset_after_compaction"


def _record(monitor: CacheBreakMonitor, session_key: str, tokens: int = 5000) -> None:
    snapshot = monitor.record_prompt_state(
        messages=[
            Message(role="user", content=f"old {session_key}"),
            Message(role="user", content="now"),
        ],
        tools=None,
        config=ChatConfig(system="stable system"),
        model="model-a",
    )
    monitor.check_response_for_cache_break(session_key, snapshot, tokens)


def test_evict_drops_baseline_and_pending_reset_for_one_session() -> None:
    monitor = CacheBreakMonitor(min_drop_tokens=10, min_drop_ratio=0.05)
    _record(monitor, "agent:main:s1")
    _record(monitor, "agent:main:s2")
    monitor.notify_compaction("agent:main:s1")

    assert monitor.evict("agent:main:s1") is True
    assert monitor.tracked_session_count == 1

    # The evicted session re-initializes instead of reporting a stale break,
    # and the sibling session keeps its own baseline.
    after_evicted = monitor.record_prompt_state(
        messages=[Message(role="user", content="fresh"), Message(role="user", content="now")],
        tools=None,
        config=ChatConfig(system="different system"),
        model="model-b",
    )
    report = monitor.check_response_for_cache_break("agent:main:s1", after_evicted, 0)

    assert report.break_detected is False
    assert report.reason == "baseline_initialized"


def test_evict_reports_false_for_untracked_session() -> None:
    monitor = CacheBreakMonitor()

    assert monitor.evict("agent:main:never-seen") is False


def test_module_level_evict_session_targets_the_default_monitor(monkeypatch) -> None:
    monitor = CacheBreakMonitor(min_drop_tokens=10, min_drop_ratio=0.05)
    monkeypatch.setattr(cache_break_monitor, "default_cache_break_monitor", monitor)
    _record(monitor, "agent:main:s1")

    assert cache_break_monitor.evict_session("agent:main:s1") is True
    assert monitor.tracked_session_count == 0


def test_tracked_sessions_stay_bounded_without_explicit_eviction() -> None:
    monitor = CacheBreakMonitor(max_sessions=4)

    for index in range(50):
        _record(monitor, f"agent:main:s{index}")

    assert monitor.tracked_session_count == 4


def test_lru_eviction_never_reports_a_false_break() -> None:
    monitor = CacheBreakMonitor(min_drop_tokens=10, min_drop_ratio=0.05, max_sessions=2)
    _record(monitor, "agent:main:cold", tokens=5000)
    # Push the cold session out of the bound with unrelated traffic.
    _record(monitor, "agent:main:hot1")
    _record(monitor, "agent:main:hot2")

    # A cache-read collapse that WOULD have been attributed as a break now only
    # re-initializes: losing diagnostics state must never fabricate a finding.
    changed = monitor.record_prompt_state(
        messages=[Message(role="user", content="changed"), Message(role="user", content="now")],
        tools=None,
        config=ChatConfig(system="different system"),
        model="model-b",
    )
    report = monitor.check_response_for_cache_break("agent:main:cold", changed, 0)

    assert report.break_detected is False
    assert report.reason == "baseline_initialized"


def test_most_recently_used_session_survives_the_bound() -> None:
    monitor = CacheBreakMonitor(min_drop_tokens=10, min_drop_ratio=0.05, max_sessions=2)
    _record(monitor, "agent:main:keep", tokens=5000)
    _record(monitor, "agent:main:filler1")
    # Touching "keep" again must move it back to the fresh end of the bound.
    _record(monitor, "agent:main:keep", tokens=5000)
    _record(monitor, "agent:main:filler2")

    changed = monitor.record_prompt_state(
        messages=[Message(role="user", content="changed"), Message(role="user", content="now")],
        tools=None,
        config=ChatConfig(system="different system"),
        model="model-b",
    )
    report = monitor.check_response_for_cache_break("agent:main:keep", changed, 0)

    assert report.break_detected is True
    assert report.reason == "cache_read_drop"


def test_pending_resets_stay_bounded_without_a_paired_baseline() -> None:
    """`notify_compaction` accepts keys the baseline trim will never evict."""
    monitor = CacheBreakMonitor(max_sessions=4)

    for index in range(50):
        monitor.notify_compaction(f"agent:main:ghost{index}")

    assert monitor.tracked_session_count == 4
    # The most recent notification still does what it is for.
    snapshot = monitor.record_prompt_state(
        messages=[Message(role="user", content="old"), Message(role="user", content="now")],
        tools=None,
        config=ChatConfig(system="stable system"),
        model="model-a",
    )
    report = monitor.check_response_for_cache_break("agent:main:ghost49", snapshot, 0)

    assert report.reason == "baseline_reset_after_compaction"


def test_evict_reports_true_for_a_pending_reset_without_a_baseline() -> None:
    """A session can hold a pending reset and no baseline; eviction still counts."""
    monitor = CacheBreakMonitor()
    monitor.notify_compaction("agent:main:pending-only")

    assert monitor.evict("agent:main:pending-only") is True
    assert monitor.tracked_session_count == 0
    assert monitor.evict("agent:main:pending-only") is False
