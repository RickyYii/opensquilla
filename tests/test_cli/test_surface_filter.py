"""`sessions list --channel` filters on the fields the gateway actually sends.

The gateway derives each session's surface in `_derive_source_metadata` and
projects it as `source_kind`/`channel_kind` (plus camelCase aliases): a WebChat
session arrives as `webui`/`webchat`, a cron one as `cron`/`cron`. The raw
`channel` field stays null for both, which is what `--json` shows.

The CLI filter looked only at `channel`, `last_channel` and `source_channel`,
so `--channel webchat`, `--channel webui` and `--channel cron` all matched
nothing while `--agent` worked — issue #1538. An empty result is
indistinguishable from "no such sessions", which is why it reads as data loss
rather than a filter bug.
"""

from __future__ import annotations

from typing import Any

import pytest

from opensquilla.cli.sessions_cmd import _filter_sessions


def _row(**overrides: Any) -> dict[str, Any]:
    """A row shaped like the gateway's session-list projection."""

    row: dict[str, Any] = {
        "session_key": "agent:main:webchat:s1",
        "agent_id": "main",
        "status": "idle",
        "channel": None,
        "last_channel": None,
        "source_kind": None,
        "sourceKind": None,
        "channel_kind": None,
        "channelKind": None,
    }
    row.update(overrides)
    return row


WEBCHAT = _row(source_kind="webui", sourceKind="webui", channel_kind="webchat", channelKind="webchat")
CRON = _row(
    session_key="cron:nightly:s2",
    agent_id="reminders",
    source_kind="cron",
    sourceKind="cron",
    channel_kind="cron",
    channelKind="cron",
)
SLACK = _row(
    session_key="agent:main:channel:s3",
    source_kind="channel",
    sourceKind="channel",
    channel_kind="slack",
    channelKind="slack",
    channel="slack",
)


def _keys(rows: list[dict[str, Any]]) -> list[str]:
    return [str(r["session_key"]) for r in rows]


def _filter(rows: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
    return _filter_sessions(rows, agent=None, status=None, channel=channel, since=None)


ALL = [WEBCHAT, CRON, SLACK]


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        ("webchat", ["agent:main:webchat:s1"]),
        ("webui", ["agent:main:webchat:s1"]),
        ("cron", ["cron:nightly:s2"]),
        ("slack", ["agent:main:channel:s3"]),
        ("channel", ["agent:main:channel:s3"]),
    ],
)
def test_the_derived_surface_names_are_matched(channel: str, expected: list[str]) -> None:
    assert _keys(_filter(ALL, channel)) == expected


def test_a_channel_nothing_carries_still_returns_nothing() -> None:
    """The fix must not make the filter permissive."""

    assert _filter(ALL, "telegram") == []


def test_the_match_is_case_insensitive_like_status() -> None:
    # `--status` is already compared case-insensitively in the same function.
    assert _keys(_filter(ALL, "WebChat")) == ["agent:main:webchat:s1"]
    assert _keys(_filter(ALL, "  CRON ")) == ["cron:nightly:s2"]


def test_a_gateway_that_predates_the_derivation_still_filters() -> None:
    """Older rows carry only `channel`/`last_channel`; those keep working."""

    legacy = _row(session_key="legacy:1", channel="discord")
    legacy_last = _row(session_key="legacy:2", last_channel="discord")

    assert _keys(_filter([legacy, legacy_last, WEBCHAT], "discord")) == ["legacy:1", "legacy:2"]


def test_blank_values_never_match() -> None:
    """Empty strings in the row must not collide with each other."""

    blank = _row(session_key="blank:1", channel="", source_kind="")

    assert _filter([blank], "webchat") == []


def test_the_other_filters_are_unaffected() -> None:
    rows = [WEBCHAT, CRON]

    assert _keys(
        _filter_sessions(rows, agent="main", status=None, channel=None, since=None)
    ) == ["agent:main:webchat:s1"]
    assert _keys(
        _filter_sessions(rows, agent=None, status="idle", channel="cron", since=None)
    ) == ["cron:nightly:s2"]
