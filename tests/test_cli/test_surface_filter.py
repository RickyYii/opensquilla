"""`sessions list --channel` filters on the fields the gateway actually sends.

A session row describes its surface three ways. `surface`
(`gateway/session_view._surface`) is the platform, resolved through the
configured name->type map, so a connector the operator named "slack-eng" still
reports `slack`. `channel_kind` (`rpc_sessions._derive_source_metadata`) is
that operator-chosen name. `source_kind` is the origin — `webui`, `cli`,
`subagent`, `cron`. A WebChat session arrives as `webui`/`webchat` and a cron
one as `cron`/`cron`, with the raw `channel` field null for both, which is what
`--json` shows.

The filter looked only at `channel`, `last_channel` and `source_channel`, so
`--channel webchat`, `--channel webui` and `--channel cron` all matched nothing
while `--agent` worked — issue #1538. An empty result is indistinguishable from
"there are no such sessions", which is why it reads as data loss rather than a
filter bug.
"""

from __future__ import annotations

from typing import Any

import pytest

from opensquilla.cli.sessions_cmd import _filter_sessions


def _row(**overrides: Any) -> dict[str, Any]:
    """A row shaped like the session-list projection.

    Field names taken from `rpc_sessions._session_list_payload`: the key is
    `key`, not `session_key`, and `status` is a `SessionStatus` value.
    `_derive_source_metadata` and `build_session_view_item` are both merged in,
    so the derived names and `surface` arrive together.
    """

    row: dict[str, Any] = {
        "key": "agent:main:webchat:s1",
        "agent_id": "main",
        "agentId": "main",
        "status": "done",
        "channel": None,
        "last_channel": None,
        "source_kind": None,
        "sourceKind": None,
        "channel_kind": None,
        "channelKind": None,
        "surface": "unknown",
    }
    row.update(overrides)
    return row


WEBCHAT = _row(
    source_kind="webui",
    sourceKind="webui",
    channel_kind="webchat",
    channelKind="webchat",
    surface="webchat",
)
CRON = _row(
    key="cron:nightly:s2",
    agent_id="reminders",
    agentId="reminders",
    source_kind="cron",
    sourceKind="cron",
    channel_kind="cron",
    channelKind="cron",
    surface="cron",
)
# An operator-named connector: `channel_kind` carries their name, `surface`
# resolves it to the platform, and `source_kind` is the generic bucket.
SLACK = _row(
    key="agent:main:slack:channel:C9",
    source_kind="channel",
    sourceKind="channel",
    channel_kind="slack-eng",
    channelKind="slack-eng",
    last_channel="slack-eng",
    surface="slack",
)


def _keys(rows: list[dict[str, Any]]) -> list[str]:
    return [str(r["key"]) for r in rows]


def _filter(rows: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
    return _filter_sessions(rows, agent=None, status=None, channel=channel, since=None)


ALL = [WEBCHAT, CRON, SLACK]


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        ("webchat", ["agent:main:webchat:s1"]),
        ("webui", ["agent:main:webchat:s1"]),
        ("cron", ["cron:nightly:s2"]),
        # The platform, for a connector not named after it.
        ("slack", ["agent:main:slack:channel:C9"]),
        # The operator's own name for that connector.
        ("slack-eng", ["agent:main:slack:channel:C9"]),
    ],
)
def test_each_way_a_row_names_its_surface_is_matched(
    channel: str, expected: list[str]
) -> None:
    assert _keys(_filter(ALL, channel)) == expected


def test_the_generic_bucket_is_not_a_channel() -> None:
    """`source_kind == "channel"` classifies; it is not something to filter on.

    Every channel-backed session carries it, so matching it would make
    `--channel channel` return all of them at once under a name no operator
    chose.
    """

    assert _filter(ALL, "channel") == []


def test_a_row_that_only_resolved_a_surface_is_still_reachable() -> None:
    """`_derive_source_metadata` leaves both kinds null until `last_channel` is
    written, and its `elif` chain has no fall-through. `surface` is derived
    separately from the session key, so it still answers."""

    early = _row(key="agent:main:discord:channel:C1", surface="discord")

    assert _keys(_filter([early, WEBCHAT], "discord")) == ["agent:main:discord:channel:C1"]


def test_a_camelcase_only_row_is_matched() -> None:
    """Both spellings ship together today; neither alone may be a blind spot."""

    camel = _row(key="camel:1", channelKind="telegram", surface="unknown")

    assert _keys(_filter([camel], "telegram")) == ["camel:1"]


def test_a_channel_nothing_carries_still_returns_nothing() -> None:
    """The fix must not make the filter permissive."""

    assert _filter(ALL, "telegram") == []


def test_the_match_is_case_insensitive_like_status() -> None:
    # `--status` is already compared case-insensitively in the same function.
    assert _keys(_filter(ALL, "WebChat")) == ["agent:main:webchat:s1"]
    assert _keys(_filter(ALL, "  CRON ")) == ["cron:nightly:s2"]


def test_a_blank_channel_matches_nothing_rather_than_everything() -> None:
    """An argument that looks applied must not quietly list the lot."""

    assert _filter(ALL, "   ") == []


def test_a_gateway_that_predates_the_derivation_still_filters() -> None:
    """Older rows carry only `channel`/`last_channel`; those keep working."""

    legacy = _row(key="legacy:1", channel="discord", surface="unknown")
    legacy_last = _row(key="legacy:2", last_channel="discord", surface="unknown")

    assert _keys(_filter([legacy, legacy_last, WEBCHAT], "discord")) == ["legacy:1", "legacy:2"]


def test_blank_values_never_match() -> None:
    """Empty strings in the row must not collide with each other."""

    blank = _row(key="blank:1", channel="", source_kind="", surface="")

    assert _filter([blank], "webchat") == []


def test_a_row_whose_origin_and_platform_disagree_answers_to_both() -> None:
    """A cron job delivering into Slack is reachable as either."""

    cron_into_slack = _row(
        key="cron:notify:s9",
        source_kind="cron",
        sourceKind="cron",
        channel_kind="slack-ops",
        channelKind="slack-ops",
        surface="slack",
    )

    assert _keys(_filter([cron_into_slack], "cron")) == ["cron:notify:s9"]
    assert _keys(_filter([cron_into_slack], "slack")) == ["cron:notify:s9"]
    assert _keys(_filter([cron_into_slack], "slack-ops")) == ["cron:notify:s9"]


def test_the_other_filters_still_compose() -> None:
    assert _keys(
        _filter_sessions(ALL, agent="reminders", status=None, channel=None, since=None)
    ) == ["cron:nightly:s2"]
    assert _keys(
        _filter_sessions(ALL, agent=None, status="done", channel="slack", since=None)
    ) == ["agent:main:slack:channel:C9"]
    assert (
        _filter_sessions(ALL, agent="main", status=None, channel="cron", since=None) == []
    )
