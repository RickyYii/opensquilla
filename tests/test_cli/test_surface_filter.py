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

from types import SimpleNamespace
from typing import Any

import pytest

from opensquilla.cli.sessions_cmd import _filter_sessions
from opensquilla.gateway.rpc_sessions import _derive_source_metadata
from opensquilla.gateway.session_view import build_session_view_item


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
def test_each_way_a_row_names_its_surface_is_matched(channel: str, expected: list[str]) -> None:
    assert _keys(_filter(ALL, channel)) == expected


def test_the_generic_bucket_is_not_a_channel() -> None:
    """`source_kind`'s "channel" classifies; it is not something to filter on.

    A session that arrived over a connector falls back to it, so matching it
    would return all of them at once under a name no operator chose.
    """

    assert _filter(ALL, "channel") == []


def test_the_unplaceable_surface_is_not_a_channel_either() -> None:
    """`_surface` ends in "unknown" for a session it could not place.

    `session/keys.py` even builds keys with "unknown" in the channel slot, so
    the sentinel is common. It is the same defect as the `channel` bucket:
    matching it hands back every unplaceable session under a name nobody
    chose.
    """

    unplaceable = _row(key="mystery:1", surface="unknown")

    assert _filter([unplaceable, WEBCHAT, SLACK], "unknown") == []


@pytest.mark.parametrize("reserved", ["channel", "unknown"])
def test_a_connector_actually_named_after_a_sentinel_is_still_reachable(
    reserved: str,
) -> None:
    """Connector names are free text, so both words are ones an operator can use.

    Skipping the sentinel globally would drop these rows, which is the same
    silent empty result this filter exists to remove.
    """

    named = _row(
        key=f"agent:main:{reserved}:channel:C4",
        source_kind="channel",
        sourceKind="channel",
        channel_kind=reserved,
        channelKind=reserved,
        last_channel=reserved,
        surface="unknown",
    )

    assert _keys(_filter([named, WEBCHAT], reserved)) == [f"agent:main:{reserved}:channel:C4"]


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
    """A row of empty strings must not answer to an empty-ish argument.

    Asking for a name the row does not carry would pass with or without the
    guard; the guard is what stops the blanks matching each other.
    """

    blank = _row(key="blank:1", channel="", source_kind="", channel_kind="", surface="")

    assert _filter([blank], "webchat") == []
    assert _filter([blank], "  ") == []
    # An empty string is falsy, so `_filter_sessions` skips the channel test
    # entirely — the same as not passing the flag, and what it did before.
    assert _keys(_filter([blank], "")) == ["blank:1"]


def test_surrounding_whitespace_in_a_row_value_is_ignored() -> None:
    """A connector name stored with stray whitespace still answers to itself."""

    padded = _row(key="padded:1", channel_kind="  slack-eng\n", surface="slack")

    assert _keys(_filter([padded], "slack-eng")) == ["padded:1"]


def test_the_legacy_source_channel_fields_are_matched() -> None:
    """Carried over from the filter this replaces; nothing else covers them."""

    snake = _row(key="src:1", source_channel="discord", surface="unknown")
    camel = _row(key="src:2", sourceChannel="discord", surface="unknown")

    assert _keys(_filter([snake, camel, WEBCHAT], "discord")) == ["src:1", "src:2"]


def test_an_expanded_channel_object_is_matched_by_name_and_type() -> None:
    """`sessions.list` types `channel` as `dict | str | None`.

    Stringifying the object yields `"{'name': ...}"`, which answers to
    nothing; the operator's name and the platform both have to come out.
    """

    expanded = _row(
        key="obj:1",
        channel={"name": "slack-eng", "type": "slack"},
        surface="unknown",
    )

    assert _keys(_filter([expanded], "slack-eng")) == ["obj:1"]
    assert _keys(_filter([expanded], "slack")) == ["obj:1"]
    assert _filter([expanded], "discord") == []


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
    assert _keys(_filter_sessions(ALL, agent=None, status="done", channel="slack", since=None)) == [
        "agent:main:slack:channel:C9"
    ]
    assert _filter_sessions(ALL, agent="main", status=None, channel="cron", since=None) == []


def test_a_platform_room_id_is_not_a_channel_name() -> None:
    """`channel_id` is the platform's own room id, deliberately left out.

    Folding it in would let a bare id shadow a connector named the same thing.
    """

    slack = dict(SLACK)
    slack["channel_id"] = "C9"
    slack["channelId"] = "C9"

    assert _filter([slack], "C9") == []
    assert _keys(_filter([slack], "slack-eng")) == ["agent:main:slack:channel:C9"]


def _projected_row(session: SimpleNamespace, channel_types: dict[str, str] | None = None) -> dict:
    """Build a row the way `_handle_sessions_list` does.

    `rpc_sessions.py` merges `_derive_source_metadata` and then
    `build_session_view_item` onto the base row, and the contract adapter
    returns the payload unchanged, so these are the field names that reach the
    CLI. Every hand-written row above is a claim about this function; this is
    the one place the claim is checked against the producers themselves.
    """

    row: dict[str, Any] = {
        "key": session.session_key,
        "status": getattr(session, "status", "unknown"),
        "channel": getattr(session, "channel", None),
    }
    row.update(_derive_source_metadata(session))
    row.update(
        build_session_view_item(
            session,
            entry_count=0,
            task_rows=[],
            now_ms=0,
            channel_types=channel_types,
        )
    )
    return row


def _session(session_key: str, **overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "session_key": session_key,
        "session_id": session_key,
        "agent_id": "main",
        "origin": None,
        "channel": None,
        "last_channel": None,
        "last_to": None,
        "status": "done",
        "created_at": None,
        "updated_at": None,
        "model": None,
        "title": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_the_gateway_really_sends_the_fields_this_filter_reads() -> None:
    """The whole fix rests on `surface` and the derived kinds being on the wire."""

    webchat = _projected_row(_session("agent:main:webchat:s1"))
    cron = _projected_row(_session("cron:nightly:s2"))
    slack = _projected_row(
        _session("agent:main:slack-eng:channel:C9", last_channel="slack-eng"),
        channel_types={"slack-eng": "slack"},
    )
    rows = [webchat, cron, slack]

    # The names the old filter looked at carry nothing for the first two.
    assert webchat["channel"] is None
    assert cron["channel"] is None

    assert _keys(_filter(rows, "webchat")) == ["agent:main:webchat:s1"]
    assert _keys(_filter(rows, "webui")) == ["agent:main:webchat:s1"]
    assert _keys(_filter(rows, "cron")) == ["cron:nightly:s2"]
    # The platform, resolved through the configured name->type map...
    assert _keys(_filter(rows, "slack")) == ["agent:main:slack-eng:channel:C9"]
    # ...and the name the operator actually gave that connector.
    assert _keys(_filter(rows, "slack-eng")) == ["agent:main:slack-eng:channel:C9"]


def test_a_session_the_gateway_cannot_place_is_not_swept_up() -> None:
    """`_surface` really does end at the "unknown" sentinel."""

    unplaceable = _projected_row(_session("mystery-key"))

    assert unplaceable["surface"] == "unknown"
    assert _filter([unplaceable], "unknown") == []
