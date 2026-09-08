"""A failed `sessions` command must not report success.

`sessions resume`, `delete` and `export` open their own gateway connection so
they can print a command-specific hint. That helper swallowed every failure
into a sentinel and each caller returned, so the process exited 0 whether the
gateway was unreachable, the RPC returned an error, or nothing came back —
issue #1448. `export` is the reported case: a caller running
`sessions export KEY && upload KEY.md` proceeded to a file that was never
written.

Their siblings — `list`, `show`, `abort` — go through `run_gateway_call`,
which maps the gateway's error code with `rpc_error_exit_code` and raises.
These tests pin that the three reach the same codes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from opensquilla.cli.gateway_client import GatewayRPCError
from opensquilla.cli.main import app

runner = CliRunner()

SESSION = "agent:main:webchat:s1"


@pytest.fixture(autouse=True)
def _no_ambient_gateway(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENSQUILLA_GATEWAY_URL", raising=False)
    monkeypatch.delenv("OPENSQUILLA_GATEWAY_CONFIG_PATH", raising=False)
    empty = tmp_path / "home"
    empty.mkdir()
    monkeypatch.setenv("OPENSQUILLA_STATE_DIR", str(empty))


def _install_client(monkeypatch: pytest.MonkeyPatch, failure: BaseException | None) -> None:
    """A gateway client whose connect raises `failure`, or succeeds with no data."""

    class FakeClient:
        async def connect(self, url: str, *, token: str | None = None) -> None:
            if failure is not None:
                raise failure

        async def resolve_session(self, session_id: str) -> dict[str, Any]:
            return {"key": SESSION, "status": "idle"}

        async def preview_sessions(self, keys: list[str]) -> dict[str, Any]:
            return {"previews": []}

        async def session_history(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"messages": []}

        async def delete_sessions(self, keys: list[str]) -> dict[str, Any]:
            return {"deleted": keys}

        async def close(self) -> None:
            return None

    monkeypatch.setattr("opensquilla.cli.gateway_client.GatewayClient", FakeClient)


# `--yes` keeps `delete` off the confirmation prompt; `resume` hands off to the
# chat REPL on success, which these cases never reach.
COMMANDS = {
    "resume": ["sessions", "resume", SESSION],
    "delete": ["sessions", "delete", SESSION, "--yes"],
    "export": ["sessions", "export", SESSION],
}


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_an_unreachable_gateway_exits_one(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch, SystemExit("gateway is not running"))

    result = runner.invoke(app, COMMANDS[name])

    assert result.exit_code == 1, result.stdout


@pytest.mark.parametrize("name", sorted(COMMANDS))
@pytest.mark.parametrize(
    ("code", "expected"),
    [("NOT_FOUND", 2), ("INVALID_REQUEST", 2), ("CONFLICT", 3), ("INTERNAL", 1), (None, 1)],
)
def test_an_rpc_error_exits_with_the_shared_mapping(
    name: str,
    code: str | None,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same mapping `list`/`show`/`abort` get from `run_gateway_call`."""

    _install_client(monkeypatch, GatewayRPCError("boom", code=code))

    result = runner.invoke(app, COMMANDS[name])

    assert result.exit_code == expected, result.stdout


def test_export_exits_non_zero_when_no_data_comes_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No file is written on this path, so zero would be a lie about the file."""

    class EmptyClient:
        async def connect(self, url: str, *, token: str | None = None) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr("opensquilla.cli.gateway_client.GatewayClient", EmptyClient)
    monkeypatch.setattr(
        "opensquilla.cli.sessions_cmd._with_client",
        _returning_none,
    )

    target = tmp_path / "out.md"
    result = runner.invoke(app, ["sessions", "export", SESSION, "--output", str(target)])

    assert result.exit_code == 1, result.stdout
    assert not target.exists()


async def _returning_none(action: Any) -> None:
    return None


def test_a_successful_export_still_exits_zero_and_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fix must not turn ordinary success into a failure."""

    _install_client(monkeypatch, None)

    target = tmp_path / "out.md"
    result = runner.invoke(app, ["sessions", "export", SESSION, "--output", str(target)])

    assert result.exit_code == 0, result.stdout
    assert target.exists()
