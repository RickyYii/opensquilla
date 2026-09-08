"""A failed `sessions` command must not report success.

`sessions resume`, `delete` and `export` open their own gateway connection so
they can print a command-specific hint. That helper swallowed every failure
into a sentinel and each caller returned, so the process exited 0 whether the
gateway was unreachable or the RPC returned an error — issue #1448. `export`
is the reported case: a caller running `sessions export KEY && upload KEY.md`
proceeded to a file that was never written.

Their siblings — `list`, `show`, `abort` — go through `run_gateway_call`,
which maps the gateway's error code with `rpc_error_exit_code` and raises.
These tests pin that the three reach the same codes.
"""

from __future__ import annotations

import json
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


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException | None,
    *,
    raised_by: str = "call",
    delete_result: dict[str, Any] | None = None,
    history: dict[str, Any] | None = None,
) -> None:
    """A gateway client that fails where the real one fails.

    `GatewayClient.connect` wraps every failure in `SystemExit`, so that is
    the only thing it can raise. A `GatewayRPCError` or a dropped connection
    surfaces later, out of the RPC call itself, so those are raised from the
    methods — a fake that raises them from `connect` would exercise a path
    production cannot reach.
    """

    def _fail() -> None:
        if failure is not None and raised_by == "call":
            raise failure

    class FakeClient:
        async def connect(self, url: str, *, token: str | None = None) -> None:
            if failure is not None and raised_by == "connect":
                raise failure

        async def resolve_session(self, session_id: str) -> dict[str, Any]:
            _fail()
            return {"key": SESSION, "status": "idle", "model": "m", "updated_at": "u"}

        async def preview_sessions(self, keys: list[str]) -> dict[str, Any]:
            _fail()
            return {"previews": [{"lastMessage": "the last thing said"}]}

        async def session_history(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            _fail()
            return history if history is not None else {"messages": []}

        async def delete_sessions(self, keys: list[str]) -> dict[str, Any]:
            _fail()
            return delete_result if delete_result is not None else {"deleted": keys}

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
@pytest.mark.parametrize(
    "failure",
    [ConnectionRefusedError("connection refused"), OSError("socket closed")],
    ids=["connection-refused", "os-error"],
)
def test_a_dropped_connection_reports_rather_than_traces(
    name: str, failure: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_gateway_call` turns these into a message and exit 1; so must these.

    Left uncaught they reach Typer as an unhandled exception, so the operator
    sees a traceback where the sibling commands print one line.
    """

    _install_client(monkeypatch, failure)

    result = runner.invoke(app, COMMANDS[name])

    assert result.exit_code == 1, result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.stderr.strip(), "the failure has to be reported somewhere"
    assert result.stdout.strip() == ""
    # The sentinel exists to keep this apart from an unreachable gateway: the
    # gateway answered and then went away, so the hint would misdiagnose it.
    assert "requires a running gateway" not in result.stderr


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_an_unreachable_gateway_exits_one(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch, SystemExit("gateway is not running"), raised_by="connect")

    result = runner.invoke(app, COMMANDS[name])

    assert result.exit_code == 1, result.stdout
    # Both the shared diagnostic and the command-specific hint, on stderr.
    assert "gateway is not running" in result.stderr
    assert "requires a running gateway" in result.stderr
    assert result.stdout.strip() == ""


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

    _install_client(monkeypatch, GatewayRPCError("sessions.delete", code=code, message="boom"))

    result = runner.invoke(app, COMMANDS[name])

    assert result.exit_code == expected, result.stdout
    assert "boom" in result.stderr and "sessions.delete" in result.stderr


def test_a_successful_export_still_exits_zero_and_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fix must not turn ordinary success into a failure."""

    _install_client(monkeypatch, None)

    target = tmp_path / "out.md"
    result = runner.invoke(app, ["sessions", "export", SESSION, "--output", str(target)])

    assert result.exit_code == 0, result.stdout
    assert target.exists()


def test_a_failed_delete_leaves_stdout_parseable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sessions delete` writes its result as JSON on stdout.

    A diagnostic printed there too would land in the stream a caller is
    parsing, so the error panel has to go to stderr.
    """

    _install_client(
        monkeypatch,
        GatewayRPCError("sessions.delete", code="NOT_FOUND", message="boom"),
    )

    result = runner.invoke(app, COMMANDS["delete"])

    assert result.exit_code == 2
    assert result.stdout.strip() == ""
    assert "boom" in result.stderr and "sessions.delete" in result.stderr


def test_an_unwritable_output_path_reports_rather_than_traces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The transcript arrived; only the file write failed.

    Uncaught, it reached Typer as an unhandled exception and printed a
    traceback — the same defect the connection paths above had.
    """

    _install_client(monkeypatch, None)

    target = tmp_path / "missing" / "out.md"
    result = runner.invoke(app, ["sessions", "export", SESSION, "--output", str(target)])

    assert result.exit_code == 1, result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert str(target) in result.stderr
    assert not target.exists()


def test_a_bad_format_is_reported_on_stderr(tmp_path: Path) -> None:
    """Argument rejection is a failure too, and exits 2 like the RPC's."""

    result = runner.invoke(
        app,
        ["sessions", "export", SESSION, "--format", "yaml", "--output", str(tmp_path / "o")],
    )

    assert result.exit_code == 2
    assert "--format" in result.stderr
    assert result.stdout.strip() == ""


def test_a_delete_that_deleted_nothing_does_not_exit_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`sessions.delete` reports per-key failures inside a *successful* reply.

    `SessionLifecycle.delete` collects them into `DeleteSessionsResult.
    failures` and the adapter serialises them as `errors`, so the RPC never
    raises and the command used to exit 0. A script running
    `sessions delete KEY && ...` then believed the session was gone.
    """

    _install_client(
        monkeypatch,
        None,
        delete_result={"deleted": [], "errors": [f"{SESSION}: storage refused"]},
    )

    result = runner.invoke(app, COMMANDS["delete"])

    assert result.exit_code == 1, result.stdout
    assert "storage refused" in result.stderr
    # The envelope still goes to stdout, so a caller can see what did go.
    assert json.loads(result.stdout) == {
        "deleted": [],
        "errors": [f"{SESSION}: storage refused"],
    }


def test_a_partly_failed_delete_is_still_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """One key gone and one refused is not success for the caller."""

    _install_client(
        monkeypatch,
        None,
        delete_result={"deleted": ["other"], "errors": [f"{SESSION}: locked"]},
    )

    result = runner.invoke(app, COMMANDS["delete"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["deleted"] == ["other"]


def test_a_clean_delete_still_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty `errors` list must not be read as a failure."""

    _install_client(monkeypatch, None, delete_result={"deleted": [SESSION], "errors": []})

    result = runner.invoke(app, COMMANDS["delete"])

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["deleted"] == [SESSION]


@pytest.mark.parametrize("session_id", ["[/dim]", "[bold]loud"])
def test_a_session_id_is_not_read_as_markup(
    session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hint interpolates operator input into a rich markup string.

    Unescaped, `[/dim]` raised MarkupError before the exit could run — a
    traceback on the very path that exists to replace one — and `[bold]x`
    printed a mangled id naming the wrong session.
    """

    _install_client(monkeypatch, SystemExit("gateway is not running"), raised_by="connect")

    result = runner.invoke(app, ["sessions", "resume", session_id])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert session_id in result.stderr


@pytest.mark.parametrize(
    "failure",
    [OSError(), TimeoutError(), ConnectionResetError()],
    ids=["os-error", "timeout", "reset"],
)
def test_a_failure_with_no_message_still_names_itself(
    failure: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`str(OSError())` is empty, so the diagnostic was a bare "Error:".

    `TimeoutError` is an `OSError` subclass since 3.11, so an RPC timeout
    lands here too.
    """

    _install_client(monkeypatch, failure)

    result = runner.invoke(app, COMMANDS["export"])

    assert result.exit_code == 1
    assert type(failure).__name__ in result.stderr


def test_a_successful_export_writes_the_transcript_it_fetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`target.exists()` alone passes on a zero-byte file."""

    _install_client(
        monkeypatch,
        None,
        history={"messages": [{"role": "user", "content": "hello there"}]},
    )

    target = tmp_path / "out.md"
    result = runner.invoke(app, ["sessions", "export", SESSION, "--output", str(target)])

    assert result.exit_code == 0, result.stderr
    body = target.read_text(encoding="utf-8")
    assert SESSION in body
    assert "hello there" in body


def test_an_export_with_no_messages_falls_back_to_the_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The documented fallback, which no test covered."""

    _install_client(monkeypatch, None)

    target = tmp_path / "out.md"
    result = runner.invoke(app, ["sessions", "export", SESSION, "--output", str(target)])

    assert result.exit_code == 0, result.stderr
    assert "the last thing said" in target.read_text(encoding="utf-8")
