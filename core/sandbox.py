"""Sandbox - safe command execution with resource limits."""

import subprocess
import os
import platform
import signal
import time
import re
import shlex
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("igris.sandbox")


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_seconds: float
    command: str


class CommandFilter:
    """Filters commands for safety."""

    BLOCKED: List[str] = [
        r"rm\s+-rf\s+/",
        r"del\s+/[sf]",
        r"format\s+[a-zA-Z]:",
        r"mkfs\s+",
        r"dd\s+if=",
        r":\(\)\{",
        r"chmod\s+-R\s+777\s+/",
        r">\s*/dev/sd",
        r"mv\s+/\s+",
        r"wget\s+.*\|\s*sh",
        r"curl\s+.*\|\s*sh",
    ]

    REQUIRE_CONFIRM: List[str] = [
        r"rm\s+",
        r"del\s+",
        r"sudo\s+",
        r"pip\s+install",
        r"npm\s+install\s+-g",
        r"git\s+push",
        r"docker\s+rm",
        r"docker\s+rmi",
        r"git\s+reset\s+--hard",
        r"chmod\s+",
        r"chown\s+",
    ]

    def __init__(self) -> None:
        self._blocked_compiled = [re.compile(p, re.IGNORECASE) for p in self.BLOCKED]
        self._confirm_compiled = [re.compile(p, re.IGNORECASE) for p in self.REQUIRE_CONFIRM]

    def is_blocked(self, cmd: str) -> bool:
        """Return True if command matches a blocked pattern."""
        for pat in self._blocked_compiled:
            if pat.search(cmd):
                logger.warning("Blocked command: %s", cmd[:100])
                return True
        return False

    def needs_confirmation(self, cmd: str) -> bool:
        """Return True if command requires user confirmation."""
        for pat in self._confirm_compiled:
            if pat.search(cmd):
                return True
        return False


class Sandbox:
    """Execute commands in a sandboxed environment."""

    def __init__(
        self,
        working_dir: Optional[str] = None,
        timeout: int = 60,
        max_output: int = 50000,
    ) -> None:
        self.working_dir = Path(working_dir).resolve() if working_dir else Path.cwd()
        self.timeout = timeout
        self.max_output = max_output
        self.command_filter = CommandFilter()

        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        command: str,
        timeout_override: Optional[int] = None,
        env: Optional[dict] = None,
    ) -> SandboxResult:
        """Execute a command with safety checks and resource limits."""
        effective_timeout = timeout_override if timeout_override is not None else self.timeout

        if self.command_filter.is_blocked(command):
            return SandboxResult(
                stdout="",
                stderr=f"BLOCKED: Command '{command}' matches a blocked pattern and was not executed.",
                exit_code=-1,
                timed_out=False,
                duration_seconds=0.0,
                command=command,
            )

        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        is_win = platform.system() == "Windows"
        start = time.monotonic()

        try:
            if is_win:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                    cwd=str(self.working_dir),
                    env=run_env,
                )
            else:
                proc = subprocess.run(
                    shlex.split(command),
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                    cwd=str(self.working_dir),
                    env=run_env,
                )
            duration = time.monotonic() - start

            stdout = proc.stdout
            stderr = proc.stderr
            if len(stdout) > self.max_output:
                stdout = stdout[: self.max_output] + f"\n... [truncated at {self.max_output} chars]"
            if len(stderr) > self.max_output:
                stderr = stderr[: self.max_output] + f"\n... [truncated at {self.max_output} chars]"

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                timed_out=False,
                duration_seconds=round(duration, 3),
                command=command,
            )

        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            logger.warning("Command timed out after %ds: %s", effective_timeout, command[:100])
            return SandboxResult(
                stdout="",
                stderr=f"Command timed out after {effective_timeout} seconds",
                exit_code=-1,
                timed_out=True,
                duration_seconds=round(duration, 3),
                command=command,
            )

        except FileNotFoundError as exc:
            duration = time.monotonic() - start
            return SandboxResult(
                stdout="",
                stderr=f"Command not found: {exc}",
                exit_code=127,
                timed_out=False,
                duration_seconds=round(duration, 3),
                command=command,
            )

        except Exception as exc:
            duration = time.monotonic() - start
            logger.exception("Sandbox execution error")
            return SandboxResult(
                stdout="",
                stderr=str(exc),
                exit_code=-1,
                timed_out=False,
                duration_seconds=round(duration, 3),
                command=command,
            )

    def execute_script(self, script: str, interpreter: str = "python3", timeout_override: Optional[int] = None) -> SandboxResult:
        """Write script to a temp file and execute it."""
        import tempfile
        suffix = ".py" if "python" in interpreter else ".sh"
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, dir=str(self.working_dir), delete=False)
        try:
            tmp.write(script)
            tmp.close()
            if suffix == ".sh":
                os.chmod(tmp.name, 0o755)
            return self.execute(f"{interpreter} {tmp.name}", timeout_override=timeout_override)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
