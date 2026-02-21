"""
IGRIS Coding Skill.

Provides code review, refactoring suggestions, test generation,
bug analysis, language detection, and git operations.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger("igris.skills.coder")

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_EXT_LANG: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
    ".kt": "kotlin", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".cs": "csharp",
    ".php": "php", ".swift": "swift", ".dart": "dart",
    ".html": "html", ".css": "css", ".sql": "sql",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".yml": "yaml", ".yaml": "yaml", ".json": "json",
    ".toml": "toml", ".md": "markdown", ".r": "r",
}


def detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    ext = Path(file_path).suffix.lower()
    return _EXT_LANG.get(ext, "unknown")


# ---------------------------------------------------------------------------
# LLM integration helper
# ---------------------------------------------------------------------------

async def _llm_request(
    prompt: str,
    system: str = "You are IGRIS, an expert code analyst. Be concise and precise.",
    api_key: Optional[str] = None,
    model: str = "llama-3.1-70b-versatile",
) -> str:
    """Send a prompt to Groq LLM and return the response text."""
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 4096,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Code review
# ---------------------------------------------------------------------------

@dataclass
class ReviewIssue:
    line: Optional[int]
    severity: str  # "info" | "warning" | "error"
    message: str
    suggestion: str = ""


@dataclass
class ReviewResult:
    file_path: str
    language: str
    issues: list[ReviewIssue] = field(default_factory=list)
    summary: str = ""
    score: int = 0  # 0–100


async def review_file(file_path: str, api_key: Optional[str] = None) -> ReviewResult:
    """
    Analyse a source file and return structured review results.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    code = path.read_text(encoding="utf-8", errors="replace")
    lang = detect_language(file_path)

    prompt = (
        f"Review this {lang} code. Return a JSON object with:\n"
        f'- "issues": [{{"line": int|null, "severity": "info|warning|error", '
        f'"message": str, "suggestion": str}}]\n'
        f'- "summary": brief overall assessment\n'
        f'- "score": 0-100 quality score\n\n'
        f"```{lang}\n{code[:8000]}\n```"
    )

    raw = await _llm_request(prompt, api_key=api_key)

    # Parse JSON from response
    import json
    try:
        # Extract JSON block if wrapped in markdown
        match = re.search(r"```json?\s*(.*?)```", raw, re.DOTALL)
        data = json.loads(match.group(1) if match else raw)
    except (json.JSONDecodeError, AttributeError):
        return ReviewResult(file_path=file_path, language=lang, summary=raw, score=50)

    issues = [
        ReviewIssue(
            line=i.get("line"),
            severity=i.get("severity", "info"),
            message=i.get("message", ""),
            suggestion=i.get("suggestion", ""),
        )
        for i in data.get("issues", [])
    ]
    return ReviewResult(
        file_path=file_path,
        language=lang,
        issues=issues,
        summary=data.get("summary", ""),
        score=data.get("score", 50),
    )


# ---------------------------------------------------------------------------
# Refactoring suggestions
# ---------------------------------------------------------------------------

async def suggest_refactoring(file_path: str, api_key: Optional[str] = None) -> str:
    """Return refactoring suggestions for the given file."""
    code = Path(file_path).read_text(encoding="utf-8", errors="replace")
    lang = detect_language(file_path)
    prompt = (
        f"Suggest specific refactoring improvements for this {lang} code. "
        f"Focus on: readability, DRY, SOLID, performance, naming.\n\n"
        f"```{lang}\n{code[:8000]}\n```"
    )
    return await _llm_request(prompt, api_key=api_key)


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------

async def generate_tests(file_path: str, framework: str = "auto", api_key: Optional[str] = None) -> str:
    """Generate unit tests for the given source file."""
    code = Path(file_path).read_text(encoding="utf-8", errors="replace")
    lang = detect_language(file_path)

    framework_hint = ""
    if framework == "auto":
        if lang == "python":
            framework_hint = "Use pytest."
        elif lang in ("javascript", "typescript"):
            framework_hint = "Use Jest."
        elif lang == "go":
            framework_hint = "Use testing package."
    else:
        framework_hint = f"Use {framework}."

    prompt = (
        f"Generate comprehensive unit tests for this {lang} code. {framework_hint}\n"
        f"Cover edge cases, error paths, and main functionality.\n\n"
        f"```{lang}\n{code[:8000]}\n```"
    )
    return await _llm_request(prompt, api_key=api_key)


# ---------------------------------------------------------------------------
# Bug analysis
# ---------------------------------------------------------------------------

async def analyze_bug(error_message: str, code_context: str = "", api_key: Optional[str] = None) -> str:
    """Analyse an error message and suggest fixes."""
    prompt = (
        f"Analyse this error and explain the root cause and how to fix it:\n\n"
        f"Error:\n```\n{error_message[:3000]}\n```\n"
    )
    if code_context:
        prompt += f"\nRelevant code:\n```\n{code_context[:4000]}\n```"
    return await _llm_request(prompt, api_key=api_key)


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

class GitOps:
    """Wrapper for common git operations."""

    def __init__(self, repo_path: str = ".") -> None:
        self.repo = Path(repo_path).resolve()

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 and result.stderr.strip():
            logger.warning("git %s stderr: %s", args[0], result.stderr.strip())
        return result.stdout.strip()

    def status(self) -> str:
        return self._run("status", "--short")

    def diff(self, staged: bool = False) -> str:
        args = ["diff", "--stat"]
        if staged:
            args.append("--cached")
        return self._run(*args)

    def diff_full(self, staged: bool = False) -> str:
        args = ["diff"]
        if staged:
            args.append("--cached")
        return self._run(*args)

    def log(self, count: int = 10) -> str:
        return self._run("log", f"--oneline", f"-{count}")

    def add(self, paths: Optional[list[str]] = None) -> str:
        targets = paths or ["."]
        return self._run("add", *targets)

    def commit(self, message: str) -> str:
        return self._run("commit", "-m", message)

    def push(self, remote: str = "origin", branch: str = "") -> str:
        args = ["push", remote]
        if branch:
            args.append(branch)
        return self._run(*args)

    def current_branch(self) -> str:
        return self._run("branch", "--show-current")
