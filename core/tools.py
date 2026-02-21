"""IGRIS Tool System - 25+ real tools with registry pattern."""

import os
import ast
import subprocess
import shutil
import json
import platform
import psutil
import zipfile
import tarfile
import re
import hashlib
import base64
import tempfile
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

import httpx

logger = logging.getLogger("igris.tools")


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str = ""
    data: Any = None


@dataclass
class ToolParam:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    name: str
    description: str
    category: str
    params: List[ToolParam]
    danger_level: int
    execute: Callable


class ToolRegistry:
    """Central registry for all IGRIS tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        category: str,
        params: List[ToolParam],
        danger_level: int = 0,
    ) -> Callable:
        """Decorator to register a tool function."""
        def decorator(fn: Callable) -> Callable:
            self._tools[name] = Tool(
                name=name,
                description=description,
                category=category,
                params=params,
                danger_level=danger_level,
                execute=fn,
            )
            return fn
        return decorator

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[Tool]:
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_tools_for_llm(self) -> List[Dict]:
        """Return tools in OpenAI function-calling format."""
        result = []
        for tool in self._tools.values():
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for p in tool.params:
                properties[p.name] = {"type": p.type, "description": p.description}
                if p.default is not None:
                    properties[p.name]["default"] = p.default
                if p.required:
                    required.append(p.name)
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return result

    def execute_tool(self, name: str, args: Dict[str, Any], security_manager: Any = None) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, output="", error=f"Tool '{name}' not found")
        if security_manager and hasattr(security_manager, "check_tool_permission"):
            allowed = security_manager.check_tool_permission(name, tool.danger_level)
            if not allowed:
                return ToolResult(success=False, output="", error=f"Permission denied for tool '{name}' (danger_level={tool.danger_level})")
        try:
            return tool.execute(**args)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return ToolResult(success=False, output="", error=str(exc))


# ---------------------------------------------------------------------------
# Global registry instance
# ---------------------------------------------------------------------------
registry = ToolRegistry()

# ===== FILE TOOLS ==========================================================

@registry.register(
    "read_file", "Read the contents of a file", "file",
    [ToolParam("path", "string", "File path to read"),
     ToolParam("encoding", "string", "File encoding", required=False, default="utf-8")],
    danger_level=0,
)
def read_file(path: str, encoding: str = "utf-8") -> ToolResult:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return ToolResult(success=False, output="", error=f"File not found: {p}")
    if not p.is_file():
        return ToolResult(success=False, output="", error=f"Not a file: {p}")
    try:
        content = p.read_text(encoding=encoding)
        return ToolResult(success=True, output=content, data={"size": p.stat().st_size, "path": str(p)})
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "write_file", "Write content to a file (creates dirs if needed)", "file",
    [ToolParam("path", "string", "File path"),
     ToolParam("content", "string", "Content to write"),
     ToolParam("encoding", "string", "Encoding", required=False, default="utf-8")],
    danger_level=1,
)
def write_file(path: str, content: str, encoding: str = "utf-8") -> ToolResult:
    p = Path(path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return ToolResult(success=True, output=f"Written {len(content)} chars to {p}", data={"path": str(p)})
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "delete_file", "Move a file to trash (recoverable)", "file",
    [ToolParam("path", "string", "File path to delete")],
    danger_level=1,
)
def delete_file(path: str) -> ToolResult:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return ToolResult(success=False, output="", error=f"Not found: {p}")
    trash_dir = Path.home() / ".igris" / "trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    dest = trash_dir / f"{ts}_{p.name}"
    try:
        shutil.move(str(p), str(dest))
        meta = {"original": str(p), "deleted_at": ts, "trash_path": str(dest)}
        dest.with_suffix(dest.suffix + ".meta.json").write_text(json.dumps(meta))
        return ToolResult(success=True, output=f"Moved to trash: {dest}", data=meta)
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "move_file", "Move/rename a file or directory", "file",
    [ToolParam("source", "string", "Source path"),
     ToolParam("destination", "string", "Destination path")],
    danger_level=1,
)
def move_file(source: str, destination: str) -> ToolResult:
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if not src.exists():
        return ToolResult(success=False, output="", error=f"Source not found: {src}")
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return ToolResult(success=True, output=f"Moved {src} → {dst}")
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "copy_file", "Copy a file or directory", "file",
    [ToolParam("source", "string", "Source path"),
     ToolParam("destination", "string", "Destination path")],
    danger_level=0,
)
def copy_file(source: str, destination: str) -> ToolResult:
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if not src.exists():
        return ToolResult(success=False, output="", error=f"Source not found: {src}")
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        return ToolResult(success=True, output=f"Copied {src} → {dst}")
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "list_directory", "List files and directories", "file",
    [ToolParam("path", "string", "Directory path", required=False, default="."),
     ToolParam("recursive", "boolean", "Recurse into subdirs", required=False, default=False),
     ToolParam("pattern", "string", "Glob pattern filter", required=False, default="*")],
    danger_level=0,
)
def list_directory(path: str = ".", recursive: bool = False, pattern: str = "*") -> ToolResult:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return ToolResult(success=False, output="", error=f"Not a directory: {p}")
    try:
        items = list(p.rglob(pattern)) if recursive else list(p.glob(pattern))
        entries = []
        for item in sorted(items):
            stat = item.stat()
            entries.append({
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        lines = []
        for e in entries:
            kind = "DIR " if e["is_dir"] else "FILE"
            lines.append(f"  {kind}  {e['size']:>10}  {e['modified']}  {e['name']}")
        output = f"Directory: {p}\n" + "\n".join(lines)
        return ToolResult(success=True, output=output, data=entries)
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "search_files", "Search for text in files using regex", "file",
    [ToolParam("pattern", "string", "Regex pattern to search"),
     ToolParam("path", "string", "Directory to search in", required=False, default="."),
     ToolParam("file_pattern", "string", "File glob filter", required=False, default="*")],
    danger_level=0,
)
def search_files(pattern: str, path: str = ".", file_pattern: str = "*") -> ToolResult:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return ToolResult(success=False, output="", error=f"Path not found: {p}")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return ToolResult(success=False, output="", error=f"Invalid regex: {exc}")
    matches: List[Dict] = []
    for fp in (p.rglob(file_pattern) if p.is_dir() else [p]):
        if not fp.is_file():
            continue
        try:
            text = fp.read_text(errors="ignore")
            for lineno, line in enumerate(text.splitlines(), 1):
                if compiled.search(line):
                    matches.append({"file": str(fp), "line": lineno, "text": line.strip()})
        except Exception:
            continue
    output_lines = [f"{m['file']}:{m['line']}: {m['text']}" for m in matches[:200]]
    return ToolResult(
        success=True,
        output=f"Found {len(matches)} matches\n" + "\n".join(output_lines),
        data=matches[:200],
    )


@registry.register(
    "file_info", "Get detailed file information", "file",
    [ToolParam("path", "string", "File path")],
    danger_level=0,
)
def file_info(path: str) -> ToolResult:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return ToolResult(success=False, output="", error=f"Not found: {p}")
    try:
        stat = p.stat()
        info = {
            "path": str(p),
            "name": p.name,
            "is_file": p.is_file(),
            "is_dir": p.is_dir(),
            "size": stat.st_size,
            "permissions": oct(stat.st_mode),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
        }
        if p.is_file() and stat.st_size < 10_000_000:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            info["sha256"] = sha
        lines = [f"{k}: {v}" for k, v in info.items()]
        return ToolResult(success=True, output="\n".join(lines), data=info)
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


# ===== SYSTEM TOOLS ========================================================

@registry.register(
    "run_command", "Execute a shell command", "system",
    [ToolParam("command", "string", "Command to execute"),
     ToolParam("timeout", "integer", "Timeout in seconds", required=False, default=60),
     ToolParam("working_dir", "string", "Working directory", required=False, default=None)],
    danger_level=2,
)
def run_command(command: str, timeout: int = 60, working_dir: Optional[str] = None) -> ToolResult:
    cwd = working_dir or os.getcwd()
    is_win = platform.system() == "Windows"
    try:
        if is_win:
            proc = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=cwd,
            )
        else:
            import shlex
            proc = subprocess.run(
                shlex.split(command), capture_output=True, text=True,
                timeout=timeout, cwd=cwd,
            )
        output = proc.stdout
        if len(output) > 50000:
            output = output[:50000] + "\n... [truncated]"
        stderr = proc.stderr
        if len(stderr) > 10000:
            stderr = stderr[:10000] + "\n... [truncated]"
        return ToolResult(
            success=proc.returncode == 0,
            output=output,
            error=stderr if proc.returncode != 0 else "",
            data={"exit_code": proc.returncode, "command": command},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error=f"Command timed out after {timeout}s")
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "system_info", "Get system information (CPU, RAM, disk, OS)", "system",
    [], danger_level=0,
)
def system_info() -> ToolResult:
    try:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "ram_total_gb": round(mem.total / (1024 ** 3), 2),
            "ram_used_gb": round(mem.used / (1024 ** 3), 2),
            "ram_percent": mem.percent,
            "disk_total_gb": round(disk.total / (1024 ** 3), 2),
            "disk_used_gb": round(disk.used / (1024 ** 3), 2),
            "disk_percent": disk.percent,
        }
        lines = [f"{k}: {v}" for k, v in info.items()]
        return ToolResult(success=True, output="\n".join(lines), data=info)
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "list_processes", "List running processes", "system",
    [ToolParam("sort_by", "string", "Sort field: cpu or memory", required=False, default="memory"),
     ToolParam("limit", "integer", "Max processes to return", required=False, default=20)],
    danger_level=0,
)
def list_processes(sort_by: str = "memory", limit: int = 20) -> ToolResult:
    try:
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
            try:
                info = p.info
                procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        key = "memory_percent" if sort_by == "memory" else "cpu_percent"
        procs.sort(key=lambda x: x.get(key) or 0, reverse=True)
        procs = procs[:limit]
        lines = [f"{'PID':>7}  {'CPU%':>6}  {'MEM%':>6}  {'STATUS':<12}  NAME"]
        for pr in procs:
            lines.append(
                f"{pr['pid']:>7}  {(pr.get('cpu_percent') or 0):>6.1f}  "
                f"{(pr.get('memory_percent') or 0):>6.1f}  {pr.get('status', '?'):<12}  {pr.get('name', '?')}"
            )
        return ToolResult(success=True, output="\n".join(lines), data=procs)
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "kill_process", "Kill a process by PID", "system",
    [ToolParam("pid", "integer", "Process ID to kill")],
    danger_level=2,
)
def kill_process(pid: int) -> ToolResult:
    try:
        p = psutil.Process(pid)
        name = p.name()
        p.terminate()
        try:
            p.wait(timeout=5)
        except psutil.TimeoutExpired:
            p.kill()
        return ToolResult(success=True, output=f"Killed process {pid} ({name})")
    except psutil.NoSuchProcess:
        return ToolResult(success=False, output="", error=f"Process {pid} not found")
    except psutil.AccessDenied:
        return ToolResult(success=False, output="", error=f"Access denied for PID {pid}")
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "get_env_var", "Get an environment variable value", "system",
    [ToolParam("name", "string", "Variable name")],
    danger_level=0,
)
def get_env_var(name: str) -> ToolResult:
    val = os.environ.get(name)
    if val is None:
        return ToolResult(success=False, output="", error=f"Env var '{name}' not set")
    return ToolResult(success=True, output=val, data={"name": name, "value": val})


# ===== WEB TOOLS ===========================================================

@registry.register(
    "web_search", "Search the web via Brave API (fallback: DuckDuckGo)", "web",
    [ToolParam("query", "string", "Search query"),
     ToolParam("count", "integer", "Number of results", required=False, default=5)],
    danger_level=0,
)
def web_search(query: str, count: int = 5) -> ToolResult:
    api_key = os.environ.get("BRAVE_API_KEY")
    results: List[Dict] = []

    if api_key:
        try:
            resp = httpx.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("web", {}).get("results", [])[:count]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                })
        except Exception as exc:
            logger.warning("Brave API failed, falling back to DuckDuckGo: %s", exc)

    if not results:
        try:
            resp = httpx.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
                follow_redirects=True,
            )
            resp.raise_for_status()
            html = resp.text
            link_pattern = re.compile(r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
            snippet_pattern = re.compile(r'<a class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
            links = link_pattern.findall(html)
            snippets = snippet_pattern.findall(html)
            for i, (url, title) in enumerate(links[:count]):
                title_clean = re.sub(r"<[^>]+>", "", title).strip()
                snip = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
                results.append({"title": title_clean, "url": url, "snippet": snip})
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Search failed: {exc}")

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    return ToolResult(success=True, output="\n\n".join(lines), data=results)


@registry.register(
    "web_fetch", "Fetch a URL and return its text content", "web",
    [ToolParam("url", "string", "URL to fetch"),
     ToolParam("max_length", "integer", "Max chars to return", required=False, default=50000)],
    danger_level=0,
)
def web_fetch(url: str, max_length: int = 50000) -> ToolResult:
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; IGRIS/1.0)"},
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        text = resp.text
        if len(text) > max_length:
            text = text[:max_length] + "\n... [truncated]"
        return ToolResult(
            success=True, output=text,
            data={"url": url, "status": resp.status_code, "length": len(resp.text)},
        )
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


# ===== UTILITY TOOLS =======================================================

@registry.register(
    "calculator", "Evaluate a mathematical expression safely", "util",
    [ToolParam("expression", "string", "Math expression to evaluate")],
    danger_level=0,
)
def calculator(expression: str) -> ToolResult:
    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Call, ast.Attribute, ast.Import, ast.ImportFrom)):
                return ToolResult(success=False, output="", error="Unsafe expression: function calls not allowed")
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
        return ToolResult(success=True, output=str(result), data={"expression": expression, "result": result})
    except Exception as exc:
        return ToolResult(success=False, output="", error=f"Calc error: {exc}")


@registry.register(
    "json_format", "Pretty-print / validate JSON", "util",
    [ToolParam("text", "string", "JSON string to format")],
    danger_level=0,
)
def json_format(text: str) -> ToolResult:
    try:
        parsed = json.loads(text)
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
        return ToolResult(success=True, output=formatted, data=parsed)
    except json.JSONDecodeError as exc:
        return ToolResult(success=False, output="", error=f"Invalid JSON: {exc}")


@registry.register(
    "base64_encode", "Encode text to base64", "util",
    [ToolParam("text", "string", "Text to encode")],
    danger_level=0,
)
def base64_encode(text: str) -> ToolResult:
    encoded = base64.b64encode(text.encode()).decode()
    return ToolResult(success=True, output=encoded)


@registry.register(
    "base64_decode", "Decode base64 to text", "util",
    [ToolParam("text", "string", "Base64 string to decode")],
    danger_level=0,
)
def base64_decode(text: str) -> ToolResult:
    try:
        decoded = base64.b64decode(text).decode()
        return ToolResult(success=True, output=decoded)
    except Exception as exc:
        return ToolResult(success=False, output="", error=f"Decode error: {exc}")


@registry.register(
    "hash_text", "Hash text with md5 or sha256", "util",
    [ToolParam("text", "string", "Text to hash"),
     ToolParam("algorithm", "string", "Hash algorithm: md5 or sha256", required=False, default="sha256")],
    danger_level=0,
)
def hash_text(text: str, algorithm: str = "sha256") -> ToolResult:
    if algorithm == "md5":
        h = hashlib.md5(text.encode()).hexdigest()
    elif algorithm == "sha256":
        h = hashlib.sha256(text.encode()).hexdigest()
    else:
        return ToolResult(success=False, output="", error=f"Unsupported algorithm: {algorithm}")
    return ToolResult(success=True, output=h, data={"algorithm": algorithm, "hash": h})


@registry.register(
    "timestamp_now", "Get current UTC timestamp", "util",
    [], danger_level=0,
)
def timestamp_now() -> ToolResult:
    now = datetime.utcnow()
    return ToolResult(
        success=True,
        output=now.isoformat() + "Z",
        data={"iso": now.isoformat() + "Z", "unix": time.time()},
    )


# ===== ARCHIVE TOOLS =======================================================

@registry.register(
    "compress", "Create a zip or tar.gz archive", "archive",
    [ToolParam("source", "string", "File or directory to compress"),
     ToolParam("output", "string", "Output archive path"),
     ToolParam("format", "string", "Archive format: zip or tar.gz", required=False, default="zip")],
    danger_level=1,
)
def compress(source: str, output: str, format: str = "zip") -> ToolResult:
    src = Path(source).expanduser().resolve()
    out = Path(output).expanduser().resolve()
    if not src.exists():
        return ToolResult(success=False, output="", error=f"Source not found: {src}")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        if format == "zip":
            with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as zf:
                if src.is_file():
                    zf.write(str(src), src.name)
                else:
                    for fp in src.rglob("*"):
                        if fp.is_file():
                            zf.write(str(fp), str(fp.relative_to(src.parent)))
        elif format in ("tar.gz", "tgz"):
            with tarfile.open(str(out), "w:gz") as tf:
                tf.add(str(src), arcname=src.name)
        else:
            return ToolResult(success=False, output="", error=f"Unsupported format: {format}")
        size = out.stat().st_size
        return ToolResult(success=True, output=f"Created {out} ({size} bytes)", data={"path": str(out), "size": size})
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


@registry.register(
    "extract", "Extract a zip or tar.gz archive", "archive",
    [ToolParam("archive", "string", "Archive file path"),
     ToolParam("destination", "string", "Extraction directory", required=False, default=".")],
    danger_level=1,
)
def extract(archive: str, destination: str = ".") -> ToolResult:
    arc = Path(archive).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if not arc.exists():
        return ToolResult(success=False, output="", error=f"Archive not found: {arc}")
    try:
        dst.mkdir(parents=True, exist_ok=True)
        if arc.suffix == ".zip":
            with zipfile.ZipFile(str(arc), "r") as zf:
                zf.extractall(str(dst))
                names = zf.namelist()
        elif arc.name.endswith(".tar.gz") or arc.name.endswith(".tgz"):
            with tarfile.open(str(arc), "r:gz") as tf:
                tf.extractall(str(dst))
                names = tf.getnames()
        elif arc.name.endswith(".tar"):
            with tarfile.open(str(arc), "r") as tf:
                tf.extractall(str(dst))
                names = tf.getnames()
        else:
            return ToolResult(success=False, output="", error="Unsupported archive format")
        return ToolResult(success=True, output=f"Extracted {len(names)} items to {dst}", data={"files": names[:100]})
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


# ===== MEDIA TOOLS =========================================================

@registry.register(
    "take_screenshot", "Take a screenshot of the screen", "media",
    [ToolParam("monitor", "integer", "Monitor index (0=all)", required=False, default=1)],
    danger_level=0,
)
def take_screenshot(monitor: int = 1) -> ToolResult:
    try:
        import mss as mss_lib
        with mss_lib.mss() as sct:
            if monitor >= len(sct.monitors):
                monitor = 1
            shot = sct.grab(sct.monitors[monitor])
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            mss_lib.tools.to_png(shot.rgb, shot.size, output=tmp.name)
            tmp.close()
            return ToolResult(
                success=True,
                output=f"Screenshot saved: {tmp.name}",
                data={"path": tmp.name, "width": shot.width, "height": shot.height},
            )
    except Exception as exc:
        return ToolResult(success=False, output="", error=f"Screenshot failed: {exc}")
