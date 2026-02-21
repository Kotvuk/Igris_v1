"""Snippet Library - save and search reusable code/commands."""

import json
import os
import re
import time
import uuid
import hashlib
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger("igris.snippets")

# Patterns for auto-detecting language from content
_LANG_PATTERNS = [
    ("python", [r'\bdef \w+\(', r'\bimport \w+', r'\bclass \w+:', r'print\(']),
    ("bash", [r'^#!/bin/(ba)?sh', r'\bsudo\b', r'\bapt\b', r'\becho\b', r'\|\s*grep\b']),
    ("javascript", [r'\bconst \w+\s*=', r'\bfunction\b', r'=>', r'console\.log']),
    ("sql", [r'\bSELECT\b', r'\bFROM\b', r'\bINSERT\b', r'\bCREATE TABLE\b']),
    ("html", [r'<html', r'<div', r'<!DOCTYPE']),
    ("css", [r'\{[^}]*:[^}]*;\s*\}', r'\.[\w-]+\s*\{', r'@media']),
    ("rust", [r'\bfn \w+', r'\blet mut\b', r'\bimpl\b']),
    ("go", [r'\bfunc \w+', r'\bpackage \w+', r':=\s']),
]


@dataclass
class Snippet:
    id: str
    title: str
    content: str
    language: str
    tags: List[str]
    source: str
    created_at: float
    used_count: int = 0
    last_used_at: Optional[float] = None


class SnippetLibrary:
    """Persistent snippet library backed by a JSON file."""

    def __init__(self, path: str = "~/.igris/snippets.json"):
        self._path = os.path.expanduser(path)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._snippets: Dict[str, Snippet] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for entry in data:
                s = Snippet(**entry)
                self._snippets[s.id] = s
            logger.info("Loaded %d snippets", len(self._snippets))
        except (json.JSONDecodeError, IOError, TypeError) as e:
            logger.error("Failed to load snippets: %s", e)

    def _save(self) -> None:
        try:
            data = [asdict(s) for s in self._snippets.values()]
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error("Failed to save snippets: %s", e)

    @staticmethod
    def _detect_language(content: str) -> str:
        for lang, patterns in _LANG_PATTERNS:
            matches = sum(1 for p in patterns if re.search(p, content, re.MULTILINE | re.IGNORECASE))
            if matches >= 2:
                return lang
        return "text"

    def save(
        self,
        title: str,
        content: str,
        language: str = "",
        tags: Optional[List[str]] = None,
        source: str = "manual",
    ) -> Snippet:
        """Save a new snippet. Auto-detects language if not provided."""
        if not language:
            language = self._detect_language(content)

        snippet = Snippet(
            id=hashlib.sha256(f"{title}{content}{time.time()}".encode()).hexdigest()[:12],
            title=title,
            content=content,
            language=language,
            tags=tags or [],
            source=source,
            created_at=time.time(),
        )
        self._snippets[snippet.id] = snippet
        self._save()
        logger.info("Saved snippet %s: %s", snippet.id, title)
        return snippet

    def search(self, query: str) -> List[Snippet]:
        """Search snippets by title, content, and tags. Ranked by relevance."""
        query_lower = query.lower()
        scored: List[tuple] = []

        for s in self._snippets.values():
            score = 0
            # Exact title match
            if query_lower == s.title.lower():
                score += 100
            elif query_lower in s.title.lower():
                score += 50
            # Tag match
            for tag in s.tags:
                if query_lower in tag.lower():
                    score += 30
            # Content match
            if query_lower in s.content.lower():
                score += 10
            # Boost by usage
            score += min(s.used_count, 10)

            if score > 0:
                scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]

    def get_by_tag(self, tag: str) -> List[Snippet]:
        """Return all snippets that have the given tag."""
        tag_lower = tag.lower()
        return [s for s in self._snippets.values() if tag_lower in [t.lower() for t in s.tags]]

    def get_by_language(self, language: str) -> List[Snippet]:
        """Return all snippets of the given language."""
        lang_lower = language.lower()
        return [s for s in self._snippets.values() if s.language.lower() == lang_lower]

    def get_recent(self, limit: int = 10) -> List[Snippet]:
        """Return the most recently created snippets."""
        all_sorted = sorted(self._snippets.values(), key=lambda s: s.created_at, reverse=True)
        return all_sorted[:limit]

    def get_most_used(self, limit: int = 10) -> List[Snippet]:
        """Return snippets sorted by usage count descending."""
        all_sorted = sorted(self._snippets.values(), key=lambda s: s.used_count, reverse=True)
        return all_sorted[:limit]

    def delete(self, snippet_id: str) -> bool:
        """Delete a snippet by ID. Returns True if found and deleted."""
        if snippet_id in self._snippets:
            del self._snippets[snippet_id]
            self._save()
            logger.info("Deleted snippet %s", snippet_id)
            return True
        return False

    def use(self, snippet_id: str) -> Optional[Snippet]:
        """Mark a snippet as used. Increments count and updates timestamp."""
        s = self._snippets.get(snippet_id)
        if not s:
            return None
        s.used_count += 1
        s.last_used_at = time.time()
        self._save()
        return s

    def export_all(self) -> Dict:
        """Export all snippets as a dict for backup."""
        return {
            "version": 1,
            "exported_at": time.time(),
            "snippets": [asdict(s) for s in self._snippets.values()],
        }

    def import_snippets(self, data: Dict) -> int:
        """Import snippets from exported data. Returns count of imported snippets."""
        imported = 0
        for entry in data.get("snippets", []):
            try:
                s = Snippet(**entry)
                if s.id not in self._snippets:
                    self._snippets[s.id] = s
                    imported += 1
            except (TypeError, KeyError) as e:
                logger.warning("Skipped invalid snippet during import: %s", e)
        if imported:
            self._save()
        logger.info("Imported %d snippets", imported)
        return imported
