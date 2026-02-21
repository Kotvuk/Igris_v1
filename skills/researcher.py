"""
IGRIS Research Skill.

Deep web search, multi-source aggregation, summary generation,
source citation, and comparison tables.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger("igris.skills.researcher")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Source:
    title: str
    url: str
    snippet: str
    relevance: float = 0.0  # 0–1


@dataclass
class ResearchResult:
    query: str
    sources: list[Source] = field(default_factory=list)
    summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    comparison_table: Optional[str] = None  # markdown table


# ---------------------------------------------------------------------------
# Search providers
# ---------------------------------------------------------------------------

class BraveSearch:
    """Brave Search API wrapper."""

    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")

    async def search(self, query: str, count: int = 10, language: str = "ru") -> list[Source]:
        if not self.api_key:
            logger.warning("BRAVE_API_KEY not set, skipping Brave search")
            return []

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                self.API_URL,
                headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
                params={"q": query, "count": count, "search_lang": language},
            )
            resp.raise_for_status()
            data = resp.json()

        sources: list[Source] = []
        for item in data.get("web", {}).get("results", []):
            sources.append(Source(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                relevance=1.0 - len(sources) * 0.05,
            ))
        return sources


class DuckDuckGoSearch:
    """Fallback search using DuckDuckGo HTML (no API key needed)."""

    SEARCH_URL = "https://html.duckduckgo.com/html/"

    async def search(self, query: str, count: int = 10) -> list[Source]:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.post(
                self.SEARCH_URL,
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; IGRIS/1.0)"},
            )
            resp.raise_for_status()

        sources: list[Source] = []
        # Simple regex extraction from DDG HTML results
        blocks = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:td|div)',
            resp.text, re.DOTALL,
        )
        for url, title, snippet in blocks[:count]:
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            if url and clean_title:
                sources.append(Source(
                    title=clean_title,
                    url=url,
                    snippet=clean_snippet,
                    relevance=1.0 - len(sources) * 0.05,
                ))
        return sources


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

async def _llm(
    prompt: str,
    system: str = "You are IGRIS, an expert research analyst. Be thorough and cite sources.",
    api_key: Optional[str] = None,
    model: str = "llama-3.1-70b-versatile",
) -> str:
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
                "temperature": 0.3,
                "max_tokens": 4096,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

class Researcher:
    """
    Multi-source research engine with LLM-powered summarisation.
    """

    def __init__(
        self,
        brave_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ) -> None:
        self._brave = BraveSearch(api_key=brave_api_key)
        self._ddg = DuckDuckGoSearch()
        self._groq_key = groq_api_key

    async def research(
        self,
        topic: str,
        depth: int = 2,
        language: str = "ru",
    ) -> ResearchResult:
        """
        Perform deep research on a topic.

        Args:
            topic: Research topic or question.
            depth: 1=quick, 2=normal, 3=deep (more queries & sources).
            language: Search language.
        """
        result = ResearchResult(query=topic)

        # Phase 1: Gather sources from multiple queries
        queries = await self._expand_queries(topic, count=depth + 1)
        all_sources: list[Source] = []

        for q in queries:
            try:
                sources = await self._brave.search(q, count=5, language=language)
            except Exception:
                sources = []
            if not sources:
                try:
                    sources = await self._ddg.search(q, count=5)
                except Exception:
                    sources = []
            all_sources.extend(sources)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        for s in all_sources:
            if s.url not in seen_urls:
                seen_urls.add(s.url)
                result.sources.append(s)

        # Phase 2: Summarise
        result.summary = await self._summarise(topic, result.sources)

        # Phase 3: Extract key findings
        result.key_findings = await self._extract_findings(topic, result.summary)

        return result

    async def compare(
        self,
        items: list[str],
        criteria: Optional[list[str]] = None,
    ) -> str:
        """
        Generate a comparison table for the given items.

        Returns a markdown table string.
        """
        criteria_str = ", ".join(criteria) if criteria else "auto-detect relevant criteria"
        prompt = (
            f"Create a detailed comparison table in markdown format for: {', '.join(items)}.\n"
            f"Criteria: {criteria_str}.\n"
            f"Use | header | format. Be factual and concise."
        )
        return await _llm(prompt, api_key=self._groq_key)

    # -- internal ------------------------------------------------------------

    async def _expand_queries(self, topic: str, count: int) -> list[str]:
        """Generate search queries from a topic."""
        if count <= 1:
            return [topic]

        prompt = (
            f"Generate {count} different search queries to research this topic thoroughly:\n"
            f'"{topic}"\n\n'
            f"Return only the queries, one per line, no numbering."
        )
        try:
            raw = await _llm(prompt, api_key=self._groq_key)
            queries = [q.strip().strip('"') for q in raw.strip().split("\n") if q.strip()]
            return queries[:count] if queries else [topic]
        except Exception:
            return [topic]

    async def _summarise(self, topic: str, sources: list[Source]) -> str:
        source_text = "\n\n".join(
            f"[{i+1}] {s.title} ({s.url})\n{s.snippet}"
            for i, s in enumerate(sources[:15])
        )
        prompt = (
            f"Summarise the research on: {topic}\n\n"
            f"Sources:\n{source_text}\n\n"
            f"Write a comprehensive summary citing sources as [1], [2], etc."
        )
        return await _llm(prompt, api_key=self._groq_key)

    async def _extract_findings(self, topic: str, summary: str) -> list[str]:
        prompt = (
            f"From this research summary on '{topic}', extract 5-8 key findings as a bullet list.\n"
            f"Return only the bullets, one per line, starting with '- '.\n\n{summary}"
        )
        try:
            raw = await _llm(prompt, api_key=self._groq_key)
            return [line.strip().lstrip("- ") for line in raw.split("\n") if line.strip().startswith("-")]
        except Exception:
            return []
