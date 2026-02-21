"""
IGRIS Memory System — Short-term conversation history, long-term SQLite storage,
episode logging, user profile learning, and auto-summarization.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("igris.memory")


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    model: Optional[str] = None
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """Sliding-window conversation history."""

    def __init__(self, max_messages: int = 50, max_tokens: int = 32000):
        self._messages: deque[Message] = deque(maxlen=max_messages)
        self._max_tokens = max_tokens
        self._session_id: str = f"session_{int(time.time())}"

    @property
    def session_id(self) -> str:
        return self._session_id

    def add(self, role: str, content: str, **kwargs: Any) -> Message:
        """Add a message to the conversation."""
        msg = Message(role=role, content=content, **kwargs)
        self._messages.append(msg)
        self._trim_to_token_limit()
        return msg

    def _trim_to_token_limit(self) -> None:
        """Remove oldest messages to stay within token budget."""
        total = sum(m.tokens or (len(m.content) // 4) for m in self._messages)
        while total > self._max_tokens and len(self._messages) > 2:
            removed = self._messages.popleft()
            total -= removed.tokens or (len(removed.content) // 4)

    def get_messages(self, limit: Optional[int] = None) -> list[dict[str, str]]:
        """Get conversation history as list of role/content dicts."""
        msgs = list(self._messages)
        if limit:
            msgs = msgs[-limit:]
        return [{"role": m.role, "content": m.content} for m in msgs]

    def get_context_window(self, system_prompt: str) -> list[dict[str, str]]:
        """Build a full context window with system prompt."""
        return [{"role": "system", "content": system_prompt}] + self.get_messages()

    def clear(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


class LongTermMemory:
    """SQLite-backed persistent memory with semantic search readiness."""

    def __init__(self, db: Any):
        self._db = db

    async def store(
        self,
        category: str,
        key: str,
        value: str,
        importance: float = 0.5,
        embedding: Optional[bytes] = None,
    ) -> int:
        """Store a memory entry."""
        now = time.time()
        existing = await self._db.fetch_one(
            "SELECT id FROM memory WHERE category = ? AND key = ?",
            (category, key),
        )
        if existing:
            await self._db.execute(
                "UPDATE memory SET value = ?, importance = ?, embedding = ?, "
                "updated_at = ?, access_count = access_count + 1 WHERE id = ?",
                (value, importance, embedding, now, existing["id"]),
            )
            return existing["id"]

        return await self._db.insert("memory", {
            "category": category,
            "key": key,
            "value": value,
            "embedding": embedding,
            "importance": importance,
            "access_count": 0,
            "last_accessed": now,
            "created_at": now,
            "updated_at": now,
        })

    async def recall(self, category: str, key: Optional[str] = None, limit: int = 10) -> list[dict]:
        """Recall memories by category and optional key pattern."""
        if key:
            return await self._db.fetch_all(
                "SELECT * FROM memory WHERE category = ? AND key LIKE ? "
                "ORDER BY importance DESC, access_count DESC LIMIT ?",
                (category, f"%{key}%", limit),
            )
        return await self._db.fetch_all(
            "SELECT * FROM memory WHERE category = ? "
            "ORDER BY importance DESC, access_count DESC LIMIT ?",
            (category, limit),
        )

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across all memories."""
        return await self._db.fetch_all(
            "SELECT * FROM memory WHERE value LIKE ? OR key LIKE ? "
            "ORDER BY importance DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        )

    async def forget(self, memory_id: int) -> None:
        """Remove a specific memory."""
        await self._db.execute("DELETE FROM memory WHERE id = ?", (memory_id,))

    async def decay(self, max_age_days: int = 90, min_importance: float = 0.3) -> int:
        """Remove old low-importance memories."""
        cutoff = time.time() - (max_age_days * 86400)
        cursor = await self._db.execute(
            "DELETE FROM memory WHERE created_at < ? AND importance < ? AND access_count < 3",
            (cutoff, min_importance),
        )
        return cursor.rowcount


class EpisodeLogger:
    """Log actions taken during a session for replay and learning."""

    def __init__(self, db: Any, session_id: str):
        self._db = db
        self._session_id = session_id

    async def log(
        self,
        action: str,
        tool_name: Optional[str] = None,
        input_summary: Optional[str] = None,
        output_summary: Optional[str] = None,
        success: bool = True,
        duration_ms: float = 0,
    ) -> int:
        """Log an episode."""
        return await self._db.insert("episodes", {
            "session_id": self._session_id,
            "action": action,
            "tool_name": tool_name,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "success": 1 if success else 0,
            "duration_ms": duration_ms,
            "created_at": time.time(),
        })

    async def get_recent(self, limit: int = 20) -> list[dict]:
        """Get recent episodes from current session."""
        return await self._db.fetch_all(
            "SELECT * FROM episodes WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (self._session_id, limit),
        )

    async def get_session_summary(self) -> dict[str, Any]:
        """Summarize the current session's activity."""
        episodes = await self._db.fetch_all(
            "SELECT action, tool_name, success, duration_ms FROM episodes WHERE session_id = ?",
            (self._session_id,),
        )
        total = len(episodes)
        successful = sum(1 for e in episodes if e["success"])
        tools_used = {}
        for e in episodes:
            tn = e["tool_name"] or "unknown"
            tools_used[tn] = tools_used.get(tn, 0) + 1

        return {
            "session_id": self._session_id,
            "total_actions": total,
            "successful": successful,
            "failed": total - successful,
            "tools_used": tools_used,
            "total_duration_ms": sum(e["duration_ms"] for e in episodes),
        }


class UserProfile:
    """Learn and track user preferences over time."""

    def __init__(self, db: Any):
        self._db = db

    async def set(self, key: str, value: str, confidence: float = 0.5, source: str = "inferred") -> None:
        """Set a user preference."""
        existing = await self._db.fetch_one(
            "SELECT id, confidence FROM user_profile WHERE key = ?", (key,)
        )
        now = time.time()
        if existing:
            new_confidence = min(1.0, max(existing["confidence"], confidence))
            await self._db.execute(
                "UPDATE user_profile SET value = ?, confidence = ?, source = ?, updated_at = ? WHERE id = ?",
                (value, new_confidence, source, now, existing["id"]),
            )
        else:
            await self._db.insert("user_profile", {
                "key": key,
                "value": value,
                "confidence": confidence,
                "source": source,
                "updated_at": now,
            })

    async def get(self, key: str) -> Optional[str]:
        """Get a user preference value."""
        row = await self._db.fetch_one(
            "SELECT value FROM user_profile WHERE key = ?", (key,)
        )
        return row["value"] if row else None

    async def get_all(self) -> dict[str, Any]:
        """Get all known preferences."""
        rows = await self._db.fetch_all(
            "SELECT key, value, confidence, source FROM user_profile ORDER BY confidence DESC"
        )
        return {r["key"]: {"value": r["value"], "confidence": r["confidence"], "source": r["source"]} for r in rows}

    async def learn_from_message(self, message: str) -> None:
        """Extract preferences from user messages (basic heuristics)."""
        lower = message.lower()
        if "prefer" in lower or "always" in lower or "i like" in lower:
            await self.set("last_preference_hint", message, confidence=0.4, source="message_analysis")
        if any(lang in lower for lang in ["python", "javascript", "typescript", "rust", "go", "java"]):
            for lang in ["python", "javascript", "typescript", "rust", "go", "java"]:
                if lang in lower:
                    await self.set("preferred_language", lang, confidence=0.6, source="message_analysis")
                    break


class ConversationSummarizer:
    """Auto-summarize old conversations to save context space."""

    def __init__(self, llm_client: Any, db: Any):
        self._llm = llm_client
        self._db = db

    async def summarize_and_archive(self, messages: list[dict[str, str]], session_id: str) -> str:
        """Summarize a conversation and store it in long-term memory."""
        if len(messages) < 4:
            return ""

        conversation_text = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in messages)
        summary_prompt = [
            {"role": "system", "content": "Summarize this conversation concisely. Focus on: decisions made, tasks completed, user preferences learned, important context. Max 200 words."},
            {"role": "user", "content": conversation_text[:8000]},
        ]

        try:
            resp = await self._llm.chat(summary_prompt, max_tokens=300, temperature=0.3)
            summary = resp["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Failed to summarize conversation: {e}")
            summary = f"Session {session_id}: {len(messages)} messages exchanged."

        ltm = LongTermMemory(self._db)
        await ltm.store(
            category="conversation_summary",
            key=session_id,
            value=summary,
            importance=0.6,
        )
        return summary


class MemoryManager:
    """Unified memory interface combining all memory subsystems."""

    def __init__(self, db: Any, llm_client: Optional[Any] = None):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(db)
        self.episodes = EpisodeLogger(db, self.short_term.session_id)
        self.profile = UserProfile(db)
        self._summarizer = ConversationSummarizer(llm_client, db) if llm_client else None
        self._db = db

    async def add_message(self, role: str, content: str, **kwargs: Any) -> Message:
        """Add a message and persist it."""
        msg = self.short_term.add(role, content, **kwargs)
        await self._db.insert("messages", {
            "role": role,
            "content": content,
            "model": kwargs.get("model"),
            "tokens_in": kwargs.get("tokens_in", 0),
            "tokens_out": kwargs.get("tokens_out", 0),
            "latency_ms": kwargs.get("latency_ms", 0),
            "session_id": self.short_term.session_id,
            "created_at": msg.timestamp,
        })
        if role == "user":
            await self.profile.learn_from_message(content)
        return msg

    async def end_session(self) -> Optional[str]:
        """Summarize and archive the current session."""
        if self._summarizer and len(self.short_term) > 4:
            messages = self.short_term.get_messages()
            return await self._summarizer.summarize_and_archive(
                messages, self.short_term.session_id
            )
        return None

    async def get_relevant_context(self, query: str, limit: int = 5) -> list[dict]:
        """Retrieve relevant long-term memories for context injection."""
        return await self.long_term.search(query, limit=limit)
