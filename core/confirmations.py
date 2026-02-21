"""
IGRIS Confirmation System — Strict/Medium/Trust modes with async confirmation flow.
"""

import asyncio
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("igris.confirmations")


class ConfirmationMode(Enum):
    STRICT = "strict"    # Everything requires confirmation
    MEDIUM = "medium"    # Only dangerous actions
    TRUST = "trust"      # Only critical/destructive actions


class ActionCategory(Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_MOVE = "file_move"
    FILE_RENAME = "file_rename"
    COMMAND_EXEC = "command_exec"
    NETWORK_REQUEST = "network_request"
    SYSTEM_CHANGE = "system_change"
    ELEVATED_ACTION = "elevated_action"
    CRITICAL_ACTION = "critical_action"


# Which categories need confirmation in each mode
CONFIRMATION_RULES: dict[ConfirmationMode, set[ActionCategory]] = {
    ConfirmationMode.STRICT: set(ActionCategory),
    ConfirmationMode.MEDIUM: {
        ActionCategory.FILE_DELETE,
        ActionCategory.FILE_MOVE,
        ActionCategory.FILE_RENAME,
        ActionCategory.COMMAND_EXEC,
        ActionCategory.NETWORK_REQUEST,
        ActionCategory.SYSTEM_CHANGE,
        ActionCategory.ELEVATED_ACTION,
        ActionCategory.CRITICAL_ACTION,
    },
    ConfirmationMode.TRUST: {
        ActionCategory.CRITICAL_ACTION,
        ActionCategory.ELEVATED_ACTION,
        ActionCategory.SYSTEM_CHANGE,
    },
}


@dataclass
class ConfirmationRequest:
    """A pending confirmation request."""
    request_id: str
    action: str
    category: ActionCategory
    description: str
    details: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    approved: Optional[bool] = None
    resolved_at: Optional[float] = None


class ConfirmationManager:
    """Manages the confirmation flow for dangerous or sensitive actions."""

    DEFAULT_TIMEOUT = 60.0  # seconds

    def __init__(
        self,
        mode: ConfirmationMode = ConfirmationMode.MEDIUM,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._mode = mode
        self._timeout = timeout
        self._pending: dict[str, ConfirmationRequest] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._counter = 0
        self._handler: Optional[Callable] = None
        self._history: list[ConfirmationRequest] = []

    @property
    def mode(self) -> ConfirmationMode:
        return self._mode

    @mode.setter
    def mode(self, value: ConfirmationMode) -> None:
        self._mode = value
        logger.info(f"Confirmation mode changed to: {value.value}")

    def set_handler(self, handler: Callable) -> None:
        """Set the callback that presents confirmations to the user (e.g., WebSocket push)."""
        self._handler = handler

    def needs_confirmation(self, category: ActionCategory) -> bool:
        """Check if an action category requires confirmation under current mode."""
        return category in CONFIRMATION_RULES.get(self._mode, set())

    async def request(
        self,
        action: str,
        category: ActionCategory,
        description: str,
        details: Optional[str] = None,
    ) -> bool:
        """Request user confirmation for an action. Returns True if approved."""
        if not self.needs_confirmation(category):
            return True

        self._counter += 1
        req_id = f"confirm_{self._counter}_{int(time.time())}"
        req = ConfirmationRequest(
            request_id=req_id,
            action=action,
            category=category,
            description=description,
            details=details,
        )
        self._pending[req_id] = req
        event = asyncio.Event()
        self._events[req_id] = event

        if self._handler:
            try:
                self._handler({
                    "type": "confirmation_request",
                    "request_id": req_id,
                    "action": action,
                    "category": category.value,
                    "description": description,
                    "details": details,
                })
            except Exception as e:
                logger.error(f"Confirmation handler error: {e}")

        logger.info(f"Awaiting confirmation: {description} [{req_id}]")

        try:
            await asyncio.wait_for(event.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            req.resolved = True
            req.approved = False
            req.resolved_at = time.time()
            logger.warning(f"Confirmation timed out: {req_id}")
            self._cleanup(req_id)
            return False

        result = req.approved or False
        self._cleanup(req_id)
        return result

    def resolve(self, request_id: str, approved: bool) -> bool:
        """Resolve a pending confirmation (called by UI/WebSocket handler)."""
        req = self._pending.get(request_id)
        if not req or req.resolved:
            return False

        req.resolved = True
        req.approved = approved
        req.resolved_at = time.time()

        event = self._events.get(request_id)
        if event:
            event.set()

        action_word = "approved" if approved else "denied"
        logger.info(f"Confirmation {action_word}: {request_id}")
        return True

    def _cleanup(self, request_id: str) -> None:
        req = self._pending.pop(request_id, None)
        self._events.pop(request_id, None)
        if req:
            self._history.append(req)
            if len(self._history) > 200:
                self._history = self._history[-100:]

    def get_pending(self) -> list[dict[str, Any]]:
        """Get all pending confirmation requests."""
        return [
            {
                "request_id": r.request_id,
                "action": r.action,
                "category": r.category.value,
                "description": r.description,
                "age_seconds": int(time.time() - r.created_at),
            }
            for r in self._pending.values()
            if not r.resolved
        ]

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent confirmation history."""
        return [
            {
                "request_id": r.request_id,
                "action": r.action,
                "approved": r.approved,
                "category": r.category.value,
            }
            for r in self._history[-limit:]
        ]
