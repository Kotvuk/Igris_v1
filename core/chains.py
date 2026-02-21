"""Command Chaining - parse and execute multi-step commands."""

import asyncio
import re
import json
import time
import uuid
import logging
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("igris.chains")


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@dataclass
class ChainStep:
    index: int
    description: str
    tool_name: str
    tool_args: Dict[str, Any]
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    depends_on: Optional[int] = None
    rollback_tool: Optional[str] = None
    rollback_args: Optional[Dict[str, Any]] = None


@dataclass
class Chain:
    id: str
    original_command: str
    steps: List[ChainStep]
    created_at: float
    status: str = "planning"  # planning, confirming, executing, completed, failed, cancelled


class ChainExecutor:
    """Parses multi-step commands and executes them as chains with rollback."""

    def __init__(
        self,
        tool_registry: Dict[str, Callable],
        on_step_complete: Optional[Callable[[Chain, ChainStep], None]] = None,
        on_confirm: Optional[Callable[[Chain], asyncio.Future]] = None,
    ):
        self._tools = tool_registry
        self._on_step_complete = on_step_complete
        self._on_confirm = on_confirm
        self._active_chains: Dict[str, Chain] = {}

    async def parse_chain(self, command: str, llm_client: Any) -> Chain:
        """Use an LLM to decompose a natural-language command into chain steps.

        The llm_client must have an async `chat(messages)` method returning a string
        with a JSON array of steps.
        """
        prompt = (
            "Decompose the following user command into sequential tool-call steps. "
            "Return ONLY a JSON array where each element has: "
            '{"description": str, "tool_name": str, "tool_args": dict, '
            '"depends_on": int|null, "rollback_tool": str|null, "rollback_args": dict|null}. '
            f"Available tools: {list(self._tools.keys())}.\n\n"
            f"Command: {command}"
        )

        response = await llm_client.chat([{"role": "user", "content": prompt}])

        # Extract JSON from response (may be wrapped in markdown)
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            raise ValueError("LLM did not return a valid JSON array for chain parsing")

        raw_steps = json.loads(json_match.group())
        steps: List[ChainStep] = []
        for i, raw in enumerate(raw_steps):
            steps.append(ChainStep(
                index=i,
                description=raw.get("description", f"Step {i}"),
                tool_name=raw["tool_name"],
                tool_args=raw.get("tool_args", {}),
                depends_on=raw.get("depends_on"),
                rollback_tool=raw.get("rollback_tool"),
                rollback_args=raw.get("rollback_args"),
            ))

        chain = Chain(
            id=uuid.uuid4().hex[:12],
            original_command=command,
            steps=steps,
            created_at=time.time(),
            status="confirming",
        )
        self._active_chains[chain.id] = chain
        logger.info("Parsed chain %s with %d steps", chain.id, len(steps))
        return chain

    def display_plan(self, chain: Chain) -> str:
        """Format the chain plan as a human-readable string."""
        lines = [f"📋 Plan for: {chain.original_command}", f"   Chain ID: {chain.id}", ""]
        for step in chain.steps:
            dep = f" (after step {step.depends_on})" if step.depends_on is not None else ""
            rb = " 🔄 has rollback" if step.rollback_tool else ""
            lines.append(f"  {step.index + 1}. [{step.tool_name}] {step.description}{dep}{rb}")
        lines.append(f"\nTotal steps: {len(chain.steps)}")
        return "\n".join(lines)

    async def execute_chain(self, chain: Chain) -> Chain:
        """Execute all steps in order, handling dependencies and rollbacks."""
        # Optionally wait for user confirmation
        if self._on_confirm and chain.status == "confirming":
            confirmed = await self._on_confirm(chain)
            if not confirmed:
                chain.status = "cancelled"
                return chain

        chain.status = "executing"
        completed_steps: List[ChainStep] = []

        for step in chain.steps:
            # Check dependency
            if step.depends_on is not None:
                dep_step = chain.steps[step.depends_on]
                if dep_step.status != StepStatus.COMPLETED:
                    step.status = StepStatus.SKIPPED
                    step.result = f"Skipped: dependency step {step.depends_on} not completed"
                    logger.info("Step %d skipped (dep %d failed)", step.index, step.depends_on)
                    continue

            # Resolve tool
            tool_fn = self._tools.get(step.tool_name)
            if not tool_fn:
                step.status = StepStatus.FAILED
                step.result = f"Tool '{step.tool_name}' not found in registry"
                chain.status = "failed"
                break

            step.status = StepStatus.RUNNING
            try:
                # Inject previous step result if referenced
                args = dict(step.tool_args)
                if completed_steps and "__prev_result__" in json.dumps(args):
                    prev_result = completed_steps[-1].result or ""
                    args = json.loads(
                        json.dumps(args).replace("__prev_result__", prev_result)
                    )

                result = tool_fn(**args)
                if asyncio.iscoroutine(result):
                    result = await result

                step.result = str(result) if result is not None else "OK"
                step.status = StepStatus.COMPLETED
                completed_steps.append(step)
                logger.info("Step %d completed: %s", step.index, step.description)
            except Exception as e:
                step.status = StepStatus.FAILED
                step.result = str(e)
                logger.error("Step %d failed: %s", step.index, e)

                # Attempt rollback
                if step.rollback_tool:
                    rb_fn = self._tools.get(step.rollback_tool)
                    if rb_fn:
                        try:
                            rb_result = rb_fn(**(step.rollback_args or {}))
                            if asyncio.iscoroutine(rb_result):
                                await rb_result
                            step.status = StepStatus.ROLLED_BACK
                            logger.info("Step %d rolled back", step.index)
                        except Exception as rb_e:
                            logger.error("Rollback for step %d failed: %s", step.index, rb_e)

                chain.status = "failed"
                break

            if self._on_step_complete:
                try:
                    self._on_step_complete(chain, step)
                except Exception as e:
                    logger.error("Step complete callback error: %s", e)

        if chain.status == "executing":
            chain.status = "completed"

        logger.info("Chain %s finished with status: %s", chain.id, chain.status)
        return chain

    def cancel_chain(self, chain_id: str) -> bool:
        """Cancel an active chain. Returns True if found and cancelled."""
        chain = self._active_chains.get(chain_id)
        if not chain or chain.status in ("completed", "failed", "cancelled"):
            return False
        chain.status = "cancelled"
        for step in chain.steps:
            if step.status == StepStatus.PENDING:
                step.status = StepStatus.SKIPPED
        logger.info("Chain %s cancelled", chain_id)
        return True

    def get_active_chains(self) -> List[Chain]:
        """Return all chains that are not yet completed/failed/cancelled."""
        return [
            c for c in self._active_chains.values()
            if c.status in ("planning", "confirming", "executing")
        ]
