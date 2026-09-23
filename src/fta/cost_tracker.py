"""Append-only cost ledger for every LLM call (request reading and the
assistant), so the spend is always visible."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "costs.jsonl"


@dataclass
class CostTracker:
    budget_usd: float = 5.0
    mode: str = "warn"  # "warn" or "block" (no AI calls once the budget is spent)
    _total_usd: float = 0.0

    def __post_init__(self):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists():
            for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._total_usd += json.loads(line).get("est_cost_usd", 0.0)

    @property
    def total_usd(self) -> float:
        return round(self._total_usd, 6)

    @property
    def blocked(self) -> bool:
        """In "block" mode, no further AI calls once the budget is spent (checked before each call)."""
        return self.mode == "block" and self._total_usd >= self.budget_usd

    @property
    def block_reason(self) -> str:
        return f"the AI spending limit (${self.budget_usd:.2f}) has been reached"

    def estimate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Uses OpenRouter's current price for the model (see llm_catalog)."""
        from .llm_catalog import estimate
        return estimate(model, tokens_in, tokens_out)

    def log_call(self, step: str, model: str, tokens_in: int, tokens_out: int) -> float:
        cost = self.estimate_cost(model, tokens_in, tokens_out)
        self._total_usd += cost
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": step, "model": model,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "est_cost_usd": round(cost, 6),
            "running_total_usd": self.total_usd,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if self._total_usd > self.budget_usd:
            print(f"[cost_tracker] WARNING: budget of ${self.budget_usd:.2f} exceeded "
                  f"(running total ${self.total_usd:.4f}) after step '{step}'.")
        return cost

    def summary(self) -> str:
        return f"Total spend so far: ${self.total_usd:.4f} / ${self.budget_usd:.2f} budget"
