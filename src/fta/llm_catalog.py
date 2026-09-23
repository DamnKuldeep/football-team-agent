"""Curated OpenRouter models to choose from, with live pricing.

Prices are fetched from OpenRouter's public model list (no key needed) and
cached for a few hours; if that fails, the fallback prices below are used.
Prices are USD per 1M tokens (input, output).
"""
from __future__ import annotations

import time

import requests

DEFAULT_MODEL = "deepseek/deepseek-chat"
_MODELS_URL = "https://openrouter.ai/api/v1/models"
_TTL_SECONDS = 6 * 3600

# (id, label, note, fallback $/1M in, fallback $/1M out) -- one per budget tier
CURATED = [
    ("deepseek/deepseek-chat", "DeepSeek V3",
     "Budget · reads requests very well (100% recall in the eval); assistant jobs take 20–40 s", 0.32, 0.89),
    ("google/gemini-3.8-flash", "Gemini 3.8 Flash", "Balanced · the most thorough assistant answers", 0.75, 3.75),
    ("anthropic/claude-sonnet-5", "Claude Sonnet 5", "Premium · best writing and judgement", 2.0, 10.0),
]

# typical (input, output) tokens per action; an assistant job resends its growing
# conversation (tool results included) on every step, so input dominates
TYPICAL_TOKENS = {"read a request": (2500, 300), "assistant: changes for a goal": (22000, 600),
                  "assistant: match analysis": (12000, 1000), "assistant: scouting": (9000, 600),
                  "assistant: team review": (25000, 900)}

_cache: dict = {"at": 0.0, "prices": {}}


def _live_prices() -> dict[str, tuple[float, float]]:
    if time.time() - _cache["at"] < _TTL_SECONDS:
        return _cache["prices"]
    try:
        data = requests.get(_MODELS_URL, timeout=8).json()["data"]
        prices = {m["id"]: (float(m["pricing"]["prompt"]) * 1e6, float(m["pricing"]["completion"]) * 1e6)
                  for m in data if "pricing" in m}
    except (requests.RequestException, KeyError, ValueError):
        prices = {}
    _cache.update(at=time.time(), prices=prices)
    return prices


def catalog() -> list[dict]:
    """Curated models with current prices; `listed` is False if OpenRouter no longer has it."""
    live = _live_prices()
    out = []
    for model_id, label, note, fin, fout in CURATED:
        pin, pout = live.get(model_id, (fin, fout))
        out.append({"id": model_id, "label": label, "note": note, "in": pin, "out": pout,
                    "listed": model_id in live if live else None})
    return out


def price(model_id: str) -> tuple[float, float]:
    """(USD per 1M input tokens, per 1M output tokens). Unknown models get a
    conservative estimate rather than zero."""
    live = _live_prices()
    if model_id in live:
        return live[model_id]
    for mid, _, _, fin, fout in CURATED:
        if mid == model_id:
            return fin, fout
    return 1.0, 5.0


def estimate(model_id: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = price(model_id)
    return tokens_in / 1e6 * pin + tokens_out / 1e6 * pout


def describe(model_id: str) -> str:
    """e.g. 'DeepSeek V3 · $0.32 / $0.89 per 1M tokens'"""
    entry = next((m for m in catalog() if m["id"] == model_id), None)
    label = entry["label"] if entry else model_id
    pin, pout = (entry["in"], entry["out"]) if entry else price(model_id)
    cost = "free" if pin == pout == 0 else f"${pin:.2f} / ${pout:.2f} per 1M tokens"
    return f"{label} · {cost}"
