"""Talking to the model (OpenRouter), and reading change requests.

  * chat_message() / _call_json() -- the transport: JSON mode, tool calls,
    rate-limit backoff, one corrective retry for unusable replies;
  * parse_brief() -- a request about one slot -> weight changes + requirements.

The tool-using assistant (recommendations, match analysis, scouting, team
review) lives in agent.py and uses the same transport. Selection, ranking,
simulation and ratings are always plain code; the model proposes and explains,
and the code verifies.

Safety net for parse_brief, since it's the one place the LLM's output feeds
a calculation:
  * context: the line-up, the current player's full profile, the slot's
    neighbours, the valid attributes and filter fields, and the nationalities
    and ages that exist in the pool -- so references like "same nationality
    as our CB_R" or "younger than him" resolve instead of being guessed;
  * validation: unknown attributes/fields dropped, weights clamped to 0.05-0.5,
    nationalities must exist in the pool, unchanged weights ignored;
  * grounding: every change must quote words that are in the request, and a
    requirement must be asked for (a number, a comparison, "natural ...");
  * the keyword parser runs alongside to flag anything the AI may have missed;
  * the human sees what was sent, the raw reply and every rejection, and can
    edit everything before the shortlist is built.
Every call asks for JSON, retries rate limits with backoff, and retries once
with a correction if the reply isn't a usable JSON object.

If OPENROUTER_API_KEY isn't set, or a call fails, every function falls back
to a deterministic offline version (clearly labelled, with the reason) so the
pipeline always runs.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

from .cost_tracker import CostTracker
from .llm_catalog import DEFAULT_MODEL
from .models import GK_ONLY

WEIGHT_MIN, WEIGHT_MAX = 0.05, 0.5
KEYWORD_WEIGHT = 0.35


# attribute -> what it means (sent to the model so it maps words to the right attribute)
GLOSSARY = {
    "pace": "top speed", "acceleration": "burst over the first few metres",
    "stamina": "endurance, work rate, lasting 90 minutes",
    "passing_short": "short/simple passing, keeping the ball",
    "passing_long": "long balls, switches of play",
    "through_ball": "defence-splitting passes, playmaking",
    "pass_power_bullet": "driven/bullet passes", "crossing": "crosses from wide",
    "finishing": "shooting, scoring goals", "heading_accuracy": "aerial ability, winning and placing headers",
    "heading_power": "power in the air, jumping and strength", "dribbling": "beating players 1v1",
    "ball_control": "first touch, close control", "vision": "seeing passes, creativity",
    "composure": "calm under pressure", "aggression": "physicality, pressing intensity",
    "tackling_standing": "standing tackles, winning the ball", "tackling_sliding": "sliding tackles",
    "marking": "tracking and marking opponents", "positioning": "reading the game, being in the right place",
    "gk_reflexes": "shot-stopping reflexes", "gk_handling": "catching, handling",
    "gk_positioning": "keeper positioning, sweeping", "gk_kicking": "keeper distribution",
}
OUTFIELD_ATTRS = [a for a in GLOSSARY if a not in GK_ONLY]
# a keeper request may touch goalkeeping plus "good with his feet"/"calm" attributes, nothing else
GK_ATTRS = sorted(GK_ONLY) + ["composure", "passing_short", "passing_long"]
VALID_ATTRS = OUTFIELD_ATTRS  # kept for backwards compatibility

# deterministic fallback + cross-check: phrase -> attributes it implies
KEYWORDS: dict[str, list[str]] = {
    "aerial": ["heading_accuracy", "heading_power"], "header": ["heading_accuracy"],
    "heading": ["heading_accuracy"], "in the air": ["heading_accuracy", "heading_power"],
    "tackl": ["tackling_standing"], "sliding": ["tackling_sliding"], "ball-win": ["tackling_standing"],
    "ball win": ["tackling_standing"], "wins the ball": ["tackling_standing"],
    "win the ball": ["tackling_standing"], "stopper": ["marking"],
    "defend": ["tackling_standing", "marking", "positioning"], "mark": ["marking"],
    "reads the game": ["positioning"], "positioning": ["positioning"],
    "long ball": ["passing_long"], "long pass": ["passing_long"], "switch": ["passing_long"],
    "through ball": ["through_ball"], "through-ball": ["through_ball"],
    "playmak": ["vision", "through_ball", "passing_short"], "creativ": ["vision", "through_ball"],
    "vision": ["vision"], "bullet pass": ["pass_power_bullet"], "driven pass": ["pass_power_bullet"],
    "pass": ["passing_short"], "cross": ["crossing"],
    "finish": ["finishing"], "clinical": ["finishing", "composure"], "goal": ["finishing"],
    "scor": ["finishing"], "composed": ["composure"], "calm": ["composure"], "composure": ["composure"],
    "pace": ["pace"], "speed": ["pace"], "fast": ["pace", "acceleration"], "quick": ["pace", "acceleration"],
    "rapid": ["pace"], "acceleration": ["acceleration"], "explosive": ["acceleration"],
    "dribbl": ["dribbling"], "flair": ["dribbling"], "1v1": ["dribbling"],
    "control": ["ball_control"], "first touch": ["ball_control"],
    "stamina": ["stamina"], "work rate": ["stamina"], "engine": ["stamina"], "tireless": ["stamina"],
    "physical": ["aggression", "heading_power"], "aggress": ["aggression"], "press": ["aggression", "stamina"],
    "reflex": ["gk_reflexes"], "shot stop": ["gk_reflexes"], "shot-stop": ["gk_reflexes"],
    "handling": ["gk_handling"], "distribution": ["gk_kicking"], "kicking": ["gk_kicking"],
    "sweeper": ["gk_positioning"], "command": ["gk_positioning"],
}


def _api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY")


def _base_url() -> str:
    return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def chat_message(model: str, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 800,
                 json_mode: bool = False, tool_choice: str = "auto") -> tuple[dict, int, int]:
    """One OpenRouter chat call. Returns (assistant message, tokens_in, tokens_out);
    the message may carry `tool_calls` (tool_choice "none" forces a plain answer).
    Rate limits and provider errors are retried with a short backoff; if a
    provider rejects JSON mode, it's dropped."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    if tools:
        body["tools"], body["tool_choice"] = tools, tool_choice
    elif json_mode:
        body["response_format"] = {"type": "json_object"}
    for wait in (2, 5, None):
        resp = requests.post(
            f"{_base_url()}/chat/completions",
            headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
            json=body, timeout=90,
        )
        if resp.status_code not in (429, 500, 502, 503, 504) or wait is None:
            break
        time.sleep(wait)
    if json_mode and not tools and resp.status_code == 400:
        return chat_message(model, messages, None, max_tokens, json_mode=False)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return data["choices"][0]["message"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _chat(model: str, messages: list[dict], max_tokens: int = 500, json_mode: bool = True
          ) -> tuple[str, int, int]:
    """Plain (no tools) call returning (text, tokens_in, tokens_out)."""
    message, tin, tout = chat_message(model, messages, None, max_tokens, json_mode)
    return message.get("content") or "", tin, tout


def _call_json(task: str, model: str, system: str, user: str, max_tokens: int,
               tracker: CostTracker | None) -> tuple[dict | None, str, str]:
    """(parsed JSON object or None, model, raw reply). Never raises. If the
    reply isn't a JSON object, tells the model what was wrong and retries once.
    When it gives up, the reason is kept in `last_error` for the UI to show."""
    global last_error
    last_error = ""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = ""
    for _ in range(2):
        if tracker and tracker.blocked:
            last_error = tracker.block_reason
            return None, model, text
        try:
            text, tin, tout = _chat(model, messages, max_tokens=max_tokens)
        except (requests.RequestException, KeyError, IndexError) as e:
            last_error = _friendly(e)
            print(f"[llm_client] {task} call failed ({e}); using offline fallback")
            return None, model, ""
        if tracker:
            tracker.log_call(task, model, tin, tout)
        cleaned = (text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed, model, text
            problem = "the reply must be a single JSON object"
        except json.JSONDecodeError as e:
            problem = f"it isn't valid JSON ({e})"
        messages += [{"role": "assistant", "content": text or ""},
                     {"role": "user", "content": f"Your reply can't be used: {problem}. Reply again with ONLY "
                                                 "the JSON object described in the instructions."}]
    last_error = "its reply wasn't usable JSON, even after a retry"
    print(f"[llm_client] {task} reply unusable after a retry; using offline fallback")
    return None, model, text


last_error = ""


def _friendly(e: Exception) -> str:
    code = getattr(getattr(e, "response", None), "status_code", None)
    return {401: "the API key was rejected", 402: "the OpenRouter account is out of credit",
            404: "the model isn't available on OpenRouter any more",
            429: "the model is rate-limited right now — try again in a minute"}.get(
        code, f"the request failed ({code})" if code else "OpenRouter couldn't be reached")


# ---------------------------------------------------------------------------
# 1. NL request -> weight changes + filters
# ---------------------------------------------------------------------------

POSITIONS = ["GK", "CB", "FB", "DM", "CM", "AM", "WING", "ST"]
# filter field -> (min, max) for numeric fields; None for categorical ones
FILTER_FIELDS: dict[str, tuple[float, float] | None] = {
    "age": (15, 45), "weak_foot_rating": (1, 5), "nationality": None, "natural_position": None,
    **{a: (0, 100) for a in GLOSSARY},
}

PARSE_SYSTEM_PROMPT = """You turn a football coach's request about ONE position in his team into
(a) attribute weight changes and (b) requirements ("filters") for the replacement.

The user message is JSON with:
- request: the coach's words;
- slot, position, role, formation: where the new player will play;
- current_weights: what this position values now (shares that sum to 1);
- current_player: who plays there now (age, foot, weak foot 1-5, nationality, natural positions, attributes 0-100);
- lineup: every slot of the team with its player's name, age, foot and nationality;
- neighbours: the slots next to this one on the pitch;
- allowed_attributes: attribute -> meaning. These are the ONLY attribute names you may use;
- allowed_filters: field -> allowed values or range. These are the ONLY filter fields you may use;
- pool: the nationalities and age range that exist among available players.

Rules:
1. Only change what the request asks for. A request for "a better / stronger / an upgrade at <position>"
   that names no quality needs NO adjustments: ranking on the current weights already finds the best fit.
2. List ONLY attributes whose weight should change. "weight" is the new share, 0.05-0.5 (0.3-0.4 for a clear
   emphasis). Never restate unchanged weights.
3. Resolve references to people with lineup/current_player:
   "same nationality as our right centre-back" -> nationality filter with that player's nationality;
   "same nationality as at least one centre-back" -> nationality filter whose "value" is a LIST of both
   centre-backs' nationalities (any of them is fine);
   "younger than him" -> age max = current player's age - 1; "older than CB_L" -> age min = that age + 1;
   "better in the air than him" -> heading_accuracy min = his value + 1 (plus a weight if it's an emphasis).
   A nationality must be copied exactly from pool.nationalities.
4. Use filters only for hard requirements the coach states ("must", "under 25", "at least 80",
   "same ... as", "natural left back", "younger/older than", "better than him at ..."). Naming the position
   ("a full-back who ...") is NOT a natural_position requirement, and a description ("pace to burn") is a
   weight, NOT a numeric minimum: only use attribute min/max when the request gives a number or compares
   with a player.
5. Every "reason" copies, word for word, the part of the request that justifies it.
6. "hard_foot" is "left" or "right" whenever a preferred foot is named ("left-footed", "a right-footer");
   otherwise null. "two-footed" / "good weak foot" is a weak_foot_rating min of 4, not a foot.
7. Read every quality in the request: "ball-winner" = tackling, "shot-stopper" = reflexes, "commanding" (a
   keeper) = gk_positioning, "switch play" = passing_long, "wins headers" = heading_accuracy.

Examples
request "a better centre-back" -> {"adjustments": [], "filters": [], "hard_foot": null}
request "quicker than him, from the same country as our left back" (current pace 70, LB nationality BR) ->
{"adjustments": [{"attribute": "pace", "weight": 0.3, "reason": "quicker than him"}],
 "filters": [{"field": "pace", "min": 71, "max": null, "value": null, "reason": "quicker than him"},
             {"field": "nationality", "min": null, "max": null, "value": "BR",
              "reason": "from the same country as our left back"}],
 "hard_foot": null}

Respond ONLY with one JSON object of exactly this shape:
{"adjustments": [{"attribute": "...", "weight": 0.35, "reason": "..."}],
 "filters": [{"field": "...", "min": null, "max": null, "value": null, "reason": "..."}],
 "hard_foot": null}"""

_NATURAL = {"left back": "FB", "right back": "FB", "full back": "FB", "fullback": "FB", "wing back": "FB",
            "centre back": "CB", "center back": "CB", "centre-back": "CB", "center-back": "CB",
            "defensive midfielder": "DM", "holding midfielder": "DM", "central midfielder": "CM",
            "attacking midfielder": "AM", "number 10": "AM", "winger": "WING", "striker": "ST",
            "centre forward": "ST", "center forward": "ST", "goalkeeper": "GK", "keeper": "GK"}
_SELF = r"(?:him|the current (?:player|one|guy)|the incumbent|current)"


def _keyword_adjustments(brief: str, allowed: list[str]) -> dict[str, str]:
    """attribute -> the phrase that triggered it."""
    text = brief.lower()
    found: dict[str, str] = {}
    for kw, attrs in KEYWORDS.items():
        # keyword must start a word: "press" matches "pressing", not "impressive"
        if re.search(r"(?<![a-z])" + re.escape(kw), text):
            for attr in attrs:
                if attr in allowed:
                    found.setdefault(attr, f'"{kw}"')
    return found


def _keyword_foot(brief: str) -> str | None:
    """'left-footed' -> left. None when no foot, both feet, or a negation ('not left-footed') is named."""
    text = brief.lower()
    if re.search(r"\b(left or right|right or left|either foot|both feet|any foot)\b", text):
        return None
    foot = r"[- ]?(?:foot|footed|footer|sided)"
    negated = set(re.findall(rf"\b(?:not|no|non)[- ](?:an? )?(left|right){foot}", text))
    found = {side for side in ("left", "right") if re.search(side + foot, text)} - negated
    return found.pop() if len(found) == 1 else None


def _find_person(phrase: str, context: dict) -> dict | None:
    """Resolve 'him', a slot id ('CB_R', 'cb r') or a player's name to a line-up entry."""
    phrase = phrase.strip().lower()
    if re.fullmatch(_SELF, phrase):
        return context.get("current_player")
    key = re.sub(r"[\s_-]+", "", phrase)
    for entry in context.get("lineup", []):
        if key == re.sub(r"[\s_-]+", "", entry["slot"].lower()) or phrase in entry["name"].lower():
            return entry
    return None


# words for a group of team-mates -> the slot-id prefixes they cover
_GROUPS = {"centre-back": ("CB",), "centre back": ("CB",), "center-back": ("CB",), "center back": ("CB",),
           "cbs": ("CB",), "full-back": ("LB", "RB"), "full back": ("LB", "RB"), "fullback": ("LB", "RB"),
           "defender": ("CB", "LB", "RB"), "midfielder": ("DM", "CM", "AM"), "winger": ("LW", "RW"),
           "striker": ("ST",), "forward": ("ST", "LW", "RW"), "keeper": ("GK",), "goalkeeper": ("GK",)}


def _find_people(phrase: str, context: dict) -> list[dict]:
    """One player ('him', 'CB_R', a name) or a group ('one of our centre-backs',
    'the other centre-back') -> line-up entries."""
    if who := _find_person(phrase, context):
        return [who]
    text = phrase.lower()
    for word, prefixes in _GROUPS.items():
        if word in text:
            people = [e for e in context.get("lineup", []) if e["slot"].startswith(prefixes)]
            for side, suffix in (("right", "R"), ("left", "L")):
                if re.search(rf"\b{side}\b", text):  # "our right centre-back" = CB_R
                    people = [e for e in people if e["slot"].endswith(f"_{suffix}") or e["slot"].startswith(suffix)]
            if "other" in text:
                people = [e for e in people if e["slot"] != context.get("slot")]
            return people
    return []


def _keyword_filters(brief: str, allowed: list[str], context: dict | None = None) -> list[dict]:
    """Deterministic filter extraction for the common phrasings (plus simple
    references to other players when the team context is available)."""
    text = brief.lower()
    out: list[dict] = []

    def add(field, reason, lo=None, hi=None, value=None, values=None):
        out.append({"field": field, "min": lo, "max": hi, "value": value, "values": values, "reason": f'"{reason}"'})

    if m := re.search(r"\bu[- ]?(\d{2})\b", text):
        add("age", m.group(0), hi=int(m.group(1)) - 1)
    elif m := re.search(r"(?:aged?|between)\s+(\d{2})\s*(?:-|to|and)\s*(\d{2})", text):
        add("age", m.group(0), lo=int(m.group(1)), hi=int(m.group(2)))
    else:
        if m := re.search(r"(?:under|younger than|below)\s+(\d{2})", text):
            add("age", m.group(0), hi=int(m.group(1)) - 1)
        elif m := re.search(r"(\d{2})\s+or\s+(?:younger|under)", text):
            add("age", m.group(0), hi=int(m.group(1)))
        elif context and (m := re.search(r"younger than (" + _SELF + r"|\w+)", text)):
            if who := _find_person(m.group(1), context):
                add("age", m.group(0), hi=who["age"] - 1)
        elif m := re.search(r"\b(young|youngster|prospect)\b", text):
            add("age", m.group(0), hi=23)
        if m := re.search(r"(?:over|older than|above)\s+(\d{2})", text):
            add("age", m.group(0), lo=int(m.group(1)) + 1)
        elif m := re.search(r"(\d{2})\s+or\s+older", text):
            add("age", m.group(0), lo=int(m.group(1)))
        elif context and (m := re.search(r"older than (" + _SELF + r"|\w+)", text)):
            if who := _find_person(m.group(1), context):
                add("age", m.group(0), lo=who["age"] + 1)
        elif m := re.search(r"\b(experienced|veteran)\b", text):
            add("age", m.group(0), lo=29)
    same_country = r"same (?:nationality|country) as (?:the |our |his )?(.+?)(?=$|[,.;]| and| with)"
    if context and (m := re.search(same_country, text)) and (people := _find_people(m.group(1), context)):
        nats = sorted({p["nationality"] for p in people if p.get("nationality")})
        if len(nats) == 1:
            add("nationality", m.group(0), value=nats[0])
        elif nats:
            add("nationality", m.group(0), values=nats)  # any of them
    if m := re.search(r"\b(two|both)[- ]footed\b|\bgood weak foot\b|\bstrong weak foot\b", text):
        add("weak_foot_rating", m.group(0), lo=4)
    if m := re.search(r"\bnatural\s+(" + "|".join(map(re.escape, _NATURAL)) + r")", text):
        add("natural_position", m.group(0), value=_NATURAL[m.group(1)])
    for m in re.finditer(r"(?:at least|minimum|min\.?)\s+(\d{2})\s+([a-z ]+?)"
                         r"(?=$|[,.;]| and| with| who| that| but| to| for)", text):
        attrs = _keyword_adjustments(m.group(2), allowed) or {
            a: "" for a in allowed if a.replace("_", " ") == m.group(2).strip()}
        for attr in attrs:
            add(attr, m.group(0), lo=int(m.group(1)))
    return out


def _grounded(reason: str, brief: str) -> bool:
    """Does the reason quote (or closely paraphrase) words that are in the request?"""
    quote = re.sub(r"[\"'“”‘’]", "", reason or "").lower().strip()
    request = brief.lower()
    if not quote:
        return False
    if quote in request:
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", quote) if len(w) > 2]
    return bool(words) and sum(w in request for w in words) / len(words) >= 0.6


def _validate_filters(items, allowed: list[str], ignored: list[str], brief: str,
                      nationalities: list[str] | None) -> list[dict]:
    out = []
    for f in items or []:
        if not isinstance(f, dict):
            continue
        field = f.get("field")
        if field not in FILTER_FIELDS or (field in GLOSSARY and field not in allowed):
            ignored.append(f"filter {field!r}: not a supported field for this position")
            continue
        reason = str(f.get("reason") or "")
        if not _grounded(reason, brief):
            ignored.append(f"filter {field}: its reason “{reason}” isn't in your request")
            continue
        # requirements must be asked for, not inferred from a description
        if field == "natural_position" and not re.search(r"\bnatural\b", brief, re.IGNORECASE):
            ignored.append(f"filter natural_position: you didn't ask for a natural {f.get('value') or 'player'}")
            continue
        if field in GLOSSARY and not re.search(r"\d|\bthan\b", reason, re.IGNORECASE):
            ignored.append(f"filter {field}: no number or comparison in “{reason}” — kept as a weight only")
            continue
        clean = {"field": field, "min": None, "max": None, "value": None, "values": None, "reason": reason}
        bounds = FILTER_FIELDS[field]
        if field == "nationality" and isinstance(f.get("value") or f.get("values"), list):
            wanted = [str(v).strip() for v in (f.get("values") or f.get("value")) if str(v).strip()]
            known = {n.lower(): n for n in (nationalities or [])}
            matched = [known.get(v.lower(), v) for v in wanted if not nationalities or v.lower() in known]
            for v in wanted:
                if nationalities and v.lower() not in known:
                    ignored.append(f"filter nationality {v!r}: no player in the pool has it")
            if not matched:
                continue
            clean["values"] = sorted(set(matched))
            out.append(clean)
            continue
        if bounds is None:
            value = str(f.get("value") or "").strip()
            if field == "natural_position":
                value = value.upper()
                if value not in POSITIONS:
                    ignored.append(f"filter natural_position {value!r}: not one of {' '.join(POSITIONS)}")
                    continue
            if field == "nationality" and nationalities:
                match = next((n for n in nationalities if n.lower() == value.lower()), None)
                if not match:
                    ignored.append(f"filter nationality {value!r}: no player in the pool has it")
                    continue
                value = match
            if not value:
                ignored.append(f"filter {field}: no value given")
                continue
            clean["value"] = value
        else:
            for key in ("min", "max"):
                v = f.get(key)
                if v is None:
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    ignored.append(f"filter {field} {key}={v!r} is not a number")
                    continue
                if not bounds[0] <= v <= bounds[1]:
                    ignored.append(f"filter {field} {key}={v:g} outside {bounds[0]:g}-{bounds[1]:g}")
                    continue
                clean[key] = v
            if clean["min"] is None and clean["max"] is None:
                continue
            if not _limits_ok(clean, reason, ignored):
                continue
        out.append(clean)
    return out


def strict_limit(text: str) -> tuple[str, int] | None:
    """'under 30' -> ("max", 29); 'over 28' / 'older than 28' -> ("min", 29); else None."""
    m = re.search(r"\b(under|below|younger than|less than|over|above|older than|more than)\s+(\d+)", text.lower())
    if not m:
        return None
    n = int(m.group(2))
    return ("max", n - 1) if m.group(1) in ("under", "below", "younger than", "less than") else ("min", n + 1)


def _limits_ok(clean: dict, reason: str, ignored: list[str]) -> bool:
    """Numeric limits must come from the words: 'under 30' is max 29 (corrected, not rejected),
    and without a comparison ('than him') the limit must be a number the request actually gives."""
    field = clean["field"]
    if limit := strict_limit(reason):
        key, n = limit
        if clean[key] is not None and clean[key] != n:
            ignored.append(f"filter {field}: “{reason}” means {key} {n}, not {clean[key]:g} — corrected")
            clean[key] = float(n)
        return True
    if re.search(r"\bthan\b", reason, re.IGNORECASE):
        return True  # compared with a player: the limit comes from his value
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", reason)]
    if numbers and any(v is not None and not any(abs(v - x) <= 1 for x in numbers)
                       for v in (clean["min"], clean["max"])):
        ignored.append(f"filter {field}: the limit isn't a number in “{reason}” — kept as a weight only")
        return False
    return True


def parse_brief(brief_text: str, model: str = DEFAULT_MODEL, tracker: CostTracker | None = None,
                position: str | None = None, current_weights: dict[str, float] | None = None,
                context: dict | None = None) -> dict:
    """Read a request for one slot. `context` (see pipeline.request_context)
    gives the model the line-up, the current player and the pool's valid values.
    Returns {"reweight", "reasons", "filters", "hard_foot", "source", "ignored",
    "keyword_check", "keyword_filters", "sent", "raw"}."""
    allowed = GK_ATTRS if position == "GK" else OUTFIELD_ATTRS
    context = context or {}
    keyword = _keyword_adjustments(brief_text, allowed)
    offline = {
        "reweight": {a: KEYWORD_WEIGHT for a in keyword}, "reasons": {a: f"keyword {p}" for a, p in keyword.items()},
        "filters": _keyword_filters(brief_text, allowed, context), "hard_foot": _keyword_foot(brief_text),
        "source": "keywords", "ignored": [], "keyword_check": keyword, "sent": None, "raw": None,
    }
    offline["keyword_filters"] = offline["filters"]
    if not _api_key() or not brief_text.strip():
        return offline

    pool = context.get("pool", {})
    sent = {
        "request": brief_text, **{k: context[k] for k in ("slot", "role", "formation", "current_player",
                                                          "lineup", "neighbours") if k in context},
        "position": position, "current_weights": current_weights or {},
        "allowed_attributes": {a: GLOSSARY[a] for a in allowed},
        "allowed_filters": {
            "age": "number, " + "-".join(map(str, pool.get("age_range", [15, 45]))),
            "weak_foot_rating": "number, 1-5", "nationality": "one of pool.nationalities",
            "natural_position": " | ".join(POSITIONS), "<any allowed attribute>": "number, 0-100",
        },
        "pool": pool,
    }
    raw, model, raw_text = _call_json("parse", model, PARSE_SYSTEM_PROMPT, json.dumps(sent), 700, tracker)
    if raw is None:
        return {**offline, "sent": sent, "raw": raw_text, "error": last_error}

    reweight, reasons, ignored = {}, {}, []
    items = raw.get("adjustments")
    if items is None and isinstance(raw.get("reweight"), dict):  # tolerate the simpler shape
        items = [{"attribute": k, "weight": v, "reason": ""} for k, v in raw["reweight"].items()]
    for item in items or []:
        if not isinstance(item, dict):
            continue
        attr, weight, reason = item.get("attribute"), item.get("weight"), str(item.get("reason") or "")
        if attr not in allowed:
            ignored.append(f"{attr!r}: not a valid attribute for this position")
            continue
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            ignored.append(f"{attr}: weight {weight!r} is not a number")
            continue
        if current_weights is not None and abs(weight - current_weights.get(attr, 0.0)) < 0.01:
            continue  # restated an unchanged weight -- not a change
        if not _grounded(reason, brief_text):
            ignored.append(f"{attr}: its reason “{reason}” isn't in your request")
            continue
        clamped = min(WEIGHT_MAX, max(WEIGHT_MIN, weight))
        if clamped != weight:
            ignored.append(f"{attr}: weight {weight} clamped to {clamped}")
        reweight[attr] = clamped
        reasons[attr] = reason
    raw_filters = [f for f in raw.get("filters") or [] if isinstance(f, dict)]
    # a foot sent as a filter is the foot requirement, not an unknown field
    foot_filters = [f for f in raw_filters if f.get("field") in ("foot", "preferred_foot")]
    filters = _validate_filters([f for f in raw_filters if f not in foot_filters], allowed, ignored, brief_text,
                                pool.get("nationalities"))
    foot = raw.get("hard_foot") or next((str(f.get("value")).lower() for f in foot_filters), None)
    if foot not in ("left", "right", None):
        ignored.append(f"hard_foot {foot!r} is not left/right/null")
        foot = None
    if offline["hard_foot"] and foot != offline["hard_foot"]:
        # "left-footed" in the request is unambiguous: the words win over a missed or contrary reply
        foot = offline["hard_foot"]
        reasons["filter:preferred_foot"] = f'"{offline["hard_foot"]}-footed" (from your words)'
    return {"reweight": reweight, "reasons": reasons, "filters": filters, "hard_foot": foot,
            "source": f"llm:{model}", "ignored": ignored, "keyword_check": keyword,
            "keyword_filters": offline["filters"], "sent": sent, "raw": raw_text}
