"""Measure how well change requests are read into weights and requirements.

Each case lists the attributes a request MUST change, ones that are also
acceptable, the expected preferred foot and (for requests that refer to other
players) the expected requirements. Recall = share of required attributes
found; precision = share of changes that were required or acceptable;
requirements must include every expected one and nothing unexpected.
Runs the AI when OPENROUTER_API_KEY is set
(about $0.004 per run with DeepSeek V3), else the offline keyword parser.
Pick the model with MODEL in .env.

Usage: python scripts/eval_brief_parser.py [--offline]
"""
from __future__ import annotations

import argparse
import os

from fta.config import llm_model, load_dotenv, make_tracker
from fta.llm_client import parse_brief

# a realistic team context for requests that refer to other players
TEAM = {
    "slot": "CB_L", "role": "Centre-back", "formation": "4-3-3",
    "current_player": {"slot": "CB_L", "name": "Diego Petrov", "age": 27, "foot": "right", "nationality": "IT",
                       "weak_foot": 3, "natural_positions": ["CB"],
                       "attributes": {"pace": 58, "acceleration": 55, "heading_accuracy": 74, "heading_power": 78,
                                      "tackling_standing": 75, "marking": 76, "positioning": 72,
                                      "passing_short": 58}},
    "lineup": [{"slot": "GK", "name": "Marco Andersen", "age": 37, "foot": "right", "nationality": "NO"},
               {"slot": "LB", "name": "Kenji Costa", "age": 23, "foot": "left", "nationality": "JP"},
               {"slot": "CB_L", "name": "Diego Petrov", "age": 27, "foot": "right", "nationality": "IT"},
               {"slot": "CB_R", "name": "Ishaan Traore", "age": 26, "foot": "left", "nationality": "BR"},
               {"slot": "RB", "name": "Sven Novak", "age": 25, "foot": "right", "nationality": "DE"}],
    "neighbours": ["CB_R", "DM", "GK", "LB"],
    "pool": {"nationalities": ["AR", "BR", "DE", "FR", "GH", "HR", "IN", "IT", "JP", "MA", "NO", "SN"],
             "age_range": [18, 37]},
}

CASES = [  # position, request, required, acceptable, foot, context, expected filters
    ("CB", "more aerial ability, stay left-footed", {"heading_accuracy"}, {"heading_power"}, "left", None, None),
    ("CB", "someone who wins the ball back and reads danger early", {"tackling_standing", "positioning"},
     {"marking", "tackling_sliding", "aggression"}, None, None, None),
    ("CB", "comfortable on the ball, can play long diagonals", {"passing_long"},
     {"passing_short", "ball_control", "composure", "vision"}, None, None, None),
    ("FB", "a right-footed full-back who bombs forward and delivers crosses", {"crossing"},
     {"stamina", "pace", "acceleration", "dribbling"}, "right", None, None),
    ("DM", "a tireless ball-winner in front of the defence", {"tackling_standing", "stamina"},
     {"positioning", "marking", "aggression"}, None, None, None),
    ("CM", "a creative playmaker who unlocks defences", {"vision", "through_ball"},
     {"passing_short", "passing_long", "composure", "ball_control"}, None, None, None),
    ("AM", "needs to score more and stay calm in front of goal", {"finishing", "composure"}, set(), None, None, None),
    ("WING", "explosive winger who beats his man", {"dribbling"}, {"pace", "acceleration", "ball_control"},
     None, None, None),
    ("WING", "left-footed winger with pace to burn", {"pace"}, {"acceleration"}, "left", None, None),
    ("ST", "a target man who is strong in the air", {"heading_accuracy"}, {"heading_power", "aggression"},
     None, None, None),
    ("ST", "clinical poacher, always in the right place", {"finishing", "positioning"}, {"composure"},
     None, None, None),
    ("GK", "a keeper with great reflexes who is good with his feet", {"gk_reflexes", "gk_kicking"},
     {"passing_short", "composure"}, None, None, None),
    # requests that refer to the team: need the context to be read correctly
    ("CB", "a better centre-back", set(), set(), None, TEAM, (set(), set())),
    ("CB", "same nationality as our right centre-back, and quicker", {"pace"}, {"acceleration"}, None, TEAM,
     ({("nationality", "BR")}, {("pace", "min", 59)})),                      # "quicker" than him is optional
    ("CB", "younger than him and better in the air than him", set(), {"heading_accuracy", "heading_power"},
     None, TEAM, ({("age", "max", 26), ("heading_accuracy", "min", 75)}, set())),
    ("GK", "better reflexes and same nationality as at least one centre back", {"gk_reflexes"}, set(), None,
     {**TEAM, "slot": "GK", "role": "Goalkeeper"}, ({("nationality", "BR"), ("nationality", "IT")}, set())),
]


def _filter_keys(filters: list[dict]) -> set[tuple]:
    out = set()
    for f in filters:
        if f["value"] is not None:
            out.add((f["field"], f["value"]))
        for v in f.get("values") or []:  # "any of" (e.g. either centre-back's nationality)
            out.add((f["field"], v))
        for bound in ("min", "max"):
            if f[bound] is not None:
                out.add((f["field"], bound, int(f[bound])))
    return out


def score(result: dict, required: set, acceptable: set) -> tuple[float, float]:
    got = set(result["reweight"])
    recall = len(got & required) / len(required) if required else 1.0
    precision = len(got & (required | acceptable)) / len(got) if got else (1.0 if not required else 0.0)
    return recall, precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="keyword parser only (no API calls)")
    args = ap.parse_args()
    load_dotenv()
    if args.offline:
        os.environ.pop("OPENROUTER_API_KEY", None)
    tracker = make_tracker()

    totals = {"recall": 0.0, "precision": 0.0, "foot": 0, "filters": 0, "filter_cases": 0}
    for position, brief, required, acceptable, foot, context, filters in CASES:
        result = parse_brief(brief, model=llm_model(), tracker=tracker, position=position, context=context,
                             current_weights={"tackling_standing": 0.2, "marking": 0.2, "heading_accuracy": 0.15,
                                              "positioning": 0.15} if position == "CB" else None)
        r, p = score(result, required, acceptable)
        f_ok = result["hard_foot"] == foot
        filt_ok = True
        if filters is None:  # no requirement asked for, apart from the foot
            filt_ok = not result["filters"]
            totals["filter_cases"] += 1
            totals["filters"] += filt_ok
        else:
            totals["filter_cases"] += 1
            required_f, optional_f = filters
            got_f = _filter_keys(result["filters"])
            filt_ok = required_f <= got_f <= required_f | optional_f
            totals["filters"] += filt_ok
        totals["recall"] += r
        totals["precision"] += p
        totals["foot"] += f_ok
        flag = "OK " if r == 1 and p == 1 and f_ok and filt_ok else "!! "
        print(f"{flag}[{result['source']}] {position:4s} {brief!r}\n"
              f"      -> {sorted(result['reweight'])} foot={result['hard_foot']} "
              f"filters={sorted(_filter_keys(result['filters']), key=str)}  recall {r:.0%} precision {p:.0%}")
        for item in result.get("ignored", []):
            print(f"      rejected: {item}")
    n = len(CASES)
    print(f"\nRecall {totals['recall'] / n:.0%} · precision {totals['precision'] / n:.0%} · "
          f"foot correct {totals['foot']}/{n} · requirements correct "
          f"{totals['filters']}/{totals['filter_cases']}")
    print("(requirements: every expected one present, and none that weren't asked for)")
    print(tracker.summary())


if __name__ == "__main__":
    main()
