"""Generate the synthetic demo pool: ~100 players with position-appropriate
attributes (archetype mean + noise), plus hand-designed "demo case" players
that make every feature visible -- foot strictness, age filters, chemistry
rules, specialists vs all-rounders, versatile players. Deterministic given a seed.

Demo-case players carry a tag starting with "demo:" that says what they show.

Usage: python scripts/generate_synthetic_players.py [--seed 42] [--count 100]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "processed" / "players_synthetic.json"

FIRST_NAMES = [
    "Arjun", "Kabir", "Rohan", "Dev", "Aarav", "Vikram", "Ishaan", "Rahul",
    "Marco", "Lucas", "Diego", "Mateus", "Thiago", "Bruno", "Kwame", "Amir",
    "Noah", "Liam", "Ethan", "Mason", "Leo", "Hugo", "Karim", "Youssef",
    "Sven", "Erik", "Jonas", "Felix", "Tomas", "Milos", "Kenji", "Haru",
]
LAST_NAMES = [
    "Verma", "Mehta", "Sharma", "Kapoor", "Nair", "Rossi", "Silva", "Costa",
    "Muller", "Novak", "Kovačić", "Diallo", "Traore", "Nakamura",
    "Sato", "Andersen", "Berg", "Fischer", "Garcia", "Martins", "Okafor",
    "Osei", "Petrov", "Lindqvist", "Haddad", "Al-Farsi", "Torres", "Reyes",
]
NATIONALITIES = ["IN", "BR", "IT", "DE", "HR", "SN", "JP", "NO", "MA", "AR", "FR", "GH"]

# archetype: mean attribute overrides on top of a 50 baseline
ARCHETYPES = {
    "GK": {"gk_reflexes": 72, "gk_handling": 68, "gk_positioning": 70, "gk_kicking": 60, "composure": 68},
    "CB_stopper": {"tackling_standing": 75, "marking": 76, "heading_accuracy": 74, "heading_power": 78,
                   "positioning": 72, "passing_short": 58, "pace": 58, "aggression": 72},
    "CB_ballplaying": {"tackling_standing": 68, "marking": 68, "heading_accuracy": 66, "heading_power": 68,
                       "positioning": 70, "passing_short": 76, "passing_long": 72, "pace": 64, "vision": 70},
    "FB": {"pace": 74, "acceleration": 75, "stamina": 78, "tackling_standing": 66, "crossing": 68,
           "passing_short": 68, "dribbling": 65, "positioning": 62},
    "DM": {"tackling_standing": 74, "positioning": 74, "passing_short": 72, "marking": 68,
           "stamina": 76, "vision": 68, "aggression": 62},
    "CM_box_to_box": {"passing_short": 74, "vision": 68, "stamina": 80, "tackling_standing": 66,
                      "dribbling": 68, "composure": 68, "passing_long": 64},
    "CM_playmaker": {"passing_short": 80, "vision": 82, "through_ball": 76, "dribbling": 70,
                     "composure": 74, "passing_long": 72, "stamina": 66},
    "AM": {"through_ball": 78, "vision": 80, "dribbling": 76, "passing_short": 76,
           "finishing": 66, "composure": 72, "ball_control": 78},
    "WING": {"pace": 82, "acceleration": 83, "dribbling": 78, "crossing": 72,
             "finishing": 62, "ball_control": 76, "stamina": 70},
    "ST_poacher": {"finishing": 82, "heading_accuracy": 68, "positioning": 80, "composure": 76,
                   "pace": 74, "ball_control": 70},
    "ST_target": {"finishing": 72, "heading_accuracy": 80, "heading_power": 80, "positioning": 74,
                  "composure": 70, "pace": 58, "ball_control": 64},
}
ARCHETYPE_TO_NATURAL_POS = {
    "GK": ["GK"], "CB_stopper": ["CB"], "CB_ballplaying": ["CB"], "FB": ["FB", "CB"],
    "DM": ["DM", "CM"], "CM_box_to_box": ["CM", "DM"], "CM_playmaker": ["CM", "AM"],
    "AM": ["AM", "CM"], "WING": ["WING"], "ST_poacher": ["ST"], "ST_target": ["ST"],
}
# share of the pool per position, and which archetypes fill it
POSITION_QUOTA = {"GK": 0.08, "CB": 0.16, "FB": 0.14, "DM": 0.10, "CM": 0.14, "AM": 0.08, "WING": 0.16, "ST": 0.14}
POSITION_ARCHETYPES = {
    "GK": ["GK"], "CB": ["CB_stopper", "CB_ballplaying"], "FB": ["FB"], "DM": ["DM"],
    "CM": ["CM_box_to_box", "CM_playmaker"], "AM": ["AM"], "WING": ["WING"], "ST": ["ST_poacher", "ST_target"],
}

# Hand-designed demo cases: (position, archetype, what it demonstrates, fixed fields, attribute overrides)
DEMO_CASES = [
    ("CB", "CB_stopper", ("Right-footed aerial CB who out-fits the left-footed option — try 'stay left-footed' "
                          "and change the foot strictness"), {"foot": "right", "age": 27},
     {"heading_accuracy": 90, "heading_power": 88, "tackling_standing": 84, "marking": 85, "positioning": 82}),
    ("CB", "CB_stopper", "Left-footed CB, slightly weaker than the right-footed one — the compliant pick",
     {"foot": "left", "age": 26},
     {"heading_accuracy": 84, "heading_power": 82, "tackling_standing": 80, "marking": 80, "positioning": 79}),
    ("CB", "CB_stopper", "Pure stopper: dominant in the air, poor on the ball", {"foot": "right", "age": 29},
     {"heading_accuracy": 88, "heading_power": 90, "passing_short": 42, "passing_long": 38, "vision": 35}),
    ("CB", "CB_ballplaying", "Ball-playing CB: great passer, weak in the air", {"foot": "left", "age": 24},
     {"passing_short": 84, "passing_long": 82, "vision": 78, "heading_accuracy": 55, "heading_power": 52}),
    ("CB", "CB_stopper", "Centre-back pair (1/2) sharing a nationality — triggers the chemistry bonus",
     {"foot": "left", "age": 28, "nationality": "BR"}, {"tackling_standing": 82, "marking": 82}),
    ("CB", "CB_stopper", "Centre-back pair (2/2) sharing a nationality — triggers the chemistry bonus",
     {"foot": "right", "age": 30, "nationality": "BR"}, {"tackling_standing": 81, "marking": 83}),
    ("FB", "FB", "Full-back who fits on the wing too — versatility shows in 'Fit by position'",
     {"foot": "left", "age": 23}, {"pace": 90, "acceleration": 90, "dribbling": 84, "crossing": 82}),
    ("FB", "FB", "Classic left-back specialist", {"foot": "left", "age": 26},
     {"tackling_standing": 80, "crossing": 80, "stamina": 88, "positioning": 76}),
    ("DM", "DM", "Tireless ball-winner (stamina 92) — pair with high-stamina mids for the chemistry bonus",
     {"foot": "right", "age": 27, "stamina_base": 92}, {"stamina": 92, "tackling_standing": 84, "aggression": 80}),
    ("DM", "DM", "Low-stamina holding midfielder — two of these in midfield costs chemistry",
     {"foot": "right", "age": 31, "stamina_base": 55}, {"stamina": 55, "passing_short": 82, "vision": 78}),
    ("CM", "CM_playmaker", "Elite playmaker with low stamina — brilliant, but fades after the hour",
     {"foot": "left", "age": 29, "stamina_base": 56}, {"vision": 90, "through_ball": 88, "stamina": 56}),
    ("CM", "CM_box_to_box", "Balanced all-rounder — no weakness, no standout", {"foot": "right", "age": 25},
     {a: 75 for a in ("passing_short", "vision", "stamina", "tackling_standing", "dribbling", "composure")}),
    ("AM", "AM", "Goal-scoring number 10: clinical and composed", {"foot": "right", "age": 26},
     {"finishing": 86, "composure": 86, "through_ball": 80}),
    ("WING", "WING", "18-year-old prospect: electric pace, raw end product — try 'under 21'",
     {"foot": "left", "age": 18}, {"pace": 92, "acceleration": 93, "finishing": 55, "composure": 50}),
    ("WING", "WING", "Two-footed winger (weak foot 5/5) — try 'two-footed'", {"foot": "right", "age": 25,
     "weak_foot": 5}, {"dribbling": 84, "crossing": 80}),
    ("WING", "WING", "One-dimensional sprinter: fastest in the pool, little else", {"foot": "right", "age": 22},
     {"pace": 97, "acceleration": 96, "dribbling": 55, "crossing": 50, "ball_control": 55}),
    ("WING", "WING", "Right-footed player who plays on the left — an inside forward, no left-foot bonus",
     {"foot": "right", "age": 27}, {"finishing": 80, "dribbling": 82}),
    ("ST", "ST_target", "35-year-old target man: superb in the air, slow — try 'under 30' or 'aerial'",
     {"foot": "right", "age": 35}, {"heading_accuracy": 91, "heading_power": 90, "pace": 45, "acceleration": 44}),
    ("ST", "ST_poacher", "Poacher: lethal finisher, doesn't link play", {"foot": "left", "age": 28},
     {"finishing": 91, "positioning": 88, "passing_short": 48, "vision": 45}),
    ("GK", "GK", "Sweeper-keeper: best distribution, average shot-stopper", {"foot": "left", "age": 27},
     {"gk_kicking": 88, "gk_positioning": 80, "gk_reflexes": 66, "passing_short": 72}),
    ("GK", "GK", "37-year-old shot-stopper: great reflexes, poor with his feet", {"foot": "right", "age": 37},
     {"gk_reflexes": 90, "gk_handling": 84, "gk_kicking": 45}),
]

ALL_ATTRS = [
    "pace", "acceleration", "stamina", "passing_short", "passing_long", "through_ball",
    "pass_power_bullet", "crossing", "finishing", "heading_accuracy", "heading_power",
    "dribbling", "ball_control", "vision", "composure", "aggression",
    "tackling_standing", "tackling_sliding", "marking", "positioning",
    "gk_reflexes", "gk_handling", "gk_positioning", "gk_kicking",
]
GK_ONLY = {"gk_reflexes", "gk_handling", "gk_positioning", "gk_kicking"}


def clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


def gen_player(rng: random.Random, idx: int, archetype: str, foot: str | None = None) -> dict:
    is_gk = archetype == "GK"
    means = ARCHETYPES[archetype]
    attrs = {}
    for name in ALL_ATTRS:
        if name in GK_ONLY and not is_gk:
            attrs[name] = None
            continue
        if name not in GK_ONLY and is_gk and name not in means:
            attrs[name] = round(clamp(rng.gauss(35, 6)), 1)  # GKs are weak outfield-wise, not null
            continue
        attrs[name] = round(clamp(rng.gauss(means.get(name, 50), 6)), 1)

    foot = foot or rng.choices(["right", "left"], weights=[68, 32])[0]
    return {
        "player_id": f"p_{idx:04d}",
        "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
        "preferred_foot": foot,
        "weak_foot_rating": rng.randint(2, 4),
        "age": int(max(17, min(36, round(rng.gauss(26, 4))))),
        "nationality": rng.choice(NATIONALITIES),
        "stamina_base": round(clamp(rng.gauss(72, 8))),
        "attributes": attrs,
        "natural_positions": list(ARCHETYPE_TO_NATURAL_POS[archetype]),
        "tags": [archetype, f"{foot}_footed"],
        "source": "synthetic",
    }


def demo_player(rng: random.Random, idx: int, case: tuple) -> dict:
    position, archetype, note, fixed, overrides = case
    p = gen_player(rng, idx, archetype, foot=fixed.get("foot"))
    p["attributes"].update({k: float(v) for k, v in overrides.items()})
    p["age"] = fixed.get("age", p["age"])
    p["nationality"] = fixed.get("nationality", p["nationality"])
    p["weak_foot_rating"] = fixed.get("weak_foot", p["weak_foot_rating"])
    p["stamina_base"] = fixed.get("stamina_base", round(p["attributes"].get("stamina") or p["stamina_base"]))
    if position == "FB" and "WING" not in p["natural_positions"] and overrides.get("pace", 0) >= 85:
        p["natural_positions"].append("WING")
    p["tags"].append(f"demo: {note}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--count", type=int, default=100)
    args = ap.parse_args()
    if args.count < 60:
        raise SystemExit("--count must be at least 60 so every position has depth for two teams")

    rng = random.Random(args.seed)
    pool = [demo_player(rng, i + 1, case) for i, case in enumerate(DEMO_CASES)]
    have = {pos: sum(c[0] == pos for c in DEMO_CASES) for pos in POSITION_QUOTA}
    for position, share in POSITION_QUOTA.items():
        target = max(have[position], round(share * args.count))
        for n in range(target - have[position]):
            archetype = rng.choice(POSITION_ARCHETYPES[position])
            # wide positions alternate feet so both flanks have natural options
            foot = ("left" if n % 2 == 0 else "right") if position in ("FB", "WING") else None
            pool.append(gen_player(rng, len(pool) + 1, archetype, foot=foot))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(pool)} synthetic players ({len(DEMO_CASES)} demo cases) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
