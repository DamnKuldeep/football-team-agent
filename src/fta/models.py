"""Core data models. Every other module speaks these schemas and nothing else,
so the data source (synthetic vs Kaggle) and the LLM provider are both fully
swappable without touching scoring/simulation/storage logic.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Foot = Literal["left", "right"]

# Every outfield/GK attribute we track. GK-only attributes are null for
# outfield players and vice versa -- enforced in Player.check_gk_fields.
ATTRIBUTE_NAMES = [
    "pace", "acceleration", "stamina",
    "passing_short", "passing_long", "through_ball", "pass_power_bullet", "crossing",
    "finishing", "heading_accuracy", "heading_power",
    "dribbling", "ball_control", "vision", "composure", "aggression",
    "tackling_standing", "tackling_sliding", "marking", "positioning",
    "gk_reflexes", "gk_handling", "gk_positioning", "gk_kicking",
]

GK_ONLY = {"gk_reflexes", "gk_handling", "gk_positioning", "gk_kicking"}
OUTFIELD_ONLY = set(ATTRIBUTE_NAMES) - GK_ONLY - {"stamina"}


class Player(BaseModel):
    player_id: str
    name: str
    preferred_foot: Foot
    weak_foot_rating: int = Field(ge=1, le=5)
    age: int = Field(ge=15, le=45)
    nationality: str = "XX"
    stamina_base: int = Field(ge=0, le=100)
    attributes: dict[str, float | None]
    natural_positions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source: Literal["synthetic", "kaggle"] = "synthetic"

    @field_validator("attributes")
    @classmethod
    def check_attribute_ranges(cls, v: dict[str, float | None]) -> dict[str, float | None]:
        for name, val in v.items():
            if val is None:
                continue
            if not (0 <= val <= 100):
                raise ValueError(f"attribute {name}={val} out of range [0,100]")
        return v

    def is_gk(self) -> bool:
        return "GK" in self.natural_positions

    def attr(self, name: str, default: float = 50.0) -> float:
        """Safe attribute getter -- returns `default` for missing/null values
        instead of raising, since a swap-engine reweight may reference an
        attribute a given player never had scored (e.g. gk_reflexes on an
        outfield player)."""
        val = self.attributes.get(name)
        return default if val is None else float(val)


class SoftBonus(BaseModel):
    attribute: str  # "preferred_foot" or any ATTRIBUTE_NAMES entry
    value: str | None = None  # for categorical attrs like preferred_foot
    bonus_pct: float


class HardFilter(BaseModel):
    attribute: str
    value: str | None = None
    values: list[str] | None = None   # "any of" (e.g. nationality of either centre-back)
    min: float | None = None
    max: float | None = None


class PositionTemplate(BaseModel):
    position: str
    formation_slot: str
    attribute_weights: dict[str, float]
    hard_filters: list[HardFilter] = Field(default_factory=list)
    soft_bonuses: list[SoftBonus] = Field(default_factory=list)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> PositionTemplate:
        total = sum(self.attribute_weights.values())
        if not (0.98 <= total <= 1.02):
            raise ValueError(
                f"attribute_weights for {self.position} sum to {total:.3f}, must be ~1.0"
            )
        for name in self.attribute_weights:
            if name not in ATTRIBUTE_NAMES:
                raise ValueError(f"unknown attribute '{name}' in template {self.position}")
        return self


class FormationSlot(BaseModel):
    slot_id: str
    position: str
    player_id: str | None = None


class Team(BaseModel):
    team_id: str
    name: str
    formation: str
    slots: list[FormationSlot]
    avg_fit_score: float = 0.0
    chemistry_score: float = 100.0
    version: int = 1
    parent_version: int | None = None
    diff: dict | None = None  # what changed vs parent_version, and why
    # position -> attribute weights this team is scored with (tuned at build
    # time or later). None = the formation's defaults (formation_weights.json).
    weights: dict[str, dict[str, float]] | None = None

    def player_ids(self) -> list[str]:
        return [s.player_id for s in self.slots if s.player_id]


class ChemistryRule(BaseModel):
    rule_id: str
    label: str
    description: str
    min_points: float = 0.0
    max_points: float


class SwapRequest(BaseModel):
    target_slot: str
    reweight: dict[str, float] = Field(default_factory=dict)
    hard_filters: list[HardFilter] = Field(default_factory=list)
    exclude_player_ids: list[str] = Field(default_factory=list)
    requested_by: Literal["human", "agent"] = "human"
    source: Literal["manual", "performance_derived"] = "manual"
    rationale: str = ""
    # why each reweight entry was made (attribute -> phrase/evidence), and who
    # interpreted the brief ("llm:<model>", "keywords", "performance", "human-edited")
    adjustment_reasons: dict[str, str] = Field(default_factory=dict)
    interpreted_by: str = ""
    # ranking policy: points subtracted per hard-filter violation ("prefer"),
    # and how much the chemistry change counts toward the rank score
    violation_penalty: float = 5.0
    chemistry_weight: float = 0.5


class ShortlistCandidate(BaseModel):
    player_id: str
    fit_score: float
    delta_fit: float
    chemistry_delta: float
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""
    # rank_score = fit_score - penalty + chemistry_bonus  (what the list is sorted by)
    penalty: float = 0.0
    chemistry_bonus: float = 0.0
    rank_score: float = 0.0


class Shortlist(BaseModel):
    target_slot: str
    current_player_id: str | None
    current_fit_score: float | None
    candidates: list[ShortlistCandidate]


class MatchEvent(BaseModel):
    event: str
    player_id: str
    team_id: str | None = None  # the side the player was on -- the same player can appear for both
    success: bool | None = None
    outcome: str | None = None
    xg: float | None = None


class Possession(BaseModel):
    possession_id: int
    attacking_team: str
    chain: list[MatchEvent]
    minute: int | None = None  # match clock, 1-90 (None in logs saved before it existed)


class EventLog(BaseModel):
    match_id: str
    seed: int
    team_a: str
    team_b: str
    possessions: list[Possession]
    final_score: dict[str, int]
    # snapshot of who played, so a match stays readable after teams change or are deleted
    played_at: str = ""                                            # ISO UTC timestamp
    team_names: dict[str, str] = Field(default_factory=dict)       # team_id -> name
    team_versions: dict[str, int] = Field(default_factory=dict)    # team_id -> version played
    lineups: dict[str, dict[str, str]] = Field(default_factory=dict)  # team_id -> slot_id -> player_id
    slot_positions: dict[str, dict[str, str]] = Field(default_factory=dict)  # team_id -> slot_id -> position
    formations: dict[str, str] = Field(default_factory=dict)       # team_id -> formation

    def name(self, team_id: str) -> str:
        return self.team_names.get(team_id, team_id)

    def name_v(self, team_id: str) -> str:
        """Name with the version that played, e.g. 'Alpha FC v2'."""
        name, v = self.name(team_id), self.team_versions.get(team_id)
        return name if v is None or name.endswith(f" v{v}") else f"{name} v{v}"

    @property
    def label(self) -> str:
        """e.g. 'M0003 · Alpha FC 2–1 Beta FC · 23 Sep 14:05'"""
        when = ""
        if self.played_at:
            from datetime import datetime
            when = " · " + datetime.fromisoformat(self.played_at).astimezone().strftime("%d %b %H:%M")
        return (f"{self.match_id} · {self.name_v(self.team_a)} {self.final_score.get(self.team_a, 0)}–"
                f"{self.final_score.get(self.team_b, 0)} {self.name_v(self.team_b)}{when}")

    def slot_of(self, player_id: str, team_id: str | None = None) -> tuple[str, str] | None:
        """(team_id, slot_id) the player started in (for `team_id` if given)."""
        for tid, slots in self.lineups.items():
            if team_id and tid != team_id:
                continue
            for slot_id, pid in slots.items():
                if pid == player_id:
                    return tid, slot_id
        return None

    def sides_of(self, player_id: str) -> list[str]:
        """Every team the player started for (two in a team-vs-own-version match)."""
        return [tid for tid, slots in self.lineups.items() if player_id in slots.values()]


class PlayerStats(BaseModel):
    player_id: str
    passes_attempted: int = 0
    passes_completed: int = 0
    tackles_won: int = 0
    tackles_lost: int = 0
    aerial_duels_won: int = 0
    aerial_duels_lost: int = 0
    key_passes: int = 0
    shots: int = 0
    goals: int = 0
    xg: float = 0.0                 # summed expected goals of this player's shots
    dribbles_attempted: int = 0
    dribbles_completed: int = 0
    saves: int = 0                  # goalkeepers
    goals_conceded: int = 0         # goalkeepers

    @property
    def pass_accuracy_pct(self) -> float:
        if self.passes_attempted == 0:
            return 0.0
        return round(100 * self.passes_completed / self.passes_attempted, 1)


class PlayerReport(BaseModel):
    player_id: str
    match_id: str
    stats: PlayerStats
    rating: float
    team_id: str | None = None
    slot_id: str | None = None
    position: str | None = None
    strengths: list[str] = Field(default_factory=list)   # evidence-backed, from the stats
    weaknesses: list[str] = Field(default_factory=list)
