"""Streamlit front end. Launch with `fta ui` or `streamlit run streamlit_app.py`.

Three areas: Squad (build, change and tune a team, with the assistant),
Matches (play any two team versions; debrief, ratings, the assistant's
analysis) and Players (browse the pool, find the best fit for your weights).
Every action calls the same pipeline.py / agent.py functions as the CLI; the UI
adds visibility and approvals, not logic.
"""
from __future__ import annotations

import html
import json
import os
import subprocess
import sys

import altair as alt
import pandas as pd
import streamlit as st

from fta import agent, data_loader, llm_catalog, pipeline, storage
from fta.config import SOURCES, llm_enabled, llm_model, load_dotenv, make_tracker, player_source
from fta.llm_client import GK_ATTRS, OUTFIELD_ATTRS
from fta.models import EventLog, FormationSlot, HardFilter, Shortlist, SwapRequest, Team
from fta.report_generator import (
    RATING_BASE,
    RATING_POINTS,
    ROLE_LABEL,
    rating_breakdown,
    role_group,
    team_stats,
)
from fta.scoring import describe_filter, fit_breakdown, fit_score
from fta.simulator import CHEMISTRY_EFFECT


def secrets_to_env() -> None:
    """Streamlit Community Cloud keeps settings in the app's Secrets; the code reads the environment."""
    try:
        secrets = {k: v for k, v in st.secrets.items() if isinstance(v, (str, int, float))}
    except FileNotFoundError:  # no secrets.toml: running locally with .env
        return
    for key, value in secrets.items():
        os.environ.setdefault(key, str(value))


load_dotenv()
secrets_to_env()
st.set_page_config(page_title="Football Team Agent", page_icon="⚽", layout="wide")

SLOT_XY = {  # (x%, y% from top) per formation slot; the team attacks upward
    "GK": (50, 91), "LB": (14, 72), "CB_L": (37, 77), "CB_R": (63, 77), "RB": (86, 72),
    "DM": (50, 60), "DM_L": (36, 60), "DM_R": (64, 60), "CM_L": (30, 45), "CM_R": (70, 45),
    "AM": (50, 38), "LW": (16, 21), "RW": (84, 21), "ST": (50, 11),
}
ATTRIBUTE_GROUPS = {
    "Pace": ["pace", "acceleration"],
    "Passing": ["passing_short", "passing_long", "through_ball", "pass_power_bullet", "crossing", "vision"],
    "Shooting": ["finishing", "composure"],
    "Aerial": ["heading_accuracy", "heading_power"],
    "On the ball": ["dribbling", "ball_control"],
    "Defending": ["tackling_standing", "tackling_sliding", "marking", "positioning", "aggression"],
    "Physical": ["stamina"],
    "Goalkeeping": ["gk_reflexes", "gk_handling", "gk_positioning", "gk_kicking"],
}
STRICTNESS = {"Ignore": 0.0, "Prefer": 5.0, "Strongly prefer": 15.0, "Require": 1000.0}
EXAMPLE_REQUESTS = {  # one-click examples in "Change players", by position: skills, limits, references
    "GK": ["a commanding shot-stopper who is good with his feet, under 30",
           "better reflexes and same nationality as at least one centre-back",
           "an experienced keeper with at least 75 handling who stays calm under pressure"],
    "CB": ["a left-footed ball-playing centre-back who wins headers, under 27",
           "quicker than him and same nationality as the other centre-back",
           "a physical stopper with at least 80 marking, older than 28"],
    "FB": ["a natural full back who bombs forward and crosses, two-footed",
           "a quicker, younger full back who still defends, under 24",
           "better at crossing than him and a tireless runner up and down the flank"],
    "DM": ["a tireless ball-winner who can also switch play, under 26",
           "a holding midfielder who reads the game, at least 78 positioning",
           "younger than him and from the same country as our keeper"],
    "CM": ["a box-to-box midfielder with the engine to press, under 25",
           "calm on the ball and left-footed, a natural central midfielder",
           "a better passer than him who can also win the ball back"],
    "AM": ["a creative playmaker who unlocks defences, with at least 75 vision",
           "a number 10 who scores as well as creates, under 26",
           "quicker than him with better close control"],
    "WING": ["an explosive left-footed winger who beats his man, at least 80 pace",
             "a right-footer who can cross and track back, under 27",
             "quicker than him and a better finisher, two-footed"],
    "ST": ["a clinical finisher who is also strong in the air, under 25",
           "a target man older than 28 with composure in front of goal",
           "a pacy striker who runs in behind, same nationality as one of our wingers"],
}
INFO_HELP = "Player profile: attributes, scouting report, match history, compare"
VERDICT_STYLE = {"Sign": ("#1baf7a", "#062b1d"), "Short-term signing": ("#9bd3b9", "#062b1d"),
                 "Monitor": ("#eda100", "#2b1d00"), "Pass": ("#e34948", "#ffffff")}


def pretty(attr: str) -> str:
    text = attr.replace("gk_", "GK ").replace("_", " ")
    return text[0].upper() + text[1:]


def colors() -> dict[str, str]:
    """Theme-aware colours: home/away = categorical slots 1-2 of the validated palette."""
    if st.context.theme.type == "dark":
        return {"home": "#3987e5", "away": "#d95926", "muted": "#c3c2b7", "track": "rgba(255,255,255,.12)",
                "surface": "#0e1117", "bar": "#3987e5"}
    return {"home": "#2a78d6", "away": "#eb6834", "muted": "#52514e", "track": "rgba(0,0,0,.08)",
            "surface": "#ffffff", "bar": "#2a78d6"}


st.html("""<style>
.fta-pitch{position:relative;width:100%;max-width:420px;aspect-ratio:68/90;margin:0 auto;
  border-radius:12px;overflow:hidden;background:repeating-linear-gradient(180deg,#2e7a45 0 10%,#2a7040 10% 20%)}
.fta-line{position:absolute;border:2px solid rgba(255,255,255,.45)}
.fta-chip{position:absolute;transform:translate(-50%,-50%);min-width:82px;max-width:31%;padding:4px 8px;
  border-radius:8px;background:rgba(8,18,12,.84);color:#fff;font-size:12px;line-height:1.25;text-align:center}
.fta-chip small{display:block;font-size:10px;opacity:.72;letter-spacing:.03em}
.fta-chip b{display:block;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fta-chip.hl{outline:2px solid #f5c542;box-shadow:0 0 0 5px rgba(245,197,66,.3)}
.fta-chip .rt{display:inline-block;margin-top:2px;padding:0 6px;border-radius:6px;font-weight:700;font-size:11px}
.fta-score{display:grid;grid-template-columns:1fr auto 1fr;align-items:start;gap:12px;text-align:center;padding:4px 0}
.fta-score .team{font-size:20px;font-weight:700}
.fta-score .num{font-size:44px;font-weight:800;line-height:1;letter-spacing:2px}
.fta-score .scorers{font-size:13px;opacity:.8;margin-top:6px;line-height:1.6}
.fta-stat{display:grid;grid-template-columns:56px 1fr 56px;gap:8px;align-items:center;margin:6px 0;font-size:14px}
.fta-stat .v{font-weight:700;font-variant-numeric:tabular-nums}
.fta-stat .lbl{text-align:center;font-size:12px;opacity:.75;margin-bottom:3px}
.fta-bar{display:flex;gap:2px;height:7px}
.fta-bar span{display:block;height:100%;border-radius:4px}
.fta-attr{display:grid;grid-template-columns:130px 1fr 34px 76px;gap:8px;align-items:center;font-size:13px;margin:3px 0}
.fta-attr .track,.fta-crit .track{height:8px;border-radius:4px;overflow:hidden}
.fta-attr .fill,.fta-crit .fill{height:100%;border-radius:4px}
.fta-attr .pct{font-size:11px;opacity:.7}
.fta-crit{display:grid;grid-template-columns:150px 64px 1fr 40px;gap:10px;align-items:center;font-size:14px;
  padding:8px 0 2px;border-top:1px solid rgba(128,128,128,.18)}
.fta-note{font-size:13px;opacity:.85;margin:2px 0 6px 0}
.fta-pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;border:1px solid rgba(128,128,128,.4)}
.fta-verdict{display:inline-block;padding:4px 14px;border-radius:10px;font-weight:800;font-size:16px}
[class*="st-key-ptable_"] [data-testid="stHorizontalBlock"]{border-bottom:1px solid rgba(128,128,128,.16);
  padding:1px 0;gap:.6rem}
[class*="st-key-ptable_"] p{margin:0;font-size:14px}
[class*="st-key-ptable_"] [data-testid="stVerticalBlock"]{gap:0}
[class*="st-key-ptable_"] button[kind="tertiary"]{color:#2a78d6}
</style>""")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _pool_stamp() -> tuple:
    return tuple(sorted((p.name, p.stat().st_mtime) for p in data_loader.PROCESSED.glob("*.json")))


@st.cache_resource(show_spinner="Loading player pool...")
def get_context(source: str, _stamp: tuple) -> pipeline.Context:
    return pipeline.load_context(source)


def generate_pool(seed: int = 42, count: int = 100) -> None:
    script = pipeline.ROOT / "scripts" / "generate_synthetic_players.py"
    subprocess.run([sys.executable, str(script), "--seed", str(seed), "--count", str(count)], check=True)
    get_context.clear()


def team_label(team_id: str) -> str:
    entry = storage.list_teams().get(team_id, {})
    return f"{entry.get('name', team_id)} ({team_id})"


def team_name(team_id: str) -> str:
    return storage.list_teams().get(team_id, {}).get("name", team_id)


def match_label(match_id: str) -> str:
    return pipeline.load_event_log(match_id).label


def player_name(ctx: pipeline.Context, player_id: str | None) -> str:
    p = ctx.lookup.get(player_id) if player_id else None
    return p.name if p else (player_id or "<empty>")


def ensure_choice(key: str, options: list, preferred=None) -> None:
    """Keep a keyed widget's stored value valid, optionally steering it."""
    if preferred in options:
        st.session_state[key] = preferred
    elif st.session_state.get(key) not in options and options:
        st.session_state[key] = options[0]


def full_height(n_rows: int) -> int:
    return 35 * (n_rows + 1) + 3


def minute_of(p) -> int:
    return p.minute or p.possession_id


def rating_style(r: float) -> tuple[str, str]:
    """(background, text) for a rating badge; the number is always shown too."""
    if r >= 7.5:
        return "#1baf7a", "#062b1d"
    if r >= 6.5:
        return "#dfe7df", "#1b2a1b"
    if r >= 5.5:
        return "#eda100", "#2b1d00"
    return "#e34948", "#fff"


def model_id() -> str:
    return st.session_state.get("model") or llm_model()


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def player_rows(ctx: pipeline.Context, pids: list[str], columns: list[tuple[str, float]], cells, key: str,
                info_kwargs: dict | None = None, action: tuple[str, callable] | None = None) -> None:
    """A compact table: ⓘ button, then `cells(pid, i)` values under `columns`
    [(header, width)], plus an optional (label, callback(pid)) action button."""
    widths = [0.3] + [w for _, w in columns] + ([0.9] if action else [])
    with st.container(key=f"ptable_{key}", gap=None):
        head = st.columns(widths, vertical_alignment="bottom")
        for col, (name, _) in zip(head[1:], columns):
            col.caption(name)
        for i, pid in enumerate(pids):
            row = st.columns(widths, vertical_alignment="center")
            if info_kwargs is not None and row[0].button(":material/info:", key=f"{key}_i{i}", type="tertiary",
                                                         help=INFO_HELP):
                player_dialog(pid, **info_kwargs)
            for col, value in zip(row[1:], cells(pid, i)):
                col.markdown(value)
            if action:
                row[-1].button(action[0], key=f"{key}_a{i}", on_click=action[1], args=(pid,), type="secondary")


def weight_inputs(key: str, weights: dict[str, float], allowed: list[str],
                  defaults: dict[str, float] | None = None, columns: int = 4) -> dict[str, float]:
    """Type how much each attribute counts (decimals). Add or remove attributes
    with the picker. Returns raw values (rescaled to sum to 1.00 when used)."""
    active = [a for a in sorted(weights, key=lambda a: -weights[a]) if weights[a] > 0 and a in allowed]
    chosen = st.multiselect("Attributes that count", allowed, default=active, key=f"{key}_attrs",
                            format_func=pretty, help="Add or remove attributes, then type their weights below.")
    cols = st.columns(columns)
    out = {}
    for i, a in enumerate(chosen):
        d = (defaults or {}).get(a)
        out[a] = cols[i % columns].number_input(
            pretty(a), min_value=0.0, max_value=1.0, value=round(float(weights.get(a, 0.10)), 2), step=0.05,
            format="%.2f", key=f"{key}_{a}", help=f"Default {d:.2f}" if d else "Not in the default weights")
    st.caption(f"Total **{sum(out.values()):.2f}** — rescaled to 1.00 when used, so only the proportions matter.")
    return out


def position_weight_editor(ctx: pipeline.Context, formation: str, base: dict[str, dict], key: str
                           ) -> dict[str, dict]:
    """One position at a time, remembered while you switch positions."""
    store = st.session_state.setdefault(f"{key}_store", {})
    defaults = pipeline.formation_defaults(ctx, formation)
    positions = pipeline.formation_positions(ctx, formation)
    pos = st.segmented_control("Position", positions, default=positions[0], key=f"{key}_pos",
                               format_func=lambda p: f"{p} · {pipeline.role_name(ctx, formation, p)}") or positions[0]
    allowed = GK_ATTRS if pos == "GK" else OUTFIELD_ATTRS
    store[pos] = weight_inputs(f"{key}_{formation}_{pos}", store.get(pos, base[pos]), allowed, defaults[pos])
    if st.button("Reset every position to the formation defaults", key=f"{key}_reset", type="tertiary",
                 icon=":material/restart_alt:"):
        st.session_state.pop(f"{key}_store", None)
        for k in [k for k in st.session_state if str(k).startswith(f"{key}_{formation}_")]:
            del st.session_state[k]
        st.rerun()
    return {p: store.get(p, base[p]) for p in positions}


def pitch(ctx: pipeline.Context, team: Team, highlight: set[str] | None = None,
          ratings: dict[str, float] | None = None) -> None:
    highlight = highlight or set()
    lines = "".join(
        f'<div class="fta-line" style="{css}"></div>' for css in (
            "inset:3%", "left:3%;right:3%;top:50%;height:0;border-width:2px 0 0",
            "left:36%;right:36%;top:41%;height:18%;border-radius:50%",
            "left:24%;right:24%;top:3%;height:14%;border-top:0",
            "left:24%;right:24%;bottom:3%;height:14%;border-bottom:0",
        ))
    templates = pipeline.team_templates(ctx, team) if team.formation in ctx.formations else ctx.templates
    chips = []
    for i, s in enumerate(team.slots):
        pid = s.player_id
        x, y = SLOT_XY.get(s.slot_id, (10 + 8 * i, 96))
        p = ctx.lookup.get(pid) if pid else None
        if ratings is not None and pid in ratings:
            bg, fg = rating_style(ratings[pid])
            extra = f'<span class="rt" style="background:{bg};color:{fg}">{ratings[pid]:.1f}</span>'
        else:
            extra = f"<small>fit {fit_score(p, templates[s.position]):.0f}</small>" if p else ""
        cls = "fta-chip hl" if s.slot_id in highlight else "fta-chip"
        chips.append(f'<div class="{cls}" style="left:{x}%;top:{y}%"><small>{html.escape(s.slot_id)}</small>'
                     f"<b>{html.escape(player_name(ctx, pid))}</b>{extra}</div>")
    st.html(f'<div class="fta-pitch">{lines}{"".join(chips)}</div>')


def attribute_bars(profile: dict) -> None:
    c = colors()
    by_attr = {a["attribute"]: a for a in profile["attributes"]}
    groups = [(g, [a for a in attrs if a in by_attr]) for g, attrs in ATTRIBUTE_GROUPS.items()]
    groups = [(g, attrs) for g, attrs in groups if attrs]
    cols = st.columns(2)
    for i, (group, attrs) in enumerate(groups):
        rows = "".join(
            f'<div class="fta-attr"><span>{pretty(a)}</span><div class="track" style="background:{c["track"]}">'
            f'<div class="fill" style="width:{by_attr[a]["value"]:.0f}%;background:{c["bar"]}"></div></div>'
            f'<b>{by_attr[a]["value"]:.0f}</b><span class="pct">beats {by_attr[a]["percentile"]:.0f}%</span></div>'
            for a in attrs)
        with cols[i % 2]:
            st.html(f'<div style="margin-bottom:10px"><div style="font-weight:600;font-size:13px;margin-bottom:2px">'
                    f'{group}</div>{rows}</div>')
    st.caption(f"“Beats X%” = share of every {profile['main_position']} in the pool with a lower value.")


def stat_bars(log: EventLog, stats: dict[str, dict[str, float]]) -> None:
    c = colors()
    a, b = log.team_a, log.team_b
    rows = []
    for label in stats[a]:
        va, vb = stats[a][label], stats[b][label]
        share = 100 * va / (va + vb) if (va + vb) else 50
        home, away = (c["home"], c["away"]) if (va + vb) else (c["track"], c["track"])
        fmt = "{:.2f}" if label.startswith("Expected") else "{:.0f}"
        rows.append(
            f'<div class="fta-stat"><span class="v">{fmt.format(va)}</span><div><div class="lbl">{label}</div>'
            f'<div class="fta-bar"><span style="width:{share}%;background:{home}"></span>'
            f'<span style="width:{100 - share}%;background:{away}"></span></div></div>'
            f'<span class="v" style="text-align:right">{fmt.format(vb)}</span></div>')
    st.html("".join(rows))


def scoreboard(ctx: pipeline.Context, log: EventLog) -> None:
    c = colors()
    goals = {log.team_a: [], log.team_b: []}
    for p in log.possessions:
        for e in p.chain:
            if e.event == "shot" and e.outcome == "goal":
                goals[p.attacking_team].append(f"⚽ {html.escape(player_name(ctx, e.player_id))} {minute_of(p)}'")

    def side(tid: str, color: str) -> str:
        return (f'<div><div class="team" style="color:{color}">{html.escape(log.name_v(tid))}</div>'
                f'<div class="scorers">{"<br>".join(goals[tid]) or "&nbsp;"}</div></div>')

    st.html(f'<div class="fta-score">{side(log.team_a, c["home"])}'
            f'<div class="num">{log.final_score[log.team_a]} – {log.final_score[log.team_b]}</div>'
            f'{side(log.team_b, c["away"])}</div>')


def change_lines(ctx: pipeline.Context, changes: list[dict]) -> str:
    return "  \n".join(f"**{c['slot']}**: {player_name(ctx, c['old_player_id'])} → "
                       f"{player_name(ctx, c['new_player_id'])}" for c in changes)


def fit_explainer(ctx: pipeline.Context, team: Team, key: str) -> None:
    with st.popover("Fit", icon=":material/help:", help="How fit is calculated"):
        st.markdown("**Fit** = Σ (weight × attribute) for the player's position, with the weights adding up to "
                    "1.00. **Average fit** is the mean over the XI. Weights come from the formation's roles and "
                    "can be tuned under *Position weights*.")
        templates = pipeline.team_templates(ctx, team)
        slot = st.selectbox("Breakdown for", [s.slot_id for s in team.slots], key=f"{key}_fitslot")
        s = next(s for s in team.slots if s.slot_id == slot)
        p = ctx.lookup.get(s.player_id)
        if p:
            rows, _, fit = fit_breakdown(p, templates[s.position])
            st.dataframe(pd.DataFrame([{"Attribute": pretty(r["attribute"]), "Weight": r["weight"],
                                        "Value": r["value"], "Points": r["contribution"]} for r in rows]),
                         hide_index=True, column_config={"Weight": st.column_config.NumberColumn(format="%.2f"),
                                                         "Points": st.column_config.NumberColumn(format="%.2f")})
            st.markdown(f"{p.name} at {slot}: **{fit:.2f}**")


def chemistry_explainer(ctx: pipeline.Context, team: Team, key: str) -> None:
    with st.popover("Chemistry", icon=":material/help:", help="How chemistry is calculated"):
        score, items = pipeline.chemistry_details(ctx, team)
        st.markdown(f"Starts at **100**, plus four rules. In matches, every point above 100 adds "
                    f"**{CHEMISTRY_EFFECT}** to each of the team's duels.")
        st.dataframe(pd.DataFrame([{"Rule": i["label"], "Points": i["points"], "Max": i["max"], "Now": i["detail"]}
                                   for i in items]), hide_index=True, key=f"{key}_chem",
                     column_config={"Points": st.column_config.NumberColumn(format="%+.1f"),
                                    "Max": st.column_config.NumberColumn(format="%+.0f")})
        st.markdown(f"Total **{score:.1f}**")
        for i in items:
            st.caption(f"**{i['label']}** — {i['description']}")


# ---------------------------------------------------------------------------
# Player dialog
# ---------------------------------------------------------------------------

def weight_sources(ctx: pipeline.Context, position: str, extra: dict | None = None) -> dict[str, dict]:
    out = dict(extra or {})
    for formation in ctx.formation_weights:
        if position in pipeline.formation_positions(ctx, formation):
            out[f"{formation} · {pipeline.role_name(ctx, formation, position)}"] = \
                pipeline.formation_defaults(ctx, formation)[position]
    for tid in storage.list_teams():
        t = storage.load_team(tid)
        if position in pipeline.team_weights(ctx, t):
            out[f"{t.name} (team weights)"] = pipeline.team_weights(ctx, t)[position]
    out[f"Base {position} weights"] = ctx.templates[position].attribute_weights
    out["Custom (type your own)"] = None
    return out


def weight_picker(ctx: pipeline.Context, side: str, positions: list[str], default_pos: str,
                  extra: dict | None, is_gk: bool) -> tuple[str, dict[str, float]]:
    c1, c2 = st.columns(2)
    pos = c1.selectbox("Position", positions, index=positions.index(default_pos), key=f"cmp_pos_{side}")
    sources = weight_sources(ctx, pos, extra if pos == default_pos else None)
    label = c2.selectbox("Weight set", list(sources), key=f"cmp_src_{side}")
    weights = sources[label]
    if weights is None:
        weights = weight_inputs(f"cmp_custom_{side}_{pos}", ctx.templates[pos].attribute_weights,
                                GK_ATTRS if is_gk else OUTFIELD_ATTRS, ctx.templates[pos].attribute_weights, 3)
        if not weights:
            st.warning("Give at least one attribute a weight.")
            weights = ctx.templates[pos].attribute_weights
    return pos, pipeline.normalize_weights({"x": weights})["x"]


def compare_view(ctx: pipeline.Context, pid: str, compare_with: str | None, position: str | None,
                 weights: dict | None) -> None:
    me = ctx.lookup[pid]
    pool_ids = sorted((p.player_id for p in ctx.pool if p.is_gk() == me.is_gk() and p.player_id != pid),
                      key=lambda x: ctx.lookup[x].name)
    if st.session_state.get("cmp_other") not in pool_ids:
        st.session_state["cmp_other"] = compare_with if compare_with in pool_ids else pool_ids[0]
    other = ctx.lookup[st.selectbox(
        "Compare with", pool_ids, key="cmp_other",
        format_func=lambda x: f"{ctx.lookup[x].name} · {'/'.join(ctx.lookup[x].natural_positions)} · "
                              f"age {ctx.lookup[x].age}")]
    positions = ["GK"] if me.is_gk() else [p for p in ctx.templates if p != "GK"]
    default_pos = position or next((p for p in me.natural_positions if p in positions), positions[0])
    extra = {"This request's weights": weights} if weights else None
    mode = st.segmented_control("Weights", ["Same for both", "Different per player"], default="Same for both",
                                key="cmp_mode") or "Same for both"
    if mode == "Same for both":
        pos_a, wa = weight_picker(ctx, "a", positions, default_pos, extra, me.is_gk())
        wb = wa
    else:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.caption(f"Weights for **{me.name}**")
            pos_a, wa = weight_picker(ctx, "a", positions, default_pos, extra, me.is_gk())
        with c2:
            st.caption(f"Weights for **{other.name}**")
            _, wb = weight_picker(ctx, "b", positions, default_pos, extra, me.is_gk())

    base_t = ctx.templates[pos_a]
    rows_a, _, fit_a = fit_breakdown(me, base_t.model_copy(update={"attribute_weights": wa}))
    rows_b, _, fit_b = fit_breakdown(other, base_t.model_copy(update={"attribute_weights": wb}))
    m1, m2 = st.columns(2)
    m1.metric(me.name, f"{fit_a:.2f}", f"{fit_a - fit_b:+.2f}")
    m2.metric(other.name, f"{fit_b:.2f}", f"{fit_b - fit_a:+.2f}")
    better, margin = (me.name, fit_a - fit_b) if fit_a >= fit_b else (other.name, fit_b - fit_a)
    st.markdown(f"**{better}** fits better by **{margin:.2f}**"
                + (" — each judged against his own weights." if mode != "Same for both" else "."))
    ra, rb = {r["attribute"]: r for r in rows_a}, {r["attribute"]: r for r in rows_b}
    attrs = sorted(set(ra) | set(rb), key=lambda a: -max(ra.get(a, {}).get("weight", 0), rb.get(a, {}).get("weight", 0)))
    df = pd.DataFrame([{
        "Attribute": pretty(a), "Weight A": ra[a]["weight"] if a in ra else 0.0,
        me.name: me.attr(a), other.name: other.attr(a), "Weight B": rb[a]["weight"] if a in rb else 0.0,
        "Points A": ra[a]["contribution"] if a in ra else 0.0, "Points B": rb[a]["contribution"] if a in rb else 0.0,
    } for a in attrs])
    df["Edge"] = df["Points A"] - df["Points B"]
    if mode == "Same for both":
        df = df.drop(columns=["Weight B"]).rename(columns={"Weight A": "Weight"})
    dec = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(df, hide_index=True, column_config={
        "Weight": dec, "Weight A": dec, "Weight B": dec, "Points A": dec, "Points B": dec,
        "Edge": st.column_config.NumberColumn(format="%+.2f", help="Points A − Points B")})

    st.markdown("**Best fits in the pool for these weights**")
    top = pipeline.best_fits(ctx, wa, goalkeepers=me.is_gk(), top_n=8)

    def compare_with_player(target: str) -> None:
        st.session_state["cmp_other"] = target

    player_rows(ctx, [r["player_id"] for r in top],
                [("Player", 2.2), ("Positions", 1), ("Age", 0.5), ("Foot", 0.6), ("Fit", 0.6)],
                lambda p, i: [f"**{ctx.lookup[p].name}**" + (" (this player)" if p == pid else ""),
                              "/".join(ctx.lookup[p].natural_positions), str(ctx.lookup[p].age),
                              ctx.lookup[p].preferred_foot, f"**{top[i]['fit']:.1f}**"],
                key="cmp_best", action=("Compare", compare_with_player))


def scouting_view(ctx: pipeline.Context, pid: str) -> None:
    a = pipeline.scouting_assessment(ctx, pid)
    bg, fg = VERDICT_STYLE[a["verdict"]]
    st.html(f'<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap">'
            f'<span class="fta-verdict" style="background:{bg};color:{fg}">{a["verdict"]}</span>'
            f'<span style="font-size:18px"><b>{a["overall"]:.1f}</b>/10 as a {a["main_position"]}</span>'
            f'<span class="fta-pill">{a["age"]} · {a["age_profile"]["stage"]}</span>'
            f'<span class="fta-pill">best as {html.escape(a["best_role"])}</span></div>')
    st.caption(f"{a['verdict_rule']} {a['age_profile']['note']}")
    c = colors()
    rows = []
    for crit in a["criteria"]:
        rows.append(
            f'<div class="fta-crit"><b>{crit["name"]}</b><span class="fta-pill">{100 * crit["share"]:.0f}%</span>'
            f'<div class="track" style="background:{c["track"]}"><div class="fill" '
            f'style="width:{max(2, crit["score"] * 10)}%;background:{c["bar"]}"></div></div>'
            f'<b>{crit["score"]:.1f}</b></div>'
            f'<div class="fta-note">{html.escape(crit["detail"])}</div>')
    st.html("".join(rows))
    with st.popover("How it's scored", icon=":material/calculate:"):
        st.markdown(
            f"- Only criteria that matter for a **{a['main_position']}** are used — a defender is never judged on "
            "finishing, a striker never on tackling.\n"
            "- **Score** = average percentile of the criterion's attributes against every player at the same "
            "position, out of 10 (8.0 = better than 80% of them).\n"
            "- The % is the criterion's share of the overall. **Match form** (average rating plus the role's key "
            "numbers) joins once he plays, growing to 30% of the overall over his first 5 matches.\n"
            "- **Verdict**: 8+ and ≤31 → Sign · 8+ and 32+ → Short-term signing · 6+ → Monitor · else Pass.\n"
            "- The assistant's report below can't change a score or the verdict.")

    st.divider()
    st.markdown("**Assistant's scouting report**")
    teams = [t for t in storage.list_teams() if pid not in storage.load_team(t).player_ids()]
    c1, c2 = st.columns([3, 1.4], vertical_alignment="bottom")
    focus = c1.selectbox("Scout for", [None] + teams, key="scout_focus",
                         format_func=lambda t: "The player only (no team)" if t is None else f"{team_name(t)} — "
                                               "also test him in this team",
                         help="The report is about the player himself. Pick a team to also test where he'd "
                              "play in it and what he'd change (measured by code).")
    cached = pipeline.load_scouting(pid, focus)
    if c2.button("Refresh" if cached else "Scout with the assistant", key="scout_btn", width="stretch",
                 icon=":material/auto_awesome:",
                 help="Strengths, risks and roles, compared with the best alternatives at his position"):
        with st.spinner("Scouting..."):
            pipeline.generate_scouting(ctx, pid, model=model_id(), tracker=make_tracker(), team_id=focus)
        st.rerun(scope="fragment")
    if not cached:
        st.caption("Not scouted yet" + (f" for {team_name(focus)}." if focus else "."))
        return
    an = {**cached["analysis"], "generated_at": cached["generated_at"]}
    if cached["matches_seen"] != a["form"]["matches"]:
        st.caption(":orange[He has played since this report — refresh it.]")
    st.markdown(an["summary"])
    assistant_status(an)
    s1, s2 = st.columns(2)
    s1.markdown("\n".join(f"- :green[✓] {x}" for x in an["strengths"]) or "–")
    s2.markdown("\n".join(f"- :orange[✗] {x}" for x in an["risks"]) or "–")
    if an["compared_with"]:
        st.markdown("**Compared with**  \n" + "  \n".join(
            f"{player_name(ctx, x['player_id'])} — {x['note']}" for x in an["compared_with"]))
    if an["fit_for_teams"]:
        st.markdown(f"**In {team_name(focus)}** · tested by code")
        for i, f in enumerate(an["fit_for_teams"]):
            c1, c2 = st.columns([6, 1.3], vertical_alignment="center")
            c1.markdown(f"**{team_name(f['team_id'])} {f['slot']}** · {f['verdict']} — {f.get('reason', '')}  \n"
                        + measured_text(f.get("measured")))
            if f["verdict"] != "invalid" and c2.button("Stage", key=f"scout_stage_{i}", icon=":material/add_circle:",
                                                       help="Add him to that team's pending changes"):
                error = stage_many(f["team_id"], {f["slot"]: pid})
                if error:
                    st.error(error)
                else:
                    st.session_state["select_team"] = f["team_id"]
                    st.session_state["flash"] = f"Staged {player_name(ctx, pid)} for {f['slot']} — save it in Squad"
                    st.rerun()
        st.caption("Verdict from the test: upgrade = better over paired simulations; squad option = fits the slot "
                   "at least 2.0 better but no proven result; otherwise not needed.")
    trace_popover(an)


@st.dialog("Player", width="large")
def player_dialog(pid: str, compare_with: str | None = None, position: str | None = None,
                  weights: dict | None = None) -> None:
    ctx = st.session_state["_ctx"]
    prof = pipeline.player_profile(ctx, pid)
    stars = "★" * prof["weak_foot"] + "☆" * (5 - prof["weak_foot"])
    st.markdown(f"### {prof['name']}")
    st.caption(f"{' / '.join(prof['positions'])} · {prof['foot']}-footed (weak foot {stars}) · age {prof['age']} "
               f"· {prof['nationality']} · base stamina {prof['stamina_base']}")
    t_profile, t_scout, t_matches, t_compare = st.tabs(["Profile", "Scouting report", "Matches", "Compare"])

    with t_profile:
        attribute_bars(prof)
        c1, c2 = st.columns([2, 3])
        fits = pd.DataFrame([{"Position": f["position"], "Fit": f["fit"], "Natural": "✓" if f["natural"] else ""}
                             for f in prof["position_fits"]])
        c1.markdown("**Fit by position**")
        c1.dataframe(fits, hide_index=True, height=full_height(len(fits)),
                     column_config={"Fit": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f")})
        with c2:
            st.markdown(f"**Stands out** among {prof['main_position']}s")
            for a in prof["strengths"]:
                st.markdown(f":material/arrow_upward: {pretty(a['attribute'])} {a['value']:.0f} "
                            f"— beats {a['percentile']:.0f}%")
            st.markdown("**Weaker areas**")
            for a in prof["weaknesses"]:
                st.markdown(f":material/arrow_downward: {pretty(a['attribute'])} {a['value']:.0f} "
                            f"— beats only {a['percentile']:.0f}%")
            if not prof["weaknesses"] and prof["lowest_relevant"]:
                low = prof["lowest_relevant"]
                st.caption(f"Nothing below the {prof['main_position']} median. Lowest relevant attribute: "
                           f"{pretty(low['attribute'])} {low['value']:.0f} (beats {low['percentile']:.0f}%).")

    with t_scout:
        scouting_view(ctx, pid)

    with t_matches:
        history = pipeline.player_history(pid)
        if not history:
            st.info("No matches played yet. Matches feed his scouting report as *Match form*.")
        else:
            reps = [h["report"] for h in history if h["report"]]
            m = st.columns(4)
            m[0].metric("Matches", len(history))
            m[1].metric("Avg rating", f"{sum(r.rating for r in reps) / len(reps):.2f}" if reps else "–")
            m[2].metric("Goals", sum(r.stats.goals for r in reps))
            m[3].metric("Chances created", sum(r.stats.key_passes for r in reps))
            st.dataframe(pd.DataFrame([{
                "Match": h["label"], "For": h["team"], "Result": h["result"], "Slot": h["slot"],
                "Rating": h["report"].rating if h["report"] else None,
                "Went well": h["report"].strengths[0] if h["report"] else "",
                "To improve": h["report"].weaknesses[0] if h["report"] else "",
            } for h in history]), hide_index=True, column_config={
                "Rating": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f")})

    with t_compare:
        compare_view(ctx, pid, compare_with or prof["best_rival"], position, weights)


# ---------------------------------------------------------------------------
# Drafts: pending changes per team, saved together as one version
# ---------------------------------------------------------------------------

def drafts() -> dict:
    return st.session_state.setdefault("drafts", {})


def draft_state(ctx: pipeline.Context, team_id: str) -> tuple[Team, Team, dict, bool]:
    """(saved latest version, draft preview, pending changes, is_stale)."""
    saved = pipeline.rescored(ctx, storage.load_team(team_id))
    d = drafts().get(team_id)
    if not d or not d["changes"]:
        return saved, saved, {}, False
    if d["base_version"] != saved.version:
        return saved, saved, d["changes"], True
    return saved, pipeline.preview_swaps(ctx, saved, d["changes"]), d["changes"], False


def draft_signature(team_id: str) -> tuple:
    changes = (drafts().get(team_id) or {}).get("changes") or {}
    return storage.list_teams()[team_id]["latest_version"], tuple(sorted(changes.items()))


def stage_change(team: Team, slot: str, player_id: str) -> None:
    d = drafts().setdefault(team.team_id, {"base_version": team.version, "changes": {}})
    original = next(s.player_id for s in team.slots if s.slot_id == slot)
    if player_id == original:
        d["changes"].pop(slot, None)
    else:
        d["changes"][slot] = player_id


def pending_panel(ctx: pipeline.Context, team_id: str) -> None:
    """Pending changes: list, undo, save as one version, discard."""
    saved, _, changes, stale = draft_state(ctx, team_id)
    if stale:
        st.warning(f"{saved.name} changed (now v{saved.version}) since these edits were staged.")
        if st.button("Discard the old edits", key="stale_discard"):
            drafts().pop(team_id, None)
            st.rerun()
        return
    if not changes:
        return
    with st.container(border=True):
        st.markdown(f"**{len(changes)} pending change(s)** — not saved yet")
        old = {s.slot_id: s.player_id for s in saved.slots}
        for slot, pid in changes.items():
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            c1.markdown(f"{slot}: {player_name(ctx, old[slot])} → **{player_name(ctx, pid)}**")
            if c2.button("Undo", key=f"undo_{slot}", type="tertiary"):
                drafts()[team_id]["changes"].pop(slot)
                st.rerun()
        b1, b2 = st.columns(2)
        if b1.button(f"Save as v{storage.next_version_number(team_id)}", type="primary", key="save_draft",
                     icon=":material/save:",
                     width="stretch"):
            new_team, _ = pipeline.apply_swaps(ctx, team_id, changes, reason="changed in the UI")
            drafts().pop(team_id, None)
            st.session_state["flash"] = f"Saved {new_team.name} v{new_team.version}"
            st.rerun()
        if b2.button("Discard", key="discard_draft", width="stretch"):
            drafts().pop(team_id, None)
            st.rerun()


def test_panel(ctx: pipeline.Context, team_id: str) -> None:
    """Saved XI vs draft over many matches, same seeds for both, fresh seeds every run."""
    saved, draft, changes, stale = draft_state(ctx, team_id)
    opponents = [t for t in storage.list_teams() if t != team_id]
    if not changes or stale:
        st.caption("Stage at least one change first — this compares your saved XI with the edited one.")
        return
    if not opponents:
        st.caption("Build a second team to test against.")
        return
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    opp = c1.selectbox("Opponent", opponents, format_func=team_label, key="test_opp")
    n = c2.selectbox("Matches", [200, 500, 1000], key="test_n")
    if pipeline.missing_players(ctx, storage.load_team(opp)):
        st.caption(f"{team_name(opp)} uses players from a different player pool — pick another opponent.")
        return
    if c3.button("Run test", icon=":material/science:", width="stretch"):
        with st.spinner(f"Playing {2 * n} matches..."):
            res = pipeline.evaluate_lineups(ctx, {f"Saved v{saved.version}": saved, "With changes": draft},
                                            pipeline.rescored(ctx, storage.load_team(opp)), n=n)
        st.session_state["test_result"] = (draft_signature(team_id), opp, res)
    cached = st.session_state.get("test_result")
    if cached and cached[0] == draft_signature(team_id):
        res = cached[2]
        st.dataframe(pd.DataFrame(res["variants"]).T, column_config={
            k: st.column_config.NumberColumn(format="%.2f") for k in ("Points per match", "Goals for",
                                                                      "Goals against")})
        d = res["difference"]
        icon = {"better": ":green[▲ better]", "worse": ":orange[▼ worse]"}.get(d["verdict"], "no clear difference")
        st.markdown(f"With changes: **{d['points']:+.2f} points per match** (± {d['margin']:.2f}) → {icon}")
        st.caption(f"vs {team_name(cached[1])} over {d['matches']} matches each. Both line-ups play the same "
                   "random seeds, so only your changes differ; every run uses new seeds, so numbers vary a little "
                   "between runs. The ± range is the 95% margin — if it includes 0, the change isn't proven.")


# ---------------------------------------------------------------------------
# Change players: request -> interpretation -> ranked shortlist -> pick
# ---------------------------------------------------------------------------

def interpretation_card(ctx: pipeline.Context, team: Team, request: SwapRequest, info: dict | None) -> None:
    source = request.interpreted_by or "manual"
    who = {"keywords": "keyword matching (offline)", "performance": "his match performance",
           "human-edited": "you"}.get(source, source.replace("llm:", "AI · "))
    rows = pipeline.weight_changes(ctx, team, request)
    changed = [r for r in rows if r["requested"]]
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown(f"**Weights** · read by {who}")
        if changed:
            st.dataframe(pd.DataFrame([{
                "Attribute": pretty(r["attribute"]), "Before": r["before"], "After": r["after"],
                "Why": r["reason"]} for r in changed]), hide_index=True,
                column_config={"Before": st.column_config.NumberColumn(format="%.2f"),
                               "After": st.column_config.NumberColumn(format="%.2f")})
        else:
            st.caption("No attribute emphasis — ranked on the position's normal weights.")
        kept = [r for r in rows if not r["requested"]]
        if kept and changed:
            st.caption("Unchanged (only rescaled): " + ", ".join(
                f"{pretty(r['attribute'])} {r['before']:.2f}→{r['after']:.2f}" for r in kept))
    with c2:
        st.markdown("**Requirements**")
        if request.hard_filters:
            for f in request.hard_filters:
                why = request.adjustment_reasons.get(f"filter:{f.attribute}", "")
                st.markdown(f":material/filter_alt: {describe_filter(f)}" + (f" — _{why}_" if why else ""))
            st.caption("Players who miss one are ranked lower, never hidden.")
        else:
            st.caption("None.")
    if info and info.get("error"):
        st.caption(f":orange[The AI couldn't be used — {info['error']}. Read with keyword matching instead; "
                   "check the result under *Adjust*.]")
    if info and info.get("source", "").startswith("llm:"):
        missed = sorted(set(info.get("keyword_check", {})) - set(request.reweight))
        if missed:
            st.caption(f":orange[Possible miss:] your words also suggest {', '.join(map(pretty, missed))}, which "
                       "the AI didn't change — add them under *Adjust* if you meant them.")
        else:
            st.caption(":green[✓ Every change quotes your request, and nothing your words point to was missed.]")
    for item in (info or {}).get("ignored", []):
        st.caption(f":orange[Rejected from the AI reply — {item}]")
    if info and info.get("sent"):
        with st.popover("What the AI saw", icon=":material/visibility:"):
            st.caption("The exact context sent with your request, and the model's raw reply. The reply is "
                       "validated before anything is used.")
            st.json(info["sent"], expanded=1)
            st.markdown("**Raw reply**")
            st.code(info.get("raw") or "(no reply)", language="json")


def request_controls(ctx: pipeline.Context, team: Team, state_key: str) -> None:
    state = st.session_state[state_key]
    request: SwapRequest = state["request"]
    position = pipeline.slot_position(team, request.target_slot)
    with st.popover("Adjust", icon=":material/tune:", help="Edit the weights, requirements or ranking"):
        rows = pipeline.weight_changes(ctx, team, request)
        weights = weight_inputs(f"{state_key}_w", {r["attribute"]: r["after"] for r in rows},
                                GK_ATTRS if position == "GK" else OUTFIELD_ATTRS,
                                {r["attribute"]: r["before"] for r in rows if r["before"]}, 3)
        current = {f.attribute: f for f in request.hard_filters}
        c1, c2, c3 = st.columns(3)
        foot_now = current["preferred_foot"].value if "preferred_foot" in current else None
        foot = c1.segmented_control("Preferred foot", ["Any", "Left", "Right"],
                                    default=(foot_now or "any").capitalize(), key=f"{state_key}_foot") or "Any"
        age_f = current.get("age")
        ages = c2.slider("Age range", 15, 45, (int(age_f.min or 15) if age_f else 15,
                                                int(age_f.max or 45) if age_f else 45), key=f"{state_key}_age")
        nats = sorted({p.nationality for p in ctx.pool})
        nat_f = current.get("nationality")
        nat_now = [n for n in ((nat_f.values or [nat_f.value]) if nat_f else []) if n in nats]
        nat = c3.multiselect("Nationality (any of)", nats, default=nat_now, key=f"{state_key}_nat",
                             placeholder="Any")
        natural = st.toggle(f"Must be a natural {position}", value="natural_position" in current,
                            key=f"{state_key}_natural")
        c1, c2 = st.columns(2)
        strict_now = next((k for k, v in STRICTNESS.items() if v == request.violation_penalty), "Prefer")
        strict = c1.select_slider("How strict are the requirements?", list(STRICTNESS), value=strict_now,
                                  key=f"{state_key}_strict",
                                  help="Points taken off the rank score for each requirement a player misses.")
        chem_w = c2.number_input("Chemistry weight", 0.0, 2.0, float(request.chemistry_weight), 0.25,
                                 format="%.2f", key=f"{state_key}_cw",
                                 help="The rank score adds this × the change in team chemistry.")
        if st.button("Re-rank", key=f"{state_key}_rerank", icon=":material/refresh:", type="primary"):
            filters = [f for f in request.hard_filters
                       if f.attribute not in ("preferred_foot", "age", "nationality", "natural_position")]
            if foot != "Any":
                filters.append(HardFilter(attribute="preferred_foot", value=foot.lower()))
            if ages != (15, 45):
                filters.append(HardFilter(attribute="age", min=ages[0] if ages[0] > 15 else None,
                                          max=ages[1] if ages[1] < 45 else None))
            if nat:
                filters.append(HardFilter(attribute="nationality", value=nat[0]) if len(nat) == 1
                               else HardFilter(attribute="nationality", values=nat))
            if natural:
                filters.append(HardFilter(attribute="natural_position", value=position))
            new = request.model_copy(update={
                "reweight": weights or request.reweight, "hard_filters": filters,
                "violation_penalty": STRICTNESS[strict], "chemistry_weight": chem_w, "interpreted_by": "human-edited",
            })
            request_shortlist(ctx, state["team_id"], new, state_key, state.get("info"))
            st.rerun()


def candidate_picker(ctx: pipeline.Context, state_key: str) -> None:
    state = st.session_state.get(state_key)
    if not state:
        return
    shortlist: Shortlist = state["shortlist"]
    request: SwapRequest = state["request"]
    team_id = state["team_id"]
    if state["signature"] != draft_signature(team_id):
        st.caption(":orange[The team changed since this shortlist was made — ask again.]")
        return
    _, draft, _, _ = draft_state(ctx, team_id)
    position = pipeline.slot_position(draft, shortlist.target_slot)
    interpretation_card(ctx, draft, request, state.get("info"))

    current = shortlist.current_player_id
    cands = shortlist.candidates
    # the current player on the SAME (request) weights as the list, so the numbers compare like for like
    now_fit = round(cands[0].fit_score - cands[0].delta_fit, 1) if cands and current else None
    c1, c2, c3 = st.columns([4, 1, 1], vertical_alignment="center")
    c1.markdown(f"**Shortlist for {shortlist.target_slot}** · now: {player_name(ctx, current)}"
                + (f" — fit **{now_fit:.1f}** on these weights" if now_fit is not None else "")
                + (f" ({shortlist.current_fit_score:.1f} on the team's usual weights)"
                   if shortlist.current_fit_score is not None and now_fit is not None
                   and abs(shortlist.current_fit_score - now_fit) >= 0.05 else ""))
    with c2:
        request_controls(ctx, draft, state_key)
    with c3.popover("Rank vs fit", icon=":material/leaderboard:"):
        st.markdown(
            "**Fit** — how well a player's attributes match the weights above (0–100): Σ weight × attribute. "
            "It says who is the better player *for this request*.\n\n"
            "**Rank** — the order of the list: **Fit − Penalty + Chem.** It also counts what else matters:\n"
            f"- **Penalty**: {request.violation_penalty:g} points per requirement he misses (foot, age, "
            "nationality…) — set by *How strict* under Adjust. Nobody is hidden.\n"
            f"- **Chem.**: {request.chemistry_weight:g} × the change in team chemistry if he comes in "
            "(links with neighbours, back-line feet, midfield stamina, shared nationality).\n\n"
            "So a player with a higher fit can rank lower when he misses a requirement or hurts chemistry.")
        if cands:
            c = cands[0]
            st.caption(f"#1 {player_name(ctx, c.player_id)}: {c.fit_score:.2f} − {c.penalty:.1f} "
                       f"{c.chemistry_bonus:+.1f} = {c.rank_score:.2f}")
    st.caption(f"Sorted by **Rank** = Fit − Penalty + Chem. · **Fit** (± vs {player_name(ctx, current)} on the "
               "same weights) · **Penalty** per requirement missed · **Chem.** = change in team chemistry × "
               f"{request.chemistry_weight:g}")
    request_weights = {r["attribute"]: r["after"] for r in pipeline.weight_changes(ctx, draft, request)
                       if r["after"] > 0}
    player_rows(
        ctx, [c.player_id for c in cands],
        [("#", 0.3), ("Player", 2), ("Age · foot", 0.9), ("Rank", 0.7), ("Fit (±)", 1.1), ("Penalty", 0.6),
         ("Chem.", 0.6), ("Requirements", 2.3)],
        lambda pid, i: [f"{i + 1}", f"**{ctx.lookup[pid].name}**",
                        f"{ctx.lookup[pid].age} · {ctx.lookup[pid].preferred_foot[0].upper()}",
                        f"**{cands[i].rank_score:.2f}**",
                        f"{cands[i].fit_score:.1f} " + (f":green[({cands[i].delta_fit:+.1f})]" if cands[i].delta_fit > 0
                                                        else f":orange[({cands[i].delta_fit:+.1f})]"),
                        f"−{cands[i].penalty:.1f}" if cands[i].penalty else "0",
                        f"{cands[i].chemistry_bonus:+.1f}",
                        ":orange[⚠ " + "; ".join(cands[i].warnings) + "]" if cands[i].warnings else ":green[✓ all met]"],
        key=f"{state_key}_rows",
        info_kwargs={"compare_with": current, "position": position, "weights": request_weights})

    force = "__force__"
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    choice = c1.radio("Pick", [c.player_id for c in cands] + [force], key=f"{state_key}_pick", horizontal=True,
                      format_func=lambda pid: "Someone else…" if pid == force else player_name(ctx, pid))
    if choice == force:
        on_team = set(draft.player_ids())
        others = sorted((p for p in ctx.pool if p.player_id not in on_team and p.is_gk() == (position == "GK")),
                        key=lambda p: p.name)
        choice = c1.selectbox("Any player in the pool", [p.player_id for p in others], key=f"{state_key}_force",
                              format_func=lambda pid: f"{ctx.lookup[pid].name} · "
                                                      f"{'/'.join(ctx.lookup[pid].natural_positions)} · "
                                                      f"{ctx.lookup[pid].preferred_foot} · age {ctx.lookup[pid].age}")
    if c2.button(f"Stage for {shortlist.target_slot}", type="primary", key=f"{state_key}_stage",
                 icon=":material/add_circle:", width="stretch"):
        stage_change(storage.load_team(team_id), shortlist.target_slot, choice)
        del st.session_state[state_key]
        st.session_state["flash"] = (f"Staged {player_name(ctx, choice)} for {shortlist.target_slot} — "
                                     f"save it in Squad")
        st.rerun()


def request_shortlist(ctx: pipeline.Context, team_id: str, request: SwapRequest, state_key: str,
                      info: dict | None = None) -> None:
    """Shortlist against the team *with* its pending changes."""
    _, draft, _, _ = draft_state(ctx, team_id)
    for k in [k for k in st.session_state if str(k).startswith(f"{state_key}_")]:
        del st.session_state[k]
    st.session_state[state_key] = {
        "team_id": team_id, "request": request, "info": info, "signature": draft_signature(team_id),
        "shortlist": pipeline.shortlist(ctx, draft, request),
    }


# ---------------------------------------------------------------------------
# Match helpers
# ---------------------------------------------------------------------------

def xg_chart(ctx: pipeline.Context, log: EventLog) -> alt.LayerChart:
    """Cumulative xG race over the 90 minutes, one step line per team, goals marked."""
    c = colors()
    ids = (log.team_a, log.team_b)
    names = {t: log.name(t) for t in ids}
    cum = dict.fromkeys(ids, 0.0)
    long_rows = [{"minute": 0, "team": names[t], "cum_xg": 0.0, "goal": False} for t in ids]
    wide_rows = {0: {"minute": 0, "a": 0.0, "b": 0.0}}
    for p in log.possessions:
        shot = next((e for e in p.chain if e.event == "shot"), None)
        if shot:
            cum[p.attacking_team] += shot.xg or 0.0
        for t in ids:
            mine = shot is not None and t == p.attacking_team
            long_rows.append({"minute": minute_of(p), "team": names[t], "cum_xg": round(cum[t], 3),
                              "goal": bool(mine and shot.outcome == "goal"),
                              "player": player_name(ctx, shot.player_id) if mine else "",
                              "xg": shot.xg if mine else None})
        wide_rows[minute_of(p)] = {"minute": minute_of(p), "a": round(cum[ids[0]], 2), "b": round(cum[ids[1]], 2)}
    long_df, wide_df = pd.DataFrame(long_rows), pd.DataFrame(list(wide_rows.values()))
    end = int(long_df.minute.max())
    last = long_df.groupby("team", sort=False).tail(1).copy()
    top = max(last.cum_xg.max(), 0.1)
    last["label_y"] = last.cum_xg
    if abs(last.cum_xg.iloc[0] - last.cum_xg.iloc[1]) < 0.08 * top:
        hi = last.cum_xg.idxmax()
        last.loc[hi, "label_y"] += 0.05 * top
        last.loc[last.index != hi, "label_y"] -= 0.05 * top
    last["label"] = last.team + "  " + last.cum_xg.map("{:.2f}".format)
    color = alt.Color("team:N", scale=alt.Scale(domain=[names[t] for t in ids], range=[c["home"], c["away"]]),
                      legend=alt.Legend(orient="top", title=None))
    x = alt.X("minute:Q", title="Minute", scale=alt.Scale(domain=[0, end]),
              axis=alt.Axis(values=list(range(0, end + 1, 15))))
    y = alt.Y("cum_xg:Q", title="Cumulative xG")
    base = alt.Chart(long_df).encode(x=x, y=y, color=color)
    hover = alt.selection_point(fields=["minute"], nearest=True, on="pointerover", clear="pointerout", empty=False)
    lines = base.mark_line(interpolate="step-after", strokeWidth=2)
    goals = base.transform_filter("datum.goal").mark_point(
        filled=True, size=110, opacity=1, stroke=c["surface"], strokeWidth=2,
    ).encode(tooltip=[alt.Tooltip("team:N", title="Goal"), alt.Tooltip("player:N", title="Scorer"),
                      alt.Tooltip("minute:Q", title="Minute"), alt.Tooltip("xg:Q", title="Chance xG", format=".2f")])
    catcher = alt.Chart(wide_df).mark_rule(strokeWidth=8, opacity=0).encode(
        x="minute:Q", tooltip=[alt.Tooltip("minute:Q", title="Minute"),
                               alt.Tooltip("a:Q", title=names[ids[0]], format=".2f"),
                               alt.Tooltip("b:Q", title=names[ids[1]], format=".2f")]).add_params(hover)
    rule = alt.Chart(wide_df).mark_rule(color=c["muted"], strokeWidth=1).encode(x="minute:Q").transform_filter(hover)
    dots = base.mark_point(filled=False, size=70, opacity=1, strokeWidth=2).transform_filter(hover)
    labels = alt.Chart(last).mark_text(align="left", dx=8, fontSize=12, color=c["muted"]).encode(
        x="minute:Q", y="label_y:Q", text="label:N")
    return (alt.layer(lines, rule, dots, goals, labels, catcher)
            .properties(height=280, padding={"left": 4, "top": 4, "right": 150, "bottom": 4})
            .configure_axis(grid=True, gridOpacity=0.25, domainOpacity=0.4, tickOpacity=0.4)
            .configure_view(strokeWidth=0))


def key_moments(ctx: pipeline.Context, log: EventLog) -> list[str]:
    out = []
    for p in log.possessions:
        shot = next((e for e in p.chain if e.event == "shot"), None)
        if not shot:
            continue
        creator = next((e for e in p.chain if e.event in ("through_ball", "cross", "dribble")), None)
        how = {"through_ball": "through ball", "cross": "header from a cross", "dribble": "solo run"}.get(
            creator.event if creator else "", "")
        assist = (f", set up by {player_name(ctx, creator.player_id)}"
                  if creator and creator.player_id != shot.player_id and creator.event != "dribble" else "")
        who = f"**{player_name(ctx, shot.player_id)}** ({log.name(p.attacking_team)})"
        if shot.outcome == "goal":
            out.append(f"`{minute_of(p)}'` ⚽ Goal — {who}, {how}{assist}")
        elif (shot.xg or 0) >= 0.3:
            out.append(f"`{minute_of(p)}'` Big chance missed — {who}, {how} ({shot.xg:.2f} xG)")
    return out


def event_log_table(ctx: pipeline.Context, log: EventLog) -> pd.DataFrame:
    return pd.DataFrame([{
        "Minute": minute_of(p), "Attacking": log.name_v(p.attacking_team), "Event": e.event.replace("_", " "),
        "Player": player_name(ctx, e.player_id), "For": log.name_v(e.team_id) if e.team_id else "",
        "Success": e.success, "Outcome": e.outcome, "xG": e.xg,
    } for p in log.possessions for e in p.chain])


def lineup_team(log: EventLog, team_id: str) -> Team:
    """The line-up as it played (old matches show who actually played)."""
    slots = [FormationSlot(slot_id=sid, position=log.slot_positions.get(team_id, {}).get(sid, "CM"), player_id=pid)
             for sid, pid in log.lineups.get(team_id, {}).items()]
    return Team(team_id=team_id, name=log.name(team_id), formation="", slots=slots)


def stat_line(r) -> str:
    s, group = r.stats, role_group(r.position)
    if group == "GK":
        return f"{s.saves} saves · {s.goals_conceded} conceded"
    if group == "DEF":
        return (f"duels {s.tackles_won}/{s.tackles_won + s.tackles_lost} · aerials {s.aerial_duels_won}/"
                f"{s.aerial_duels_won + s.aerial_duels_lost} · {s.pass_accuracy_pct:.0f}% passing")
    if group == "MID":
        return (f"{s.pass_accuracy_pct:.0f}% passing ({s.passes_attempted}) · {s.key_passes} chances · "
                f"duels {s.tackles_won}/{s.tackles_won + s.tackles_lost}")
    return (f"{s.goals} goals · {s.xg:.2f} xG · {s.shots} shots · {s.key_passes} chances · "
            f"dribbles {s.dribbles_completed}/{s.dribbles_attempted}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚽ Football Team Agent")
    st.caption("Build a team from plain-English requirements, test changes, play matches and get honest, "
               "explained feedback. An AI assistant investigates and proposes; code verifies every claim and "
               "you approve every change.")
    # the FIFA 22 pool is only offered where it has been downloaded (it's never deployed)
    sources = list(SOURCES) if data_loader.KAGGLE_PATH.exists() else ["synthetic"]
    source = st.selectbox("Player pool", sources,
                          index=sources.index(player_source()) if player_source() in sources else 0, key="source",
                          format_func={"synthetic": "Demo pool · 100 players", "kaggle": "FIFA 22 · 19,239 real",
                                       "both": "Both pools"}.get)
    try:
        ctx = get_context(source, _pool_stamp())
    except FileNotFoundError as e:
        st.error(str(e))
        if source != "kaggle" and st.button("Generate the demo pool", type="primary"):
            generate_pool()
            st.rerun()
        st.stop()
    st.session_state["_ctx"] = ctx
    teams = storage.list_teams()
    matches = pipeline.list_matches()

    options = llm_catalog.catalog()
    by_id = {m["id"]: m for m in options}
    ids = list(by_id)
    default_model = llm_model() if llm_model() in by_id else llm_catalog.DEFAULT_MODEL
    st.selectbox("AI model", ids, index=ids.index(default_model), key="model",
                 format_func=lambda i: f"{by_id[i]['label']} · ${by_id[i]['in']:.2f} / ${by_id[i]['out']:.2f}",
                 help="Price per million input / output tokens (live from OpenRouter). The AI reads requests and "
                      "runs the assistant; ratings, fit, chemistry and simulations are always plain code.")
    st.caption(by_id[model_id()]["note"])
    tracker = make_tracker()
    status = "API key set" if llm_enabled() else "No API key — offline mode"
    st.caption(f"{status} · spent \\${tracker.total_usd:.4f} of \\${tracker.budget_usd:.2f}")
    with st.popover("What does each AI action cost?", icon=":material/payments:"):
        st.dataframe(pd.DataFrame([{"Model": m["label"], **{act: llm_catalog.estimate(m["id"], *tok)
                                                            for act, tok in llm_catalog.TYPICAL_TOKENS.items()}}
                                   for m in options]),
                     hide_index=True, column_config={a: st.column_config.NumberColumn(format="$%.4f")
                                                     for a in llm_catalog.TYPICAL_TOKENS})

    st.divider()
    st.caption(f"{len(ctx.pool):,} players · {len(teams)} team{'s' * (len(teams) != 1)} · "
               f"{len(matches)} match{'es' * (len(matches) != 1)}")
    with st.expander("Data"):
        if st.button("Regenerate the demo pool", width="stretch"):
            generate_pool()
            st.rerun()
        with st.popover("Clear all teams and matches", width="stretch"):
            st.markdown("Delete **every team, match, analysis and scouting report**? The player pool stays.")
            if st.button("Yes, clear everything", type="primary", key="clear_all"):
                pipeline.clear_all()
                st.session_state.clear()
                st.rerun()

flash = st.session_state.pop("flash", None)
if flash:
    st.toast(flash)
new_match = st.session_state.pop("new_match", None)
team_ids = list(teams)

tab_squad, tab_matches, tab_players = st.tabs(["Squad", "Matches", "Players"])


# ---------------------------------------------------------------------------
# The assistant: proposals, analyses and reviews, always with the evidence
# ---------------------------------------------------------------------------

ASSIST_EXAMPLES = ["we concede too many headers: stronger in the air at the back, under 28, without losing passing",
                   "our midfield fades after the hour: more stamina, but keep the ball-winning",
                   "more pace and 1v1 threat on both wings",
                   "a better keeper with sharper reflexes, same nationality as at least one centre-back",
                   "we create chances but don't score: upgrade the finishing up front, under 27",
                   "a younger spine: centre-backs, holding midfielder and striker under 30, without getting worse"]


def source_label(out: dict) -> str:
    return "rules (offline)" if out["source"] == "offline" else out["source"].removeprefix("llm:")


def assistant_status(out: dict) -> None:
    """Who produced it, and what the code could or couldn't verify."""
    bits = [f"By {source_label(out)}"]
    if out.get("confidence"):
        bits.append(f"confidence: {out['confidence']}")
    if out.get("generated_at"):
        bits.append(pd.Timestamp(out["generated_at"]).tz_convert(None).strftime("%d %b %H:%M"))
    st.caption(" · ".join(bits))
    if out["source"] == "offline":
        st.caption(":orange[AI not used" + (f" — {out['error']}" if out.get("error") else " — no API key")
                   + ". This is the rules-based version: numbers only, no reasoning.]")
    elif out.get("error"):
        st.caption(f":orange[Stopped early — {out['error']}.]")
    if out.get("problems"):
        with st.popover(f"{len(out['problems'])} statement(s) not verified", icon=":material/warning:"):
            st.caption("The code checked the answer and sent these back; the model didn't fix them in 2 rounds. "
                       "Treat them with care.")
            for p in out["problems"]:
                st.markdown(f"- {p}")
    elif out["source"] != "offline":
        st.caption(":green[✓ Verified by code: every player, slot and requirement exists, and every number "
                   "matches the data the tools returned.]")


def trace_popover(out: dict) -> None:
    trace = out.get("trace") or []
    if not trace:
        return
    with st.popover("How it got there", icon=":material/route:"):
        st.caption("Every tool call in order, with what came back. Tools only read data and run test "
                   "simulations — nothing is saved until you approve.")
        n = 0
        for t in trace:
            if "thought" in t:
                st.markdown(f"> {t['thought']}")
                continue
            n += 1
            args = json.dumps(t["args"], ensure_ascii=False)
            st.markdown(f"`{n}` **{t['tool']}** `{args[:140] + ('…' if len(args) > 140 else '')}`  \n"
                        f"→ {t['result']}")


def note_text(p: dict) -> str:
    """A player note without the name repeated in front ("Sven Novak: 5.3 ..." -> "5.3 ...")."""
    name = player_name(ctx, p["player_id"])
    return str(p.get("note", "")).removeprefix(f"{name}: ").removeprefix(f"{name} — ").removeprefix(f"{name} - ")


def measured_text(m: dict | None) -> str:
    """The code-measured impact of a change set (never the model's numbers)."""
    if not m:
        return ""
    if m.get("error"):
        return f":orange[{m['error']}]"
    text = (f"average fit {m['avg_fit_before']:.2f} → **{m['avg_fit_after']:.2f}** · chemistry "
            f"{m['chemistry_before']:.1f} → **{m['chemistry_after']:.1f}**")
    if m.get("points_delta") is not None:
        icon = {"better": ":green[▲ better]", "worse": ":orange[▼ worse]"}.get(m["verdict"], "no clear difference")
        text += (f" · **{m['points_delta']:+.2f}** points per match (± {m['margin']:.2f}) vs {m['opponent']} "
                 f"→ {icon}")
    return text


def stage_many(team_id: str, changes: dict[str, str]) -> str | None:
    """Add changes to the team's draft (on top of pending ones). Returns an error, or None."""
    saved = storage.load_team(team_id)
    d = drafts().get(team_id)
    pending = d["changes"] if d and d["base_version"] == saved.version else {}
    merged = {**pending, **changes}
    try:
        pipeline.preview_swaps(ctx, saved, merged)
    except ValueError as e:
        return str(e)
    drafts()[team_id] = {"base_version": saved.version, "changes": {}}
    for slot, pid in merged.items():
        stage_change(saved, slot, pid)
    return None


def recommendation_rows(recs: list[dict], key: str) -> None:
    """Suggested fixes; a tested replacement can be staged with one click."""
    for i, r in enumerate(recs):
        team = team_name(r["team_id"])
        who = f" → **{player_name(ctx, r['player_id'])}**" if r.get("player_id") else ""
        where = f"{team} {r['slot']}" if r.get("slot") else team
        c1, c2 = st.columns([6, 1.2], vertical_alignment="center")
        c1.markdown(f"**{where}**{who} — {r['reason']}" + (f"  \n{measured_text(r['measured'])}"
                                                            if r.get("measured") else ""))
        if r.get("player_id") and c2.button("Stage", key=f"{key}_stage_{i}", icon=":material/add_circle:",
                                            help="Add this change to the team's pending changes (Squad)"):
            error = stage_many(r["team_id"], {r["slot"]: r["player_id"]})
            if error:
                st.error(error)
            else:
                st.session_state["select_team"] = r["team_id"]
                st.session_state["flash"] = (f"Staged {player_name(ctx, r['player_id'])} for {r['slot']} — "
                                             f"review and save it in Squad")
                st.rerun()


def assistant_panel(team_id: str) -> None:
    st.caption("Say what you want. The assistant investigates the team, its match history and the pool with "
               "tools, tests its idea over simulated matches and proposes up to 3 changes. Code checks every "
               "claim and measures the impact itself; nothing changes until you approve.")
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")

    def use_example() -> None:
        if st.session_state.get("assist_example"):
            st.session_state["assist_goal"] = st.session_state["assist_example"]

    goal = c1.text_input("Goal", key="assist_goal", placeholder="e.g. we lose too many headers at the back")
    st.pills("Examples", ASSIST_EXAMPLES, key="assist_example", on_change=use_example)
    latest = storage.list_teams()[team_id]["latest_version"]
    if c2.button("Ask", type="primary", icon=":material/auto_awesome:", width="stretch", key="assist_ask"):
        if not goal.strip():
            st.warning("Type a goal first, or pick an example.")
            return
        with st.spinner("Investigating — reading the team, searching the pool, testing changes..."):
            out = agent.recommend_changes(ctx, team_id, goal, model=model_id(), tracker=make_tracker())
        st.session_state["assist"] = {"team_id": team_id, "version": latest, "out": out,
                                      "n": st.session_state.get("assist", {}).get("n", 0) + 1}
    state = st.session_state.get("assist")
    if not state or state["team_id"] != team_id:
        return
    if state["version"] != latest:
        st.caption(":orange[The team has a new version since this answer — ask again.]")
        return
    out = state["out"]
    st.divider()
    st.markdown(out["summary"])
    assistant_status(out)
    if out["requirements"]:
        st.caption("Requirements it applied: " + ", ".join(out["requirements"]))
    for caveat in out.get("caveats") or []:
        if not caveat.startswith(("Measured by code", "Offline planner")):  # shown below / in the status
            st.caption(f":material/info: {caveat}")
    saved = storage.load_team(team_id)
    current = {s.slot_id: s.player_id for s in saved.slots}
    per_slot = {c["slot"]: c for c in (out["measured"] or {}).get("changes", [])}
    approved = {}
    for i, c in enumerate(out["changes"]):
        cols = st.columns([0.35, 0.35, 8], vertical_alignment="center")
        if cols[0].checkbox("Approve", value=True, key=f"assist_ok_{state['n']}_{i}", label_visibility="collapsed",
                            help="Include this change"):
            approved[c["slot"]] = c["player_id"]
        if cols[1].button(":material/info:", key=f"assist_i_{state['n']}_{i}", type="tertiary", help=INFO_HELP):
            player_dialog(c["player_id"], compare_with=current[c["slot"]],
                          position=pipeline.slot_position(saved, c["slot"]))
        m = per_slot.get(c["slot"], {})
        fit = f" · fit {m['fit_before']:.1f} → **{m['fit_after']:.1f}**" if m.get("fit_before") is not None else ""
        cols[2].markdown(f"**{c['slot']}** {player_name(ctx, current[c['slot']])} → "
                         f"**{player_name(ctx, c['player_id'])}**{fit}  \n{c.get('reason', '')}")
    if not out["changes"]:
        st.caption("No change proposed.")
    for d in out["dropped"]:
        st.caption(f":orange[Dropped by the checker — {d}]")
    if out["measured"]:
        st.markdown("**All proposed changes together, measured by code:** " + measured_text(out["measured"]))
    b1, b2 = st.columns([1.4, 3], vertical_alignment="center")
    if out["changes"] and b1.button(f"Stage {len(approved)} approved", type="primary", icon=":material/check:",
                                    disabled=not approved, key="assist_stage", width="stretch"):
        error = stage_many(team_id, approved)
        if error:
            st.error(error)
        else:
            st.session_state.pop("assist", None)
            st.session_state["flash"] = f"Staged {len(approved)} change(s) — test or save them below"
            st.rerun()
    with b2:
        trace_popover(out)


def history_panel(team_id: str) -> None:
    h = pipeline.team_history(ctx, team_id)
    if not h["matches"]:
        st.caption("No matches yet. Once this team plays, its results, every player's form over time and the "
                   "assistant's review appear here.")
        return
    r = h["record"]
    m = st.columns(4)
    m[0].metric("Played", len(h["matches"]))
    m[1].metric("Record", f"{r['W']}W {r['D']}D {r['L']}L")
    m[2].metric("xG for", f"{sum(x['xg_for'] for x in h['matches']):.2f}")
    m[3].metric("xG against", f"{sum(x['xg_against'] for x in h['matches']):.2f}")
    c1, c2 = st.columns([2, 3], gap="large")
    c1.dataframe(pd.DataFrame([{"Match": x["match_id"], "v": f"v{x['version']}", "Opponent": x["opponent"],
                                "Result": f"{x['result']} {x['score']}", "xG": f"{x['xg_for']:.2f}–{x['xg_against']:.2f}"}
                               for x in reversed(h["matches"])]), hide_index=True)
    c2.dataframe(pd.DataFrame([{"Player": p["name"], "Apps": p["apps"], "Avg": p["avg_rating"],
                                "Last": p["last_rating"], "Trend": p["trend"],
                                "Keeps showing": "; ".join(p["recurring_weaknesses"])} for p in h["players"]]),
                 hide_index=True, column_config={
                     "Avg": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.2f"),
                     "Last": st.column_config.NumberColumn(format="%.1f"),
                     "Trend": st.column_config.NumberColumn(format="%+.2f", help="Last rating minus the average")})
    st.divider()
    review = pipeline.load_team_review(team_id)
    total = len(h["matches"])
    st.markdown("**Assistant's review of the team's form**")
    c1, c2, c3 = st.columns([2, 1, 1.4], vertical_alignment="bottom")
    pick = c1.segmented_control("Games to review", ["Last", "Random", "All"], default="Last",
                                key=f"review_pick_{team_id}",
                                help="Last = current form (usually best). Random = a spread across the whole "
                                     "history. All = everything (costs more on a long history).") or "Last"
    n = c2.number_input("How many", 1, max(1, total), min(10, total), key=f"review_n_{team_id}",
                        disabled=pick == "All")
    limit = None if pick == "All" else int(n)
    if c3.button("Refresh review" if review else "Review form", icon=":material/auto_awesome:", key="review_btn",
                 width="stretch", help="Reads the selected games with tools and tests any fix"):
        with st.spinner("Reviewing..."):
            pipeline.generate_team_review(ctx, team_id, model=model_id(), tracker=make_tracker(), limit=limit,
                                          pick=pick.lower())
        st.rerun()
    games = min(limit or total, total)
    tokens = 15000 + 2500 * min(games, 6)   # the brief grows with the games; match look-ups are the big part
    st.caption(f"{games} of {total} game(s) · roughly \\${llm_catalog.estimate(model_id(), tokens, 900):.3f} "
               f"with {llm_catalog.describe(model_id()).split(' · ')[0]}")
    if not review:
        return
    if review["matches_seen"] != total:
        st.caption(f":orange[Written when there were {review['matches_seen']} game(s); there are {total} now — "
                   "refresh.]")
    scope = review.get("scope")
    if scope:
        st.caption(f"Based on {agent.scope_label(scope)}"
                   + (f": {', '.join(scope['match_ids'])}" if scope["pick"] == "random" else "") + ".")
    st.markdown(review["summary"])
    assistant_status(review)
    for t in review["trends"]:
        st.markdown(f"- {t}")
    for p in review["players"]:
        st.markdown(f":material/person: **{player_name(ctx, p['player_id'])}** — {note_text(p)}")
    if review["recommendations"]:
        st.markdown("**Suggested fixes**")
        recommendation_rows(review["recommendations"], key=f"review_{team_id}")
    trace_popover(review)


def analysis_panel(log: EventLog) -> None:
    analysis = pipeline.load_match_analysis(log.match_id)
    c1, c2 = st.columns([4, 1.3], vertical_alignment="center")
    c1.caption("The assistant reads the match, both sides' history and the players involved with tools, and "
               "tests every replacement it suggests. Code checks every claim.")
    if c2.button("Re-analyse" if analysis else "Analyse", icon=":material/auto_awesome:", width="stretch",
                 key=f"analyse_{log.match_id}"):
        with st.spinner("Analysing the match..."):
            pipeline.generate_match_analysis(ctx, log.match_id, model=model_id(), tracker=make_tracker())
        st.rerun()
    if not analysis:
        return
    st.markdown(f"#### {analysis['headline']}")
    st.markdown(analysis["summary"])
    assistant_status(analysis)
    cols = st.columns(2, gap="large")
    for col, tid in zip(cols, (log.team_a, log.team_b)):
        with col:
            st.markdown(f"**{log.name_v(tid)}**")
            lines = [f"- {f['point']}" + (f" — _{f['evidence']}_" if f.get("evidence") else "")
                     for f in analysis["key_factors"] if f["team_id"] == tid]
            lines += [f"- :material/person: **{player_name(ctx, p['player_id'])}** — {note_text(p)}"
                      for p in analysis["players"] if p["team_id"] == tid]
            st.markdown("\n".join(lines) or "Nothing noted.")
    if analysis["recommendations"]:
        st.markdown("**Suggested fixes**")
        recommendation_rows(analysis["recommendations"], key=f"analysis_{log.match_id}")
    trace_popover(analysis)


# ---------------------------------------------------------------------------
# Squad
# ---------------------------------------------------------------------------

def new_team_form() -> None:
    c1, c2, c3, c4 = st.columns(4)
    suggested = next(t for t in ("t_A", "t_B", "t_C", "t_D", "t_E") if t not in teams)
    new_id = c1.text_input("Team id", value=suggested)
    new_name = c2.text_input("Name", value={"t_A": "Alpha FC", "t_B": "Beta FC"}.get(suggested, "New FC"))
    formation = c3.selectbox("Formation", list(ctx.formations), key="build_formation")
    exclude = c4.selectbox("Leave out players of", [None] + team_ids, key=f"build_exclude_{len(team_ids)}",
                           index=len(team_ids) if team_ids else 0,
                           format_func=lambda t: "— nobody —" if t is None else team_name(t))
    with st.expander(f"Position weights · {formation} defaults", icon=":material/tune:"):
        st.caption("What each position values. Start from the formation's role defaults and type your own.")
        build_weights = position_weight_editor(ctx, formation, pipeline.formation_defaults(ctx, formation),
                                               key=f"bw_{formation}")
    if st.button("Build team", type="primary", icon=":material/groups:"):
        if not new_id.strip() or not new_name.strip():
            st.error("Team id and name are required.")
            return
        try:
            team, excluded = pipeline.build_team(ctx, new_id.strip(), new_name.strip(), formation, exclude,
                                                 weights=build_weights)
        except ValueError as e:
            st.error(str(e))
            return
        drafts().pop(team.team_id, None)
        st.session_state["select_team"] = team.team_id  # applied before the picker is drawn next run
        st.session_state["flash"] = f"Built {team.name}" + (f" (left out {excluded} players)" if excluded else "")
        st.rerun()


def squad_body(view_id: str, version: int, latest: int, is_latest: bool) -> None:
    """Everything under the team picker in the Squad tab."""
    saved, draft, changes, _ = draft_state(ctx, view_id)
    team = draft if is_latest else pipeline.rescored(ctx, storage.load_team(view_id, version))
    if not is_latest:
        st.caption(f"Viewing v{version} (read-only). Changes always start from the latest version.")

    left, right = st.columns([5, 6], gap="large")
    with left:
        history = pipeline.diff_changes(storage.load_team(view_id, version).diff)
        pitch(ctx, team, highlight=set(changes) if is_latest and changes else {c["slot"] for c in history})
    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Average fit", f"{team.avg_fit_score:.2f}",
                  f"{team.avg_fit_score - saved.avg_fit_score:+.2f}" if is_latest and changes else None)
        m2.metric("Chemistry", f"{team.chemistry_score:.1f}",
                  f"{team.chemistry_score - saved.chemistry_score:+.1f}" if is_latest and changes else None)
        m3.metric("Formation", team.formation)
        e1, e2, e3 = st.columns(3)
        with e1:
            fit_explainer(ctx, team, key="squad_fit")
        with e2:
            chemistry_explainer(ctx, team, key="squad_chem")
        with e3.popover("Manage", icon=":material/more_horiz:"):
            versions = storage.list_versions(view_id)
            if len(versions) > 1:
                st.markdown(f"**Delete v{version}** only. Matches it played keep their line-ups; "
                            "the number v" f"{version} won't be reused.")
                if st.button(f"Delete v{version}", key="confirm_delete_version"):
                    pipeline.delete_version(view_id, version)
                    drafts().pop(view_id, None)
                    st.session_state["flash"] = f"Deleted {team_name(view_id)} v{version}"
                    st.rerun()
                st.divider()
            played = pipeline.matches_for_team(view_id)
            st.markdown(f"**Delete {team_name(view_id)}** with all {len(versions)} version(s)"
                        + (f" and the {len(played)} match(es) it played." if played else "."))
            if st.button("Delete team", type="primary", key="confirm_delete_team"):
                pipeline.delete_team(view_id)
                drafts().pop(view_id, None)
                st.session_state["flash"] = f"Deleted {view_id}"
                st.rerun()
        if is_latest:
            pending_panel(ctx, view_id)
        templates = pipeline.team_templates(ctx, team)
        lineup = [s for s in team.slots if s.player_id]
        player_rows(
            ctx, [s.player_id for s in lineup],
            [("Slot", 0.6), ("Role", 2), ("Player", 1.9), ("Age · foot", 0.8), ("Fit", 0.5)],
            lambda pid, i: [lineup[i].slot_id, pipeline.role_name(ctx, team.formation, lineup[i].position),
                            f"**{ctx.lookup[pid].name}**" + (" ●" if lineup[i].slot_id in changes else ""),
                            f"{ctx.lookup[pid].age} · {ctx.lookup[pid].preferred_foot[0].upper()}",
                            f"**{fit_score(ctx.lookup[pid], templates[lineup[i].position]):.1f}**"],
            key="lineup", info_kwargs={})

    if is_latest:
        with st.expander("Assistant", icon=":material/auto_awesome:",
                         expanded=st.session_state.get("assist", {}).get("team_id") == view_id):
            assistant_panel(view_id)

        with st.expander("Change players", icon=":material/swap_horiz:",
                         expanded="sl_swap" in st.session_state or bool(changes)):
            st.caption("Describe what a position needs — skills, age, foot, nationality, natural position. "
                       "You'll see how it was read, a ranked shortlist, and can stage several changes before "
                       "saving them as one version.")
            c1, c2, c3 = st.columns([1.2, 3, 1], vertical_alignment="bottom")
            slot = c1.selectbox(
                "Position", [s.slot_id for s in draft.slots], key=f"swap_slot_{view_id}",
                format_func=lambda sid: f"{sid} — {player_name(ctx, next(s.player_id for s in draft.slots if s.slot_id == sid))}")
            brief = c2.text_input("What do you want?", key="swap_brief",
                                  placeholder="e.g. an aerial left-footed centre-back under 28")

            def use_example() -> None:
                if st.session_state.get("swap_example"):
                    st.session_state["swap_brief"] = st.session_state["swap_example"]

            st.pills("Examples", EXAMPLE_REQUESTS[pipeline.slot_position(draft, slot)], key="swap_example",
                     on_change=use_example,
                     help="Click to fill the request, then edit it. Mix qualities, limits (under 25, at least "
                          "80 pace, two-footed) and team-mates: 'him', a slot ('our CB_R'), or a group "
                          "('at least one centre-back').")
            if c3.button("Find players", icon=":material/manage_search:", type="primary", width="stretch"):
                with st.spinner("Reading the request and ranking the pool..."):
                    request, info = pipeline.swap_request_from_brief(ctx, draft, slot, brief, model=model_id(),
                                                                     tracker=make_tracker())
                    request_shortlist(ctx, view_id, request, "sl_swap", info)
            if st.session_state.get("sl_swap", {}).get("team_id") == view_id:
                st.divider()
                candidate_picker(ctx, "sl_swap")

        with st.expander("Test changes over many matches", icon=":material/science:"):
            test_panel(ctx, view_id)

        with st.expander("Position weights", icon=":material/tune:"):
            st.caption("What each position values in this team. Saving keeps the players and re-scores them; "
                       "rebuilding re-picks the XI with these weights.")
            tuned = position_weight_editor(ctx, team.formation, pipeline.team_weights(ctx, saved),
                                           key=f"tw_{view_id}_v{latest}")
            b1, b2, _ = st.columns([1, 1, 2])
            next_v = storage.next_version_number(view_id)
            if b1.button(f"Save as v{next_v}", key="retune", type="primary", width="stretch"):
                try:
                    pipeline.retune_team(ctx, view_id, tuned)
                except ValueError as e:
                    st.error(str(e))
                else:
                    st.session_state["flash"] = f"Saved new weights as v{next_v}"
                    st.rerun()
            if b2.button("Rebuild the XI", key="rebuild", width="stretch"):
                pipeline.build_team(ctx, view_id, saved.name, saved.formation, weights=tuned)
                drafts().pop(view_id, None)
                st.session_state["flash"] = f"Rebuilt {saved.name} as v{next_v}"
                st.rerun()

    with st.expander("Form & history", icon=":material/monitoring:"):
        history_panel(view_id)

    with st.expander("Version history", icon=":material/history:"):
        rows = []
        for v in storage.list_versions(view_id):
            t = storage.load_team(view_id, v)
            ch = pipeline.diff_changes(t.diff)
            rows.append({"Version": f"v{v}", "Change": (t.diff or {}).get("reason", "built"),
                         "Players changed": ", ".join(f"{c['slot']}: {player_name(ctx, c['old_player_id'])} → "
                                                      f"{player_name(ctx, c['new_player_id'])}" for c in ch),
                         "Average fit": t.avg_fit_score})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"Average fit": st.column_config.NumberColumn(format="%.2f")})


with tab_squad:
    if not team_ids:
        st.subheader("Build your first team")
        st.caption("The builder picks the best-fitting player for every position from the pool. Build two teams "
                   "(the second leaving out the first team's players) to play matches.")
        new_team_form()
    else:
        c1, c2, c3 = st.columns([3, 1.3, 1.2], vertical_alignment="bottom")
        ensure_choice("squad_team", team_ids, st.session_state.pop("select_team", None))
        view_id = c1.selectbox("Team", team_ids, format_func=team_label, key="squad_team")
        latest = teams[view_id]["latest_version"]
        version = c2.selectbox("Version", storage.list_versions(view_id), key=f"squad_version_{view_id}_{latest}",
                               format_func=lambda v: f"v{v}" + (" · latest" if v == latest else ""))
        with c3.popover("New team", icon=":material/add:", width="stretch"):
            new_team_form()
        is_latest = version == latest
        missing = pipeline.missing_players(ctx, storage.load_team(view_id, version))
        if missing:
            st.warning(f"{team_name(view_id)} was built from a different player pool ({len(missing)} of its "
                       f"players aren't in the one selected). Switch **Player pool** in the sidebar to see it.",
                       icon=":material/swap_horiz:")
        else:
            squad_body(view_id, version, latest, is_latest)


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

with tab_matches:
    if len(team_ids) < 1:
        st.info("Build a team in **Squad** first.")
    else:
        with st.container(border=True):
            st.markdown("**Play a match** · 90 minutes between any two saved team versions — including a team "
                        "against an older version of itself")
            c = st.columns([2, 1, 2, 1, 1.2, 1.2], vertical_alignment="bottom")
            home = c[0].selectbox("Home", team_ids, format_func=team_label, key="m_home")
            hv = c[1].selectbox("Version", storage.list_versions(home),
                                key=f"m_hv_{home}_{teams[home]['latest_version']}",
                                format_func=lambda v: f"v{v}")
            away_default = next((t for t in team_ids if t != home), home)
            away = c[2].selectbox("Away", team_ids, index=team_ids.index(away_default), format_func=team_label,
                                  key=f"m_away_{len(team_ids)}")  # re-defaults when a team is added
            av = c[3].selectbox("Version", storage.list_versions(away),
                                key=f"m_av_{away}_{teams[away]['latest_version']}",
                                format_func=lambda v: f"v{v}")
            seed = c[4].number_input("Seed", value=None, min_value=1, step=1, placeholder="random",
                                     key="match_seed", help="Leave empty for a new match; enter an earlier "
                                                            "match's seed to replay it exactly.")
            if c[5].button("Play", type="primary", icon=":material/sports_soccer:", width="stretch"):
                try:
                    log = pipeline.simulate(ctx, home, away, int(seed) if seed else None, hv, av)
                except ValueError as e:
                    st.error(str(e))
                else:
                    st.session_state["new_match"] = log.match_id
                    st.rerun()

    if matches:
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        ensure_choice("match_view", matches, new_match)
        match_id = c1.selectbox("Match", matches, key="match_view", format_func=match_label)
        with c2.popover("Delete", icon=":material/delete:", width="stretch"):
            if st.button("Delete this match", type="primary", key="confirm_delete_match"):
                pipeline.delete_match(match_id)
                st.rerun()
        log = pipeline.load_event_log(match_id)
        reports = pipeline.load_report(match_id) or pipeline.build_report(match_id)
        scoreboard(ctx, log)
        st.caption(f"{log.label} · seed {log.seed}")

        left, right = st.columns([5, 4], gap="large")
        with left:
            stat_bars(log, team_stats(log, reports))
        with right:
            motm = max(reports.values(), key=lambda r: r.rating, default=None)
            if motm:
                st.markdown(f":material/star: **Player of the match** · {player_name(ctx, motm.player_id)} "
                            f"({log.name(motm.team_id)}, {motm.slot_id}) · **{motm.rating:.1f}**")
                st.caption(" · ".join(motm.strengths[:2]))
            st.markdown("**Key moments**")
            moments = key_moments(ctx, log)
            st.markdown("  \n".join(moments) if moments else "No goals or big chances.")

        with st.expander("Coach's debrief", icon=":material/record_voice_over:", expanded=True):
            record = pipeline.match_debrief(ctx, match_id)
            st.caption("Computed from the match numbers: what worked, what didn't and what to change — each line "
                       "measured against the opponent.")
            cols = st.columns(2, gap="large")
            for col, tid in zip(cols, (log.team_a, log.team_b)):
                d = record[tid]
                with col:
                    st.markdown(f"**{log.name_v(tid)}** — {d['summary']}")
                    st.markdown("  \n".join([f":green[✓] {x}" for x in d["worked"]]
                                            + [f":orange[✗] {x}" for x in d["didnt"]]))
                    st.markdown("**Changes to consider**  \n" + "  \n".join(f"→ {x}" for x in d["changes"]))

        with st.expander("Assistant's analysis", icon=":material/auto_awesome:",
                         expanded=pipeline.load_match_analysis(match_id) is not None):
            analysis_panel(log)

        with st.expander("Player ratings & feedback", icon=":material/groups:"):
            st.caption("Each player is rated and judged on what his role is for: defenders on duels, aerials and "
                       "clean sheets; midfielders on passing, chances and ball-winning; attackers on goals, xG and "
                       "chances; keepers on saves.")
            fb = st.session_state.get("sl_feedback")
            if fb:
                st.markdown(f"**Replacement search** · {team_name(fb['team_id'])} "
                            f"{fb['shortlist'].target_slot} — {fb['request'].rationale}")
                candidate_picker(ctx, "sl_feedback")
                st.divider()
            side = st.segmented_control("Team", [log.team_a, log.team_b], default=log.team_a,
                                        format_func=log.name_v, key=f"ratings_side_{match_id}") or log.team_a
            base_id = pipeline.base_team_id(side)
            team_now = storage.load_team(base_id) if base_id in teams else None
            players = sorted((r for r in reports.values() if r.team_id == side), key=lambda r: -r.rating)
            cols = st.columns(2)
            for i, r in enumerate(players):
                with cols[i % 2], st.container(border=True):
                    bg, fg = rating_style(r.rating)
                    head, info_btn = st.columns([12, 1], vertical_alignment="center")
                    head.html(f'<div style="display:flex;justify-content:space-between;align-items:center">'
                              f'<div><b>{html.escape(player_name(ctx, r.player_id))}</b> '
                              f'<span style="opacity:.7">{r.slot_id} · {ROLE_LABEL[role_group(r.position)]}</span>'
                              f'</div><span style="background:{bg};color:{fg};padding:2px 10px;border-radius:8px;'
                              f'font-weight:800">{r.rating:.1f}</span></div>')
                    if info_btn.button(":material/info:", key=f"prof_{match_id}_{side}_{r.player_id}", type="tertiary",
                                       help=INFO_HELP):
                        player_dialog(r.player_id)
                    st.caption(stat_line(r))
                    st.markdown("  \n".join([f":green[✓] {x}" for x in r.strengths]
                                            + [f":orange[✗] {x}" for x in r.weaknesses]))
                    b1, b2 = st.columns(2)
                    with b1.popover("Rating", icon=":material/calculate:"):
                        opp = log.team_b if side == log.team_a else log.team_a
                        total, terms = rating_breakdown(r.stats, r.position, log.final_score[opp])
                        st.dataframe(pd.DataFrame([{"Action": k.replace("_", " "), "Count": v, "Points": p}
                                                   for k, v, p in terms]), hide_index=True,
                                     column_config={"Points": st.column_config.NumberColumn(format="%+.2f")})
                        st.caption(f"{RATING_BASE} + {sum(p for *_, p in terms):+.2f} = {total:.1f} (kept 0–10)")
                    still_there = team_now and any(sl.slot_id == r.slot_id and sl.player_id == r.player_id
                                                   for sl in team_now.slots)
                    if still_there and b2.button("Find replacement", key=f"fb_{match_id}_{side}_{r.player_id}",
                                                 type="tertiary", icon=":material/swap_horiz:"):
                        request, _, message = pipeline.feedback(ctx, base_id, match_id, r.slot_id, side)
                        if request is None:
                            st.session_state["flash"] = message
                            st.session_state.pop("sl_feedback", None)
                        else:
                            request_shortlist(ctx, base_id, request, "sl_feedback")
                        st.rerun()

        with st.expander("How ratings work", icon=":material/calculate:"):
            st.caption(f"Everyone starts at {RATING_BASE}; each action adds or removes points by role; the total "
                       "is kept between 0 and 10. Goals conceded, team conceded and clean sheet all count only "
                       "the goals scored against the player's own side.")
            actions = sorted({a for pts in RATING_POINTS.values() for a in pts})
            st.dataframe(pd.DataFrame([{"Action": a.replace("_", " "),
                                        **{ROLE_LABEL[g]: RATING_POINTS[g].get(a) for g in RATING_POINTS}}
                                       for a in actions]), hide_index=True,
                         column_config={ROLE_LABEL[g]: st.column_config.NumberColumn(format="%+.2f")
                                        for g in RATING_POINTS})

        with st.expander("Match details", icon=":material/query_stats:"):
            t_race, t_lineups, t_log = st.tabs(["xG race", "Line-ups", "Event log"])
            with t_race:
                st.altair_chart(xg_chart(ctx, log), width="stretch")
            with t_lineups:
                l1, l2 = st.columns(2)
                for col, tid in ((l1, log.team_a), (l2, log.team_b)):
                    with col:
                        st.markdown(f"**{log.name_v(tid)}** · {log.formations.get(tid, '')}")
                        ratings = {pid: r.rating for pid, r in pipeline.team_reports(reports, tid).items()}
                        pitch(ctx, lineup_team(log, tid), ratings=ratings)
            with t_log:
                st.dataframe(event_log_table(ctx, log), hide_index=True)
    elif team_ids:
        st.caption("No matches yet — play one above.")


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

with tab_players:
    browse, finder = st.tabs(["Browse", "Find the best fit"])
    with browse:
        c1, c2, c3, c4 = st.columns([2, 2.4, 1, 1], vertical_alignment="bottom")
        search = c1.text_input("Search by name", key="pool_search", placeholder="e.g. Silva")
        positions = sorted({pos for p in ctx.pool for pos in p.natural_positions})
        picked = c2.pills("Positions", positions, selection_mode="multi", key="pool_positions")
        pfoot = c3.segmented_control("Foot", ["left", "right"], key="pool_foot")
        with c4.popover("More filters", icon=":material/filter_alt:", width="stretch"):
            ages = st.slider("Age", 15, 45, (15, 45), key="pool_age")
            nats = st.multiselect("Nationality", sorted({p.nationality for p in ctx.pool}), key="pool_nat")
        shown = [p for p in ctx.pool
                 if (not picked or set(picked) & set(p.natural_positions))
                 and (not pfoot or p.preferred_foot == pfoot)
                 and (not search or search.lower() in p.name.lower())
                 and ages[0] <= p.age <= ages[1] and (not nats or p.nationality in nats)]
        best = {p.player_id: max((fit_score(p, ctx.templates[pos]) for pos in p.natural_positions
                                  if pos in ctx.templates), default=0) for p in shown}
        shown.sort(key=lambda p: -best[p.player_id])
        page_size = 20
        pages = max(1, -(-len(shown) // page_size))
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        c1.caption(f"{len(shown):,} players · best fit at their natural position first")
        page = c2.number_input(f"Page (of {pages})", 1, pages, 1, key="pool_page") if pages > 1 else 1
        chunk = shown[(page - 1) * page_size: page * page_size]
        player_rows(ctx, [p.player_id for p in chunk],
                    [("Player", 2.4), ("Positions", 1.2), ("Age · foot", 0.9), ("Nationality", 0.9),
                     ("Best fit", 0.7)],
                    lambda pid, i: [f"**{ctx.lookup[pid].name}**", "/".join(ctx.lookup[pid].natural_positions),
                                    f"{ctx.lookup[pid].age} · {ctx.lookup[pid].preferred_foot[0].upper()}",
                                    ctx.lookup[pid].nationality, f"{best[pid]:.1f}"],
                    key="browse", info_kwargs={})

    with finder:
        st.caption("Type your own weights and requirements — every player in the pool is ranked by fit. Players "
                   "who miss a requirement are listed after those who meet them.")
        c1, c2 = st.columns([1, 3], gap="large")
        with c1:
            keepers = st.toggle("Goalkeepers", key="find_gk")
            start = st.selectbox("Start from", [p for p in ctx.templates if (p == "GK") == keepers],
                                 key="find_start", help="Prefills the weights with that position's defaults.")
            age_rng = st.slider("Age", 15, 45, (15, 45), key="find_age")
            foot = st.segmented_control("Foot", ["Any", "Left", "Right"], default="Any", key="find_foot") or "Any"
            natural = st.toggle(f"Natural {start} only", key="find_natural")
        with c2:
            weights = weight_inputs(f"find_{start}", ctx.templates[start].attribute_weights,
                                    GK_ATTRS if keepers else OUTFIELD_ATTRS, ctx.templates[start].attribute_weights)
        filters = []
        if age_rng != (15, 45):
            filters.append(HardFilter(attribute="age", min=age_rng[0] if age_rng[0] > 15 else None,
                                      max=age_rng[1] if age_rng[1] < 45 else None))
        if foot != "Any":
            filters.append(HardFilter(attribute="preferred_foot", value=foot.lower()))
        if natural:
            filters.append(HardFilter(attribute="natural_position", value=start))
        if weights:
            top = pipeline.best_fits(ctx, weights, goalkeepers=keepers, filters=filters, top_n=15)
            player_rows(ctx, [r["player_id"] for r in top],
                        [("#", 0.3), ("Player", 2), ("Positions", 1), ("Age · foot", 0.9), ("Fit", 0.6),
                         ("Requirements", 2.4)],
                        lambda pid, i: [f"{i + 1}", f"**{ctx.lookup[pid].name}**",
                                        "/".join(ctx.lookup[pid].natural_positions),
                                        f"{ctx.lookup[pid].age} · {ctx.lookup[pid].preferred_foot[0].upper()}",
                                        f"**{top[i]['fit']:.1f}**",
                                        ":orange[⚠ " + "; ".join(top[i]["issues"]) + "]" if top[i]["issues"]
                                        else (":green[✓ all met]" if filters else "")],
                        key="finder", info_kwargs={"position": start, "weights": pipeline.normalize_weights(
                            {"x": weights})["x"]})
        else:
            st.warning("Give at least one attribute a weight.")
