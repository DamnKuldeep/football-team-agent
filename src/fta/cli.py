"""Command-line front end. Each stage is its own subcommand so a human decides
at every checkpoint (review the team, pick a replacement, act on a report);
`full-demo` runs the whole arc non-interactively (auto-accepting top choices),
printing a banner wherever a human would normally weigh in. `ui` opens the
same pipeline in a browser.

All logic lives in pipeline.py and agent.py -- this module only parses options
and prints results.

Usage examples:
  python -m fta.cli build-team --team-id t_A --name "Alpha FC"
  python -m fta.cli show-team --team-id t_A
  python -m fta.cli swap --team-id t_A --slot CB_L --brief "more aerial, stay left-footed"
  python -m fta.cli apply-swap --team-id t_A --slot CB_L --player-id p_0033 --slot RB --player-id p_0041
  python -m fta.cli simulate --team-a t_A --team-b t_B            # random seed; --seed N replays one
  python -m fta.cli report --match-id M0001 --analyse          # + the assistant's analysis
  python -m fta.cli feedback --team-id t_A --match-id M0001 --slot CB_L
  python -m fta.cli assist --team-id t_A --goal "we lose too many headers at the back"
  python -m fta.cli review --team-id t_A --last 10              # or --random 10, or all games
  python -m fta.cli scout --player-id p_0012 [--team-id t_A]
  python -m fta.cli matches
  python -m fta.cli delete-match --match-id M0001
  python -m fta.cli delete-team --team-id t_A
  python -m fta.cli full-demo
  python -m fta.cli ui
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import click

from . import data_loader, pipeline, storage
from .config import SOURCES, llm_model, load_dotenv, make_tracker

SOURCE_OPTION = click.option("--source", type=click.Choice(SOURCES), default=None,
                             help="Player pool (default: PLAYER_SOURCE from .env, else synthetic)")
MODEL_OPTION = click.option("--model", default=None,
                            help="OpenRouter model id (default: MODEL in .env, else deepseek/deepseek-chat)")


def _banner(text: str) -> None:
    click.secho(f"\n=== {text} ===", fg="cyan", bold=True)


def _human_checkpoint(text: str) -> None:
    click.secho(f"\n[HUMAN CHECKPOINT] {text}", fg="yellow", bold=True)


def _print_team(team, lookup) -> None:
    click.echo(f"{team.name}  ({team.formation})  v{team.version}   "
               f"avg_fit={team.avg_fit_score}  chemistry={team.chemistry_score}")
    for slot in team.slots:
        p = lookup.get(slot.player_id)
        pname = p.name if p else "<empty>"
        foot = p.preferred_foot if p else "-"
        click.echo(f"  {slot.slot_id:6s} ({slot.position:5s}) -> {pname:22s} foot={foot}")


@click.group()
def cli():
    pass


@cli.command("build-team")
@click.option("--team-id", required=True)
@click.option("--name", required=True)
@click.option("--formation", default="4-3-3")
@SOURCE_OPTION
@click.option("--exclude-team-id", default=None, help="Exclude another team's players from the pool (e.g. building Team B after Team A)")
def build_team_cmd(team_id, name, formation, source, exclude_team_id):
    """Build an XI from the player pool."""
    _banner("Build team")
    ctx = pipeline.load_context(source)
    team, excluded = pipeline.build_team(ctx, team_id, name, formation, exclude_team_id)
    if excluded:
        click.echo(f"Excluded {excluded} players already on {exclude_team_id}")
    _print_team(team, ctx.lookup)
    _human_checkpoint(f"Review {team.name} above. Accept as-is, or run `swap` to request a change to any slot.")


@cli.command("show-team")
@click.option("--team-id", required=True)
@click.option("--version", type=int, default=None)
@SOURCE_OPTION
def show_team_cmd(team_id, version, source):
    team = storage.load_team(team_id, version)
    _print_team(team, pipeline.load_context(source).lookup)


@cli.command("swap")
@click.option("--team-id", required=True)
@click.option("--slot", required=True)
@click.option("--brief", default="", help="Natural-language requirement, e.g. 'more aerial, left-footed'")
@SOURCE_OPTION
@MODEL_OPTION
def swap_cmd(team_id, slot, brief, source, model):
    """Read a request for a slot and show the ranked shortlist."""
    _banner(f"Shortlist for {slot}")
    ctx = pipeline.load_context(source)
    tracker = make_tracker()
    team = storage.load_team(team_id)
    request, info = pipeline.swap_request_from_brief(ctx, team, slot, brief, model=model or llm_model(), tracker=tracker)
    shortlist = pipeline.shortlist(ctx, team, request)

    click.echo(f"Interpreted by {info['source']}:")
    for row in pipeline.weight_changes(ctx, team, request):
        mark = "*" if row["requested"] else " "
        click.echo(f"  {mark} {row['attribute']:18s} {row['before']:.2f} -> {row['after']:.2f}  {row['reason']}")
    for item in info["ignored"]:
        click.echo(f"  rejected: {item}")
    current = ctx.lookup.get(shortlist.current_player_id) if shortlist.current_player_id else None
    click.echo(f"Current: {current.name if current else '<empty>'}  fit={shortlist.current_fit_score}")
    for c in shortlist.candidates:
        p = ctx.lookup[c.player_id]
        warn = f"  WARNINGS: {c.warnings}" if c.warnings else ""
        click.echo(f"  {p.name:22s} rank={c.rank_score:6.2f} (fit {c.fit_score:.2f} - penalty {c.penalty:.1f} "
                   f"+ chem {c.chemistry_bonus:+.1f}) Δfit={c.delta_fit:+6.2f}{warn}\n    {c.notes}")
    click.echo(tracker.summary())
    _human_checkpoint(f"Pick a candidate and run `apply-swap --team-id {team_id} --slot {slot} --player-id <id>`, "
                       f"or force any player_id even with warnings.")


@cli.command("apply-swap")
@click.option("--team-id", required=True)
@click.option("--slot", "slots", required=True, multiple=True, help="Repeat with --player-id to change several slots")
@click.option("--player-id", "player_ids", required=True, multiple=True)
@click.option("--reason", default="human-selected from shortlist")
@SOURCE_OPTION
def apply_swap_cmd(team_id, slots, player_ids, reason, source):
    """Save one or more slot changes as ONE new team version."""
    if len(slots) != len(player_ids):
        raise click.UsageError("give one --player-id for every --slot")
    changes = dict(zip(slots, player_ids))
    _banner("Swap: " + ", ".join(f"{s} -> {p}" for s, p in changes.items()))
    ctx = pipeline.load_context(source)
    try:
        new_team, diff = pipeline.apply_swaps(ctx, team_id, changes, reason)
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    _print_team(new_team, ctx.lookup)
    click.echo(f"Diff: {json.dumps(diff, indent=2)}")


@cli.command("simulate")
@click.option("--team-a", required=True)
@click.option("--team-b", required=True)
@click.option("--version-a", type=int, default=None, help="Version of team A (default: latest)")
@click.option("--version-b", type=int, default=None, help="Version of team B (default: latest)")
@click.option("--seed", type=int, default=None, help="Replay a specific match (default: new random seed)")
@SOURCE_OPTION
def simulate_cmd(team_a, team_b, version_a, version_b, seed, source):
    """Play a 90-minute match between any saved team versions."""
    _banner("Match")
    try:
        log = pipeline.simulate(pipeline.load_context(source), team_a, team_b, seed, version_a, version_b)
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Match {log.match_id}  (seed={log.seed})")
    click.echo(f"  {log.label}")
    click.echo(f"Saved -> matches/{log.match_id}/")
    _human_checkpoint(f"Run `report --match-id {log.match_id}` for ratings and the coach's debrief.")


@cli.command("report")
@click.option("--match-id", required=True)
@click.option("--analyse/--no-analyse", default=False,
              help="Also run the assistant's match analysis (tools + verification; offline without a key)")
@MODEL_OPTION
@SOURCE_OPTION
def report_cmd(match_id, analyse, model, source):
    """Ratings and feedback for every player of both sides, and the coach's debrief."""
    _banner("Match report")
    ctx = pipeline.load_context(source)
    names = {pid: p.name for pid, p in ctx.lookup.items()}
    log = pipeline.load_event_log(match_id)
    reports = pipeline.load_report(match_id) or pipeline.build_report(match_id)
    click.echo(log.label)
    debrief = pipeline.match_debrief(ctx, match_id)
    for tid in (log.team_a, log.team_b):
        side = pipeline.team_reports(reports, tid)
        click.secho(f"\n{log.name_v(tid)} — {debrief[tid]['summary']}", bold=True)
        for r in sorted(side.values(), key=lambda r: r.rating, reverse=True):
            click.echo(f"  {r.slot_id or '':5s} {names.get(r.player_id, r.player_id):24s} {r.rating:4.1f}  "
                       f"+ {r.strengths[0]}  - {r.weaknesses[0]}")
        for label, key in (("Worked", "worked"), ("Didn't", "didnt"), ("Change", "changes")):
            for line in debrief[tid][key]:
                click.echo(f"  {label}: {line}")
    if analyse:
        tracker = make_tracker()
        a = pipeline.generate_match_analysis(ctx, match_id, model=model, tracker=tracker)
        _print_assistant(a, ("headline", "summary"))
        for f in a["key_factors"]:
            click.echo(f"  Factor ({log.name_v(f['team_id'])}): {f['point']} {f.get('evidence') or ''}")
        for p in a["players"]:
            click.echo(f"  Player: {names.get(p['player_id'], p['player_id'])} — {p['note']}")
        _print_recommendations(a["recommendations"], ctx)
        click.echo(tracker.summary())
    _human_checkpoint("Run `feedback` to shortlist a replacement for an underperforming slot.")


def _print_assistant(out: dict, keys: tuple[str, ...]) -> None:
    source = "offline rules" if out["source"] == "offline" else out["source"].removeprefix("llm:")
    click.secho(f"\nAssistant ({source})" + (f" — AI unavailable: {out['error']}" if out.get("error") else ""),
                fg="magenta", bold=True)
    for k in keys:
        if out.get(k):
            click.echo(f"  {out[k]}")
    for problem in out.get("problems") or []:
        click.secho(f"  ! unverified: {problem}", fg="yellow")


def _print_measured(m: dict | None, indent: str = "    ") -> None:
    if not m:
        return
    if m.get("error"):
        click.echo(f"{indent}measured: {m['error']}")
        return
    line = (f"{indent}measured: fit {m['avg_fit_before']} -> {m['avg_fit_after']}, chemistry "
            f"{m['chemistry_before']} -> {m['chemistry_after']}")
    if m.get("points_delta") is not None:
        line += f", {m['points_delta']:+.2f} pts/match ± {m['margin']:.2f} vs {m['opponent']} ({m['verdict']})"
    click.echo(line)


def _print_recommendations(recs: list[dict], ctx) -> None:
    for r in recs:
        who = f" -> {ctx.lookup[r['player_id']].name} ({r['player_id']})" if r.get("player_id") else ""
        click.echo(f"  Fix [{r['team_id']}{' ' + r['slot'] if r.get('slot') else ''}]{who}: {r['reason']}")
        _print_measured(r.get("measured"))


@cli.command("assist")
@click.option("--team-id", required=True)
@click.option("--goal", required=True, help="What you want, e.g. 'we lose too many headers at the back'")
@MODEL_OPTION
@SOURCE_OPTION
def assist_cmd(team_id, goal, model, source):
    """The assistant investigates the team and proposes up to 3 verified, measured changes."""
    from . import agent
    _banner(f"Assistant: {goal}")
    ctx = pipeline.load_context(source)
    tracker = make_tracker()
    out = agent.recommend_changes(ctx, team_id, goal, model=model, tracker=tracker)
    _print_assistant(out, ("summary",))
    if out["requirements"]:
        click.echo(f"  Requirements: {', '.join(out['requirements'])}")
    current = {s.slot_id: s.player_id for s in storage.load_team(team_id).slots}
    for c in out["changes"]:
        old = ctx.lookup.get(current[c["slot"]])
        click.echo(f"  {c['slot']:5s} {old.name if old else '<empty>'} -> {ctx.lookup[c['player_id']].name} "
                   f"({c['player_id']}): {c.get('reason', '')}")
    for d in out["dropped"]:
        click.secho(f"  dropped by the checker: {d}", fg="yellow")
    _print_measured(out["measured"], "  ")
    for caveat in out.get("caveats") or []:
        click.echo(f"  Caveat: {caveat}")
    click.echo(tracker.summary())
    if out["changes"]:
        args = " ".join(f"--slot {c['slot']} --player-id {c['player_id']}" for c in out["changes"])
        _human_checkpoint(f"Approve by running `apply-swap --team-id {team_id} {args}` (or pick a subset).")
    else:
        _human_checkpoint("No change proposed.")


@cli.command("review")
@click.option("--team-id", required=True)
@click.option("--last", "last_n", type=int, default=None, help="Review only the last N games (current form)")
@click.option("--random", "random_n", type=int, default=None, help="Review N games picked at random")
@MODEL_OPTION
@SOURCE_OPTION
def review_cmd(team_id, last_n, random_n, model, source):
    """The assistant reviews a team's form: all its games, or the last / a random N."""
    from .agent import scope_label
    if last_n and random_n:
        raise click.UsageError("use --last or --random, not both")
    _banner("Team review")
    ctx = pipeline.load_context(source)
    tracker = make_tracker()
    out = pipeline.generate_team_review(ctx, team_id, model=model, tracker=tracker, limit=last_n or random_n,
                                        pick="random" if random_n else "last")
    click.echo(f"Based on {scope_label(out['scope'])}: {', '.join(out['scope']['match_ids'])}")
    _print_assistant(out, ("summary",))
    for t in out["trends"]:
        click.echo(f"  Trend: {t}")
    for p in out["players"]:
        click.echo(f"  Player: {ctx.lookup[p['player_id']].name} — {p['note']}")
    _print_recommendations(out["recommendations"], ctx)
    click.echo(tracker.summary())


@cli.command("scout")
@click.option("--player-id", required=True)
@click.option("--team-id", default=None, help="Also test him in this team (default: the player only)")
@MODEL_OPTION
@SOURCE_OPTION
def scout_cmd(player_id, team_id, model, source):
    """Criteria scores plus the assistant's scouting report on the player (optionally tested in a team)."""
    ctx = pipeline.load_context(source)
    if player_id not in ctx.lookup:
        raise click.ClickException(f"no player {player_id!r} in the pool")
    _banner(f"Scouting: {ctx.lookup[player_id].name}")
    tracker = make_tracker()
    record = pipeline.generate_scouting(ctx, player_id, model=model, tracker=tracker, team_id=team_id)
    a = record["assessment"]
    click.echo(f"  {a['verdict']} — {a['overall']}/10 as {a['best_role']}  ({a['verdict_rule']})")
    for c in a["criteria"]:
        click.echo(f"    {c['name']:18s} {c['score']:4.1f}  {c['detail']}")
    out = record["analysis"]
    _print_assistant(out, ("summary",))
    for label, key in (("Strength", "strengths"), ("Risk", "risks")):
        for line in out[key]:
            click.echo(f"  {label}: {line}")
    for c in out["compared_with"]:
        click.echo(f"  Compared: {ctx.lookup[c['player_id']].name} — {c['note']}")
    for f in out["fit_for_teams"]:
        click.echo(f"  Fit for {f['team_id']} at {f['slot']}: {f['verdict']} — {f.get('reason', '')}")
        _print_measured(f.get("measured"))
    click.echo(tracker.summary())


@cli.command("feedback")
@click.option("--team-id", required=True)
@click.option("--match-id", required=True)
@click.option("--slot", required=True)
@SOURCE_OPTION
def feedback_cmd(team_id, match_id, slot, source):
    """Shortlist a replacement for a slot based on how its player performed in a match."""
    _banner(f"Performance feedback for {slot}")
    ctx = pipeline.load_context(source)
    request, shortlist, message = pipeline.feedback(ctx, team_id, match_id, slot)
    if request is None:
        click.echo(message)
        return

    click.echo(f"Derived request: {request.model_dump()}")
    for c in shortlist.candidates:
        p = ctx.lookup[c.player_id]
        click.echo(f"  {p.name:22s} fit={c.fit_score:6.2f} Δfit={c.delta_fit:+6.2f}  {c.notes}")
    _human_checkpoint(f"Run `apply-swap --team-id {team_id} --slot {slot} --player-id <id>` to lock in a resim-ready change.")


@cli.command("load-kaggle")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True))
@click.option("--anonymize/--no-anonymize", default=True)
def load_kaggle_cmd(csv_path, anonymize):
    """Convert + verify a Kaggle CSV into data/processed/players_kaggle.json."""
    from .kaggle_loader import load_and_convert, write_processed
    players, warnings = load_and_convert(Path(csv_path), anonymize_names=anonymize)
    write_processed(players, data_loader.PROCESSED / "players_kaggle.json")
    click.echo(f"Loaded {len(players)} players, {len(warnings)} warnings")
    for w in warnings[:20]:
        click.echo(f"  WARN: {w}")


@cli.command("matches")
def matches_cmd():
    """List saved matches, newest first."""
    ids = pipeline.list_matches()
    if not ids:
        click.echo("No matches yet.")
    for m in ids:
        click.echo(pipeline.load_event_log(m).label)


@cli.command("delete-match")
@click.option("--match-id", required=True)
def delete_match_cmd(match_id):
    """Delete a match's event log and report."""
    try:
        pipeline.delete_match(match_id)
    except KeyError as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Deleted {match_id}")


@cli.command("delete-version")
@click.option("--team-id", required=True)
@click.option("--version", type=int, required=True)
def delete_version_cmd(team_id, version):
    """Delete one saved version of a team (its number is never reused)."""
    try:
        pipeline.delete_version(team_id, version)
    except (KeyError, ValueError) as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Deleted {team_id} v{version}")


@cli.command("delete-team")
@click.option("--team-id", required=True)
@click.confirmation_option(prompt="Delete this team, all its versions and every match it played?")
def delete_team_cmd(team_id):
    """Delete a team, all its versions, and the matches it played."""
    try:
        matches = pipeline.delete_team(team_id)
    except KeyError as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Deleted {team_id}" + (f" and {len(matches)} match(es): {', '.join(matches)}" if matches else ""))


@cli.command("full-demo")
@click.option("--seed", type=int, default=None, help="Replay a specific match (default: new random seed)")
def full_demo_cmd(seed):
    """Runs the whole arc non-interactively (auto-accepting the top shortlist
    candidate) so you can see it end to end fast."""
    from click.testing import CliRunner
    runner = CliRunner()

    def run(args):
        click.echo(f"\n$ python -m fta.cli {' '.join(args)}")
        result = runner.invoke(cli, args, catch_exceptions=False)
        click.echo(result.output)
        return result

    brief = "more aerial ability, stay left-footed"
    run(["build-team", "--team-id", "t_A", "--name", "Alpha FC"])
    run(["build-team", "--team-id", "t_B", "--name", "Beta FC", "--exclude-team-id", "t_A"])

    run(["swap", "--team-id", "t_A", "--slot", "CB_L", "--brief", brief])
    ctx = pipeline.load_context()
    team_a = storage.load_team("t_A")
    request, _ = pipeline.swap_request_from_brief(ctx, team_a, "CB_L", brief)
    top_pick = pipeline.shortlist(ctx, team_a, request).candidates[0].player_id
    run(["apply-swap", "--team-id", "t_A", "--slot", "CB_L", "--player-id", top_pick,
         "--reason", "auto-picked top shortlist candidate (full-demo)"])

    run(["simulate", "--team-a", "t_A", "--team-b", "t_B"] + (["--seed", str(seed)] if seed else []))
    match_id = pipeline.list_matches()[0]
    run(["report", "--match-id", match_id])

    log = pipeline.load_event_log(match_id)
    side = pipeline.team_reports(pipeline.load_report(match_id), pipeline.side_for(log, "t_A"))
    weak_slot = pipeline.weakest_outfield_slot(storage.load_team("t_A"), side)
    run(["feedback", "--team-id", "t_A", "--match-id", match_id, "--slot", weak_slot])
    run(["assist", "--team-id", "t_A", "--goal", "we need to be stronger in the air at the back"])

    _banner("full-demo complete")


@cli.command("ui")
@click.option("--port", type=int, default=8501)
def ui_cmd(port):
    """Open the interactive web UI (requires: pip install -e ".[ui]")."""
    if importlib.util.find_spec("streamlit") is None:
        raise click.ClickException('Streamlit is not installed. Run: pip install -e ".[ui]"')
    app = Path(__file__).with_name("ui_app.py")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app),
                        "--server.port", str(port)], check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from None
    except KeyboardInterrupt:
        pass


def main() -> None:
    load_dotenv()
    cli()


if __name__ == "__main__":
    main()
