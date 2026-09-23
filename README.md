# Football Team Agent

Build a football XI from plain-English requirements, test changes over simulated
matches, play matches, and get role-aware ratings and feedback. An AI assistant
(optional) investigates with tools and proposes changes. Code checks every claim
it makes, and nothing changes until you approve.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py    # http://localhost:8501
```

For the AI, copy `.env.example` to `.env` and add an OpenRouter key. Without a
key, everything still works using rules-based fallbacks.

## The data

**Players** (`data/processed/players_synthetic.json`, 100 demo players):

| Field | Meaning |
|---|---|
| `name`, `age`, `nationality` | Identity (nationality is a 2-letter code) |
| `preferred_foot`, `weak_foot_rating` | Left/right, and weak-foot quality 1–5 |
| `natural_positions` | GK, CB, FB, DM, CM, AM, WING, ST |
| `stamina_base` | How long he lasts. Low values fade after the hour |
| `attributes` | 24 skills, each 0–100 (below) |

| Group | Attributes |
|---|---|
| Pace | pace, acceleration |
| Endurance | stamina |
| Passing | passing_short, passing_long, through_ball, pass_power_bullet, crossing, vision |
| Shooting | finishing, composure |
| Aerial | heading_accuracy, heading_power |
| On the ball | dribbling, ball_control |
| Defending | tackling_standing, tackling_sliding, marking, positioning, aggression |
| Goalkeeping (keepers only) | gk_reflexes, gk_handling, gk_positioning, gk_kicking |

**Rules** (`data/templates/`):

| File | Contents |
|---|---|
| `positions.json` | How much each attribute counts for each position |
| `formations.json` | The slots of each formation (4-3-3, 4-2-3-1) |
| `formation_weights.json` | Each position's role and weights per formation |
| `chemistry_rules.json` | The four chemistry rules and their point ranges |

**How the numbers work:**
- **Fit** = Σ weight × attribute (0–100): how well a player suits a position.
- **Rank** (in a shortlist) = fit − penalty for missed requirements + chemistry change.
- **Chemistry** = 100 + familiarity, back-line feet, midfield stamina and shared nationality.
- **Ratings** start at 6.0 and add or subtract points per action, by role.

The optional FIFA 22 pool (19,239 real players) stays on your machine and is
never committed: run `scripts/download_kaggle_players.py`, then `fta load-kaggle`.

## What each part does

| Where | What |
|---|---|
| **Squad** tab | Build a team; ask the **Assistant** for changes; change players by request; test changes over many matches; view form and history; tune position weights; view versions |
| **Matches** tab | Play any two team versions; scoreboard, stats, coach's debrief, the assistant's analysis, every player's rating and feedback |
| **Players** tab | Browse and filter the pool; rank everyone by your own weights; ⓘ opens a player's profile, scouting report, matches and comparison |
| `src/fta/scoring.py` | Fit, requirements, chemistry |
| `src/fta/team_builder.py`, `swap_engine.py` | Pick the XI; rank replacements |
| `src/fta/simulator.py` | 90-minute match engine (seeded duels) |
| `src/fta/report_generator.py` | Per-side stats, ratings, feedback, debrief |
| `src/fta/llm_client.py` | Reads requests with the AI (validated), plus the OpenRouter transport |
| `src/fta/agent.py` | The assistant: tools, verification, fact-check, offline fallbacks |
| `src/fta/pipeline.py` | Every action as a function (shared by UI and CLI) |
| `src/fta/ui_app.py`, `cli.py` | Streamlit UI and `fta` command line |
| `scripts/` | Generate the demo pool, download FIFA 22, evaluate request reading |
| `tests/` | `pytest` (the AI is stubbed; no API calls) |

## Settings (`.env` locally, **Secrets** on Streamlit Cloud)

| Key | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | (none) | Turns the AI on |
| `MODEL` | `deepseek/deepseek-chat` | Also selectable in the sidebar |
| `BUDGET_USD`, `BUDGET_MODE` | `5.0`, `warn` | With `block`, AI calls stop once the budget is spent |
| `PLAYER_SOURCE` | `synthetic` | `synthetic`, `kaggle` or `both` |

Teams, matches and reports are saved as JSON in `teams/`, `matches/` and
`scouting/`, and AI costs are logged to `logs/`. All four are created at run
time and ignored by git.
