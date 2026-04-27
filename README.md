# nba-bot

Logging-only prediction bot for NBA playoff series markets on Kalshi.

Compares a Bayesian-updated, net-rating-based model probability against live
Kalshi prices, logs the comparison to Supabase, and renders a dashboard.

**This bot does not place trades.** It logs what it would do.

## Architecture

- `model.py` — pure probability functions (Pythagorean expectation + Bayesian
  updating + series path enumeration). Unit-tested. No I/O.
- `config.py` — team net ratings, current series state, model constants. Edit
  this file when ratings or series state change.
- `kalshi.py` — read-only Kalshi public API client (no auth needed).
- `db.py` — Supabase write client.
- `bot.py` — orchestrator. Fetches prices, runs model, writes a row per series.
- `dashboard/index.html` — static page that reads from Supabase via the anon
  key. Deployable as-is to GitHub Pages.
- `tests/test_model.py` — unit tests covering the model math.
- `schema.sql` — Postgres schema for Supabase.

## Setup

### 1. Supabase

Create a Supabase project (already done — `wgdoqojxeaurrivctqwr`).

In the Supabase SQL Editor, run the contents of `schema.sql`. This creates
two tables (`runs`, `observations`) and disables RLS for v0.

### 2. Local environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Supabase URL and anon key
```

### 3. Run once locally

```bash
python bot.py
```

This pulls current Kalshi prices for the configured NBA series markets, runs
the model, and writes a row per series to Supabase.

### 4. Run tests

```bash
pytest tests/
```

### 5. Dashboard

Open `dashboard/index.html` in a browser. It reads directly from Supabase via
the anon key embedded in the page. To deploy to GitHub Pages, push the repo
and enable Pages on the `dashboard/` directory.

## Updating series state

After each game completes, edit `config.py`:

1. Update the relevant `SeriesState` (wins, last game margin, who's home next)
2. Commit + push. The bot picks up the new state on its next run.

## Updating net ratings

Net ratings should be the regular-season end values from
basketball-reference. Update `TEAM_NET_RATINGS` in `config.py`. **Many values
in the initial commit are placeholders flagged with `# TODO_VERIFY`** — verify
before relying on outputs.

## Model parameters

In `config.py`:

- `HCA = 2.5` — home court advantage in points
- `SIGMA_GAME = 11.5` — std dev of single-game margin around expectation
- `SIGMA_THETA = 2.0` — prior uncertainty on team net rating
- `TAIL_CORRECTION_PP = 0.02` — magnitude of tail-bias adjustment

These are NBA-analytics-standard defaults. Both raw and tail-adjusted
probabilities are logged so the tail correction's value can be tested
independently.

## What's logged per series per run

- Model probability (raw, no tail correction)
- Model probability (tail-adjusted, for downstream comparison)
- Kalshi YES bid / ask / last price
- Implied raw edge and tail-adjusted edge
- Inputs to the model (net ratings used, series state, who's home next)
- Timestamp + run ID

This gives a full audit trail for backtesting later.

## Next steps (not in v0)

- GitHub Actions cron for automatic 5-minute polling
- Historical seed-matchup priors (currently uses static base rates)
- Quarter/eighth-Kelly sizing logic (still simulation only)
- "Would-have-traded" backtest using logged prices
