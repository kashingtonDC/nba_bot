# nba-bot

Logging-only prediction bot for NBA playoff series markets on Kalshi.

Compares a Bayesian-updated, net-rating-based model probability against live
Kalshi prices, logs the comparison to Supabase, and renders a dashboard.

**Live dashboard:** https://kashingtondc.github.io/nba_bot/

**This bot does not place trades.** It logs what it would do.

## Architecture

```
nba-bot/
├── bot.py                       # Main orchestrator
├── model.py                     # Pure probability functions (unit-tested)
├── factors.py                   # Four Factors (Oliver) — pure math
├── config.py                    # Constants + loads ratings.json + series_state.json
├── kalshi.py                    # Read-only Kalshi public API client
├── db.py                        # Supabase write client
├── schema.sql                   # Postgres schema (run once in Supabase)
├── migrations/                  # Idempotent column-additions for evolving schema
├── ratings.json                 # Generated: team net ratings
├── series_state.json            # Generated: current series state with box scores
├── scripts/
│   ├── refresh_ratings.py       # Pull net ratings (basketball-reference)
│   ├── refresh_series.py        # Pull series state + box scores (ESPN)
│   ├── fetch_historical.py      # One-time historical playoff fetch (basketball-reference)
│   ├── run_backtest.py          # Sweep KAPPA over historical data
│   ├── run_diagnostics.py       # Diagnostic experiments for calibration
│   └── inspect_br.py            # Diagnostic helper for BR HTML changes
├── backtest/
│   ├── historical_data.py       # Cached historical playoff scraper
│   ├── replay.py                # Walk model state-by-state through past series
│   ├── metrics.py               # log-loss, Brier, calibration curve, ECE
│   └── data/<season>/           # Cached fetch results (gitignored)
├── tests/test_model.py
└── docs/
    ├── index.html               # Static dashboard, reads from Supabase
    └── diagnostics/             # Calibration plots from run_backtest/run_diagnostics
```

The three sources of truth, kept separate:

| Source | Script | Output | Refresh cadence |
|---|---|---|---|
| Net ratings | `refresh_ratings.py` | `ratings.json` | Once at start of playoffs; nightly during regular season |
| Series state | `refresh_series.py` | `series_state.json` | Before every bot run |
| Market prices | (built into `bot.py`) | Supabase rows | Every bot run |

`config.py` reads `ratings.json` and `series_state.json` if they exist, falling
back to hardcoded values otherwise. The hardcoded values are placeholders —
always run the refresh scripts before trusting outputs.

## One-time setup

### 1. Supabase

In the Supabase SQL Editor, run the contents of `schema.sql`. This creates two
tables (`runs`, `observations`) with RLS disabled for v0. Click **"Run without
RLS"** when prompted — that's intentional.

If you set up Supabase before orderbook depth and Four Factors columns
were added, run the migrations in `migrations/` in order. They use
`IF NOT EXISTS` so they're safe to run repeatedly.

```
migrations/001_orderbook.sql       # adds orderbook depth columns
migrations/002_four_factors.sql    # adds Four Factors logging columns
```

### 2. Local environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
SUPABASE_URL=https://wgdoqojxeaurrivctqwr.supabase.co
SUPABASE_KEY=<your_anon_key>
```

### 3. Verify with tests

```bash
pytest tests/
```

All tests should pass.

## Running the bot

The standard workflow chains the three scripts:

```bash
python scripts/refresh_ratings.py   # only needed periodically
python scripts/refresh_series.py    # before every run
python bot.py                       # fetch prices, run model, log to Supabase
```

A common shortcut for repeated runs:

```bash
python scripts/refresh_series.py && python bot.py
```

After running, open `docs/index.html` in a browser to see the latest
results. It pulls from Supabase on page load; click the refresh button after
new bot runs.

## What each script does

### `scripts/refresh_ratings.py`

Fetches end-of-season team net ratings from basketball-reference. Defaults to
strength-of-schedule-adjusted NRtg/A (more predictive than raw NRtg). Writes
to `ratings.json`.

```bash
python scripts/refresh_ratings.py                # current season
python scripts/refresh_ratings.py --season 2026  # explicit season
python scripts/refresh_ratings.py --raw          # use raw NRtg instead
```

Run this once at the start of the playoffs, or whenever ratings update during
the regular season. It's safe to skip on subsequent runs — `config.py` keeps
loading the same `ratings.json` until you regenerate it.

### `scripts/refresh_series.py`

Fetches recent NBA game results from ESPN's public scoreboard API. Filters to
completed games where both teams are configured in `config.SERIES`, computes
margins from the favorite's perspective, and writes `series_state.json`.

```bash
python scripts/refresh_series.py                       # last 14 days through today
python scripts/refresh_series.py --since 2026-04-15    # explicit start
python scripts/refresh_series.py --until 2026-04-26    # explicit end
```

Run this before every `bot.py` invocation to make sure series state is current.
Only games with `STATUS_FINAL` are counted — in-progress games are correctly
ignored until they finish.

### `bot.py`

Pulls current Kalshi prices for the configured NBA series markets, computes
model probabilities (using the loaded ratings + series state), and writes one
row per series per run to the Supabase `observations` table.

Logs include:
- Model probability (raw + tail-adjusted)
- Kalshi YES bid / ask / last / mid
- Edge (model − market) for both raw and tail-adjusted
- Posterior on net-rating differential after Bayesian updates
- Per-game home/road probabilities

## Updating things by hand

Most updates flow through `refresh_ratings.py` and `refresh_series.py`. A few
things still live in `config.py` and require a manual edit:

- **Adding a new series** (e.g., once round 2 starts) — add a `SeriesState`
  entry to `_FALLBACK_SERIES` in `config.py`. Include the `kalshi_ticker_match`
  string for the favorite's YES market.
- **Tuning model constants** — `HCA`, `SIGMA_GAME`, `SIGMA_THETA`,
  `TAIL_CORRECTION_PP`. Defaults are NBA-analytics-standard.
- **Toggling tail correction** — `TAIL_CORRECTION_PP = 0` turns it off entirely
  if you want to compare model performance with vs. without.

## Math

The model uses Bayesian conjugate updates with normal distributions. All
math here renders as proper LaTeX when viewed on GitHub.

### Notation

- $\delta$ — the latent net-rating differential between favorite and underdog (favorite minus underdog)
- $c$ — prior regression coefficient (default 0.6, calibrated)
- $\mu_0 = c \cdot (\text{NRtg}_{fav} - \text{NRtg}_{und})$ — prior mean of $\delta$, regressed toward zero
- $\sigma_\theta$ — prior uncertainty on each team's true rating (default 2.0 points)
- $\sigma_g$ — single-game margin noise (default 11.5 points)
- $\text{HCA}$ — home court advantage (default 2.5 points)
- $h_i \in \{+1, -1\}$ — home indicator for game $i$ from the favorite's perspective
- $m_i$ — observed margin from the favorite's perspective in game $i$
- $\hat m_i$ — Four Factors-implied "deserved" margin in game $i$ (when available)
- $\kappa \geq 0$ — Four Factors variance-weighting coefficient (default 0)

### Prior

Each team's true net rating is treated as $\mathcal{N}(\text{NRtg}_{obs}, \sigma_\theta^2)$. The differential between two independent normal variables is normal with summed variance, with the differential shrunk by the regression coefficient $c$:

$$\delta \sim \mathcal{N}(c \cdot (\text{NRtg}_{fav} - \text{NRtg}_{und}),\ 2\sigma_\theta^2)$$

The factor $c$ encodes that regular-season NRtg overstates playoff team-strength gaps. See the "Calibration" section for how $c = 0.6$ was chosen.

### Likelihood (per game)

The observed margin in game $i$ is the differential plus home-court adjustment plus per-game noise:

$$m_i \mid \delta \sim \mathcal{N}(\delta + h_i \cdot \text{HCA}, \sigma_{g,i}^2)$$

In the basic model, $\sigma_{g,i} = \sigma_g$ (constant). With Four Factors weighting, the variance is inflated when the actual margin diverges from what the underlying play implied:

$$\sigma_{g,i}^2 = \sigma_g^2 + \kappa \cdot (m_i - \hat m_i)^2$$

When $\kappa = 0$ this reduces to the constant-variance case. When $\kappa > 0$, "fluky" wins (where one team played well below their margin or vice versa) contribute less weight to the posterior, since they're noisier evidence of true team strength.

### Four Factors expected margin

The expected margin from Dean Oliver's Four Factors is computed per game from box scores:

$$\hat m_i = \overline{\text{pace}} \cdot \big(1.6 \cdot \Delta\text{eFG\%} - 1.4 \cdot \Delta\text{TOV\%} + 0.5 \cdot \Delta\text{ORB\%} + 0.5 \cdot \Delta\text{FT-rate}\big)$$

where each $\Delta$ is the difference between the favorite's and underdog's value, and $\overline{\text{pace}}$ is the average possessions across both teams. The weights $(1.6, 1.4, 0.5, 0.5)$ are Oliver's standard relative coefficients; pace converts per-100-possessions efficiency into per-game points. See `factors.py` for the implementation.

### Posterior (after $n$ games)

The conjugate normal-normal update gives:

$$\sigma_n^2 = \left( \frac{1}{2\sigma_\theta^2} + \sum_{i=1}^{n} \frac{1}{\sigma_{g,i}^2} \right)^{-1}$$

$$\mu_n = \sigma_n^2 \left( \frac{\mu_0}{2\sigma_\theta^2} + \sum_{i=1}^{n} \frac{m_i - h_i \cdot \text{HCA}}{\sigma_{g,i}^2} \right)$$

In words: the posterior precision (1 over variance) is the prior precision plus the sum of per-game precisions. The posterior mean is the precision-weighted average of prior mean and game observations. Each game shrinks our uncertainty about $\delta$ and pulls the estimate toward the observed evidence.

### Per-game win probability

Given the current posterior on $\delta$, the probability that the favorite wins a single upcoming game (home or road) marginalizes over the belief:

$$P(\text{fav wins}) = \Phi\left( \frac{\mu_n + h \cdot \text{HCA}}{\sqrt{\sigma_n^2 + \sigma_g^2}} \right)$$

where $\Phi$ is the standard normal CDF. The $\sigma_n^2 + \sigma_g^2$ in the denominator combines uncertainty about team strength with uncertainty about per-game outcomes.

### Series-level probability

For a best-of-seven with the standard 2-2-1-1-1 home pattern, given current state $(w_f, w_u)$ wins for favorite and underdog, we enumerate all paths through the remaining games. Letting $H_g \in \{f, u\}$ be the home team for game $g$ and $p_h, p_r$ be the per-game probabilities at home and on the road, the series win probability is computed by exhaustive recursion. With memoization this is at most ~30 nodes — trivially fast and exact.

### Tail correction

A small structural adjustment applied at the price extremes, reflecting the empirically documented longshot bias on prediction markets (see Becker's analysis of 72M Kalshi trades). For market price $p$ and threshold magnitude $\tau$ (default 0.02):

$$\text{adj}(p) = \begin{cases}
+\tau \cdot \frac{p - 0.85}{0.15} & \text{if } p \geq 0.85 \\
-\tau \cdot \frac{0.15 - p}{0.15} & \text{if } p \leq 0.15 \\
0 & \text{otherwise}
\end{cases}$$

Both raw and adjusted probabilities are logged separately so the tail correction's value can be tested independently against ground truth.

## How the parameters work

The model has six knobs that control how predictions are made. Three of
them are physically meaningful constants (HCA, sigma values), three are
empirical tuning parameters calibrated from a backtest. Understanding what
each one does is the difference between using the model and being able to
debug it.

### Home-court advantage (HCA, default `2.5`)

Points added to the favorite's expected margin in a home game (or
subtracted on the road). Higher seeds host games 1, 2, 5, 7 in the
2-2-1-1-1 bracket — four of seven — so HCA cumulatively favors them. This
is the simplest and most intuitive parameter. In the backtest, sweeping
HCA from 0 to 3.5 changed log-loss by 0.001 and signed bias by 0.002.
**It's well-calibrated; we leave it alone.**

### Game-margin noise (sigma_game, default `11.5`)

Standard deviation of single-game point margin. Empirical NBA value;
higher in the playoffs would mean upsets are more likely. Affects how
fast the Bayesian update narrows around observed games.

### Prior uncertainty (sigma_theta, default `2.0`)

Standard deviation on each team's true net rating in the prior. Lower
means we trust the published NRtg as a precise estimate; higher means we
treat it as a rough hint. Affects how fast observed games dominate the
prior.

### Prior regression (`PRIOR_REGRESSION`, calibrated to `0.6`)

A multiplier applied to the regular-season NRtg differential **before**
the Bayesian prior is built. With c = 1.0, we trust regular-season NRtg
exactly. With c = 0.6, we shrink the gap by 40%, encoding the empirical
observation that **regular-season NRtg overstates playoff team-strength
gaps**.

For example, if BOS is +7 NRtg and MIA is +1, the raw differential is
+6. With c = 0.6, the model treats them as effectively (+5.8, +2.2) —
same midpoint, gap shrunk to 3.6.

Why does the gap shrink in the playoffs? Several plausible reasons,
which we don't try to disentangle in the model:
- Top teams pile up regular-season margin in blowouts of bad teams that
  don't appear in the playoffs (garbage-time inflation)
- Rotations tighten in the playoffs, removing weak bench players who
  were bad against good opponents and good against bad opponents
- Worse teams play harder in the playoffs; better teams can no longer
  coast; the gap closes
- Strength-of-schedule adjustments are tuned for regular-season
  opponents, which may not generalize

We just measure it. **See "Calibration" below for the methodology.**

### Four Factors variance weighting (`KAPPA`, default `0.0`)

Controls how much we trust each game's observed margin as evidence of
true team strength. With KAPPA = 0, every game's margin counts equally —
a 17-point win is a 17-point win. With KAPPA > 0, games where the actual
margin diverges from what Four Factors predicted get **downweighted**.

Concretely: if MIN beats DEN by 17 but Four Factors say they "deserved"
to win by only 8, KAPPA > 0 says "treat this as ~8 points of signal plus
9 points of noise." The Bayesian update is more conservative.

KAPPA enters the **likelihood** during the Bayesian update; prior
regression enters the **prior** before any games are observed. They
operate on different parts of the model and can be tuned independently.

Currently set to 0 because the calibrated improvement is within
statistical noise on our 5-season sample (see "Calibration"). The
plumbing is in place; we'll revisit if more data accumulates.

### Tail correction (`TAIL_CORRECTION_PP`, default `0.02`)

A small structural adjustment at price extremes (>85¢, <15¢), reflecting
documented longshot bias on prediction markets. **Applied to the market
price, not the model prediction.** Both raw and adjusted probabilities
are logged separately. Currently uncalibrated against NBA-specific data;
we'll revisit once enough live Kalshi prices have been logged.

## Calibration

Model parameters were tuned against five seasons of historical playoff
data (2020-21 through 2024-25; 75 series, 422 prediction states).
Methodology and results are summarized below; full data is in
`backtest_results.json` and `diagnostics_results.json`.

### What "well-calibrated" means

A model that predicts 70% probability for a class of outcomes is
well-calibrated if those outcomes actually happen 70% of the time.
A miscalibrated model can have great accuracy but misleading confidence
levels — useless for downstream decision-making.

### How the backtest works

For each historical series, we replay the model state-by-state. At each
state (pre-G1, post-G1, post-G2, ... up to the second-to-last game) the
model produces a probability that the higher seed wins the eventual
series. We compare that probability against the actual series outcome
(binary: did the higher seed win?).

A 6-game series produces 6 prediction states, each scored independently
against the same eventual outcome. Across 75 series in our sample, this
yields 422 prediction states.

We track three metrics:

- **Log-loss**: standard scoring rule; lower is better. Penalizes
  overconfidence on wrong predictions exponentially.
- **Brier score**: mean squared error between predicted probability and
  binary outcome; lower is better. Less harsh on overconfidence.
- **Expected calibration error (ECE)**: sample-size-weighted mean
  absolute gap between predicted probability and actual outcome
  frequency, bucketed by predicted probability. Directly measures
  miscalibration. ECE = 0 means perfectly calibrated.
- **Signed bias**: same as ECE but signed. Negative = the model
  overestimates; positive = underestimates.

### What the backtest revealed

The first run (with default parameters) had **log-loss 0.490**, well
below the 0.693 of a naive 50/50 model — so there's real predictive
skill — but **calibration was systematically off across the entire
prediction range**:

| Bucket | n | Predicted | Actual | Diff |
|---|---|---|---|---|
| 0.50-0.60 | 38 | 0.546 | 0.395 | -0.151 |
| 0.60-0.70 | 52 | 0.656 | 0.538 | -0.117 |
| 0.70-0.80 | 64 | 0.747 | 0.688 | -0.059 |
| 0.80-0.90 | 65 | 0.847 | 0.800 | -0.047 |

Every bucket showed the model overestimating the higher seed. Worst
miscalibration was in the middle (50-70%): when the model said "this
is roughly a coin flip favoring the higher seed," the higher seed
actually won less than half the time.

### Diagnostic experiments

Three experiments to identify the root cause:

1. **Bias by state index.** Bias was uniform across pre-series,
   post-G1, post-G2, and post-G3+ states (-0.04 to -0.07).
   This rules out "the prior is wrong but the update is fine" —
   the bias persists through the Bayesian update.

2. **HCA sweep.** Sweeping HCA across [0.0, 1.0, 1.5, 2.0, 2.5, 3.5]
   moved signed bias by 0.002 — basically noise. **HCA is not the
   source of the calibration problem.**

3. **Prior regression sweep.** Multiplying the rating differential by
   c ∈ [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] produced a strong, monotonic
   signal:

| c | Log-loss | ECE | Signed bias |
|---|---|---|---|
| 1.0 | 0.4904 | 0.064 | -0.062 |
| 0.9 | 0.4873 | 0.064 | -0.053 |
| 0.8 | 0.4855 | 0.072 | -0.044 |
| 0.7 | 0.4851 | 0.064 | -0.035 |
| **0.6** | **0.4861** | **0.044** | **-0.024** |
| 0.5 | 0.4887 | 0.033 | -0.013 |

Two competing optima: log-loss is best at c=0.7, calibration is best
at c=0.5. We chose **c = 0.6** as the midpoint — captures most of the
calibration improvement (ECE drops from 0.064 → 0.044) at minimal
log-loss cost.

### Calibration after the fix: where we landed

After applying c = 0.6, the calibration buckets look much better in
most of the prediction range — but with one stubborn exception. Here's
the table after re-running the backtest with the calibrated prior:

| Bucket | n | Predicted | Actual | Diff |
|---|---|---|---|---|
| 0.10-0.20 | 14 | 0.148 | 0.000 | -0.148 |
| 0.20-0.30 | 29 | 0.250 | 0.172 | -0.077 |
| 0.30-0.40 | 30 | 0.349 | 0.233 | -0.116 |
| 0.40-0.50 | 29 | 0.445 | 0.517 | +0.072 |
| **0.50-0.60** | **49** | **0.549** | **0.388** | **-0.161** |
| 0.60-0.70 | 73 | 0.650 | 0.658 | +0.007 |
| 0.70-0.80 | 60 | 0.753 | 0.767 | +0.014 |
| 0.80-0.90 | 70 | 0.846 | 0.843 | -0.003 |
| 0.90-1.00 | 64 | 0.952 | 1.000 | +0.048 |

**The 0.60-0.90 range is excellent.** 203 of 422 predictions (48%) land
here, and the model's confidence essentially matches reality — when
it says 75%, the higher seed wins 77% of the time.

**The 0.50-0.60 bucket is uniquely broken.** When the model says
"the higher seed is *probably* going to win, but barely" (~55% confidence),
the higher seed actually wins **39%** of the time — well below 50%.
This is 49 predictions, large enough to take seriously. Prior regression
fixed the *average* bias but couldn't fix this specific slice.

We don't have a clean explanation. Some hypotheses worth investigating:

- "Coin flip" series may be unusually upset-prone for reasons our
  features don't capture (matchup specifics, injury status, momentum)
- Specific seed matchups (e.g. 5-vs-4, where seeding gap is small)
  might cluster in this confidence range
- This bucket is dominated by post-game predictions in tight series
  where the higher seed is leading 2-1 or 3-2; those situations may
  not behave like "55% probability events" in practice

For now, **noted and left alone**. We'd need either more data or a
targeted investigation to fix it. A trader using this model should
view 50-60% predictions on the higher seed as suspect and demand a
larger market mispricing before acting on them.

### Honest limitations of this calibration

- **Sample size is small.** 75 series across 5 seasons. The c=0.6
  optimum could realistically shift to anywhere in [0.5, 0.8] with
  more data. We picked the midpoint partly to hedge against this.
- **The 0.50-0.60 bucket has unexplained miscalibration** (-0.161
  signed bias on n=49). See above. This is the single largest
  remaining model defect.
- **The 2020-21 bubble season is in the data** with weird
  home-court dynamics. Excluding it didn't materially change the
  conclusion, but it's a real distortion.
- **Tail correction was not calibrated** — we don't have historical
  market prices for past playoffs, so we can't backtest it. Will
  revisit once enough live Kalshi data is logged.
- **KAPPA was not meaningfully tunable** on this dataset. Re-running
  the sweep at c=0.6 produced log-loss differences within 0.003
  across the full [0, 2] KAPPA range — about 1/8 of a standard error.
  The "best KAPPA = 2.0" output is misleading: the differences are
  pure noise. We left KAPPA = 0 for now; the plumbing is in place
  to retune later if more data accumulates.
- **Per-team adjustments are tempting but unjustified.** Some teams
  may have unusual home-court effects (Denver's altitude, e.g.) but
  with 5 seasons of data we can't reliably distinguish that from
  noise. Future work could fold in external knowledge as hardcoded
  per-team overrides.

### Reproducing the calibration

```bash
# 1. Fetch historical data (one-time, ~25 minutes)
python scripts/fetch_historical.py

# 2. Run the diagnostic experiments
python scripts/run_diagnostics.py

# 3. Run the KAPPA sweep with the calibrated prior regression
python scripts/run_backtest.py
```

Outputs land in `backtest_results.json`, `diagnostics_results.json`, and
`docs/diagnostics/*.png`.

## Model in 30 seconds (TL;DR)

1. **Prior**: regular-season net rating differential, shrunk by 40% to
   match playoff strength gaps (`PRIOR_REGRESSION = 0.6`).
2. **Per-game probability**: normal CDF on expected margin,
   `Phi((diff + HCA) / σ_game)`.
3. **Bayesian update**: each observed game's margin is a noisy normal
   observation; standard conjugate update on a normal prior. Optionally
   variance-weighted by Four Factors (`KAPPA`, currently 0).
4. **Series probability**: enumerate the full remaining game tree
   given current state and 2-2-1-1-1 home pattern. Memoized recursion.
5. **Tail correction**: ±2pp at price extremes, applied to the market
   price (not the model). Logged separately for later evaluation.

See `model.py` for full detail. Pure, no I/O, unit-tested.

## Deployment

The bot can run locally on demand (just `python bot.py`) or as a GitHub
Actions cron that logs market state every 15 minutes. The cron + Supabase
+ GitHub Pages dashboard combo gives you a hands-off, always-on view of
the model and market without renting a server.

### Security model (Approach A+)

We use Row-Level Security (RLS) policies in Supabase to separate read and
write access:

- **Bot writes** using the service-role key. This key bypasses RLS and has
  full access. It lives in your local `.env` and as a GitHub Actions
  repository secret. Never embedded in client-side code, never committed
  to git.
- **Dashboard reads** using the anon key, which is safe to embed in HTML
  because RLS policies (in `migrations/004_enable_rls.sql`) restrict it
  to SELECT-only on `runs` and `observations`.

This means the dashboard URL can be public without worrying about anyone
tampering with the data. The worst they can do is read what's already
there — same data that's derived from public sources (Kalshi prices,
basketball-reference NRtg, ESPN scores) anyway.

### Setting up the cron

One-time setup:

1. **Create GitHub repository secrets** at Settings → Secrets and variables
   → Actions:
   - `SUPABASE_URL` (e.g. `https://wgdoqojxeaurrivctqwr.supabase.co`)
   - `SUPABASE_SERVICE_KEY` (from Supabase → Project Settings → API →
     service_role key)
2. **Apply RLS migration** by pasting `migrations/004_enable_rls.sql` into
   the Supabase SQL editor (one-time).
3. **Verify locally** that the bot reads `SUPABASE_SERVICE_KEY` from your
   `.env` and that `python bot.py` still writes successfully.
4. **Push to main**. The workflow at `.github/workflows/bot.yml` activates
   automatically and starts running every 15 minutes.
5. **Manually trigger a test run** from the Actions tab → "nba-bot cron"
   → Run workflow. Confirms the secrets and environment are configured
   correctly.

### Deploying the dashboard via GitHub Pages

1. Go to Settings → Pages in your repository
2. Under "Source", select "Deploy from a branch"
3. Choose your branch (likely `main`) and folder `/docs`
4. Save. After a few minutes the dashboard will be live at
   `https://<your-username>.github.io/<repo-name>/`

The dashboard auto-refreshes its data on page load and via the manual
"refresh" button. The "last run" indicator at the top shows how recently
the cron last logged data.

### Monitoring the cron

Three signals you can use to tell if the cron is healthy:

1. **Dashboard staleness indicator.** The "last run" timestamp at the top
   of the dashboard turns red and stops pulsing if the most recent run
   is more than 30 minutes old (i.e., at least one cron tick was missed).
2. **GitHub Actions tab.** Shows green checks for successful runs and red
   X's for failures. Click any run for full logs.
3. **Email notifications.** GitHub sends email alerts on workflow failures
   by default; you can configure these at github.com/settings/notifications.

### Off-season

The workflow has a hard cutoff at July 1, 2026 (configurable in
`.github/workflows/bot.yml`). After this date the workflow runs but exits
cleanly without doing anything, costing zero Actions minutes per run. To
extend the playoff window for next season, edit the `CUTOFF` value in the
workflow file.

If you want to disable the workflow entirely (e.g. to stop accumulating
Supabase rows during the off-season), go to the Actions tab → "nba-bot
cron" → "..." menu → "Disable workflow".

## Round 1 retrospective

The bot ran live for the entire 2025-26 NBA Round 1 (April 18 - May 5).
With the cron logging every 15 minutes (mostly — see "Cron cadence"
below), we accumulated 119 runs and 930 observations across 8 series.

This section documents what we learned from that data. Findings come
from `scripts/analyze_predictions_vs_market.py`, which produces per-
series time-series plots and summary statistics from the Supabase log.
The dashboard's "Completed series" tab shows the visualizations.

### Headline findings

**DEN-MIN: model right, market wrong.** Through the entire series, the
model said DEN had a 16-25% chance of winning while the Kalshi market
priced them at 50-65%. MIN won the series 4-2. Mean edge of -33
percentage points sustained over five days. This is the cleanest
example of the model capturing something the market missed: regular-
season NRtg + Bayesian updating from observed games gave a confident
read that DEN was overrated, while market participants were anchored
on DEN's seeding and historical strength.

**BOS-PHI: market right, model lagged.** BOS led the series 3-1 before
PHI's three-game comeback. The interesting moment was during G7: the
Kalshi market price plummeted from ~70% to ~7% in real time as the
game played out, while the model stayed at 72% until the final whistle.
This is a structural model limitation — our model treats games as
discrete events that update the posterior at completion, but the
market prices continuously based on in-game information (lineups,
injuries, momentum). Mid-game, the market's edge over the model is not
mispricing; it's information asymmetry.

**LAL-HOU: agreement is the most common case.** Both signals tracked
each other closely (Pearson correlation 0.97, mean edge +0.06).
LAL won 4-2. When the model and the market agree, that's neither a
trading opportunity nor a model failure — it's the expected case. Most
of our 8 series looked roughly like this.

**CLE-TOR: model overreacts to single games in close matchups.** Model
swung from 81% to 28% to 63% to 28% over the course of the series —
much more variance than the market (Pearson 0.30, σ 0.19). This pattern
is consistent with what we found in the 0.50-0.60 bucket calibration
investigation: when the underlying NRtg differential is small, single-
game updates are large relative to the posterior mean, and the model
becomes whippy. The Bayesian update is doing what it's supposed to do
mathematically, but it overweights individual game results in close
matchups.

### Lessons for using the model

These findings, taken together, suggest a practical reading guide:

1. **Strong agreement (|edge| < 5pp, both confident)** — trust both,
   no betting opportunity.
2. **Strong disagreement (|edge| > 20pp, model confident in one
   direction)** — possible model edge if the matchup has a wide NRtg
   gap (DEN-MIN style). Suspect if the matchup is close (CLE-TOR style).
3. **Mid-game divergence** — market is incorporating information the
   model can't see. Treat this as data lag, not mispricing.
4. **Pre-series predictions in close matchups (model 0.45-0.60)** — see
   the calibration limitations section. The model has a ~5pp bias against
   higher seeds in this range, smaller than the per-series bias in the
   0.50-0.60 bucket but still material.

### Cron cadence

GitHub's free-tier scheduled events are reliably triggered but
sometimes delayed during peak load. Our actual cadence over Round 1
averaged 1-2 hours between runs rather than the configured 15 minutes.
This is a known GitHub limitation and not specific to our project.
The data density is still plenty for time-series analysis. We added a
job-level timeout (5 minutes) to prevent any single hung run from
delaying subsequent runs.

### What's not yet captured

- We don't have round 1 box scores for series that resolved before our
  Four Factors integration was wired in (a few games early in round 1).
  This means κ-weighted variance updates are based on slightly
  incomplete data for the earliest series.
- Market-vs-model alignment doesn't tell us about *opportunities* —
  we'd need a "would-have-traded" backtest with Kelly sizing to score
  hypothetical PnL.
- The model's mid-game lag is a real limitation; closing it would
  require either ingesting in-progress game state (hard) or accepting
  that pre-game predictions are the model's primary product.

## Dashboard

`docs/index.html` is a single-file static page that reads from
Supabase using the anon key. Two tabs:

**Live**: the current snapshot.
- Latest run's timestamp (with a staleness indicator if >30 min old)
- One row per active series with model probability, market price, and edge
- Order-book depth and spread for each market
- Posterior uncertainty and Four Factors data when available

**Completed series**: the archive.
- All series the bot has logged, organized by round
- Final score and series outcome
- Summary stats: mean edge, edge volatility, model-market correlation
- Time-series plots showing how model probability and market price
  evolved over each series (added in a follow-up; see "What's not yet
  built")

Tabs are persisted in the URL hash (`#live` and `#completed`) so users
can link directly to either view.

Open `docs/index.html` directly in a browser locally, or deploy via
GitHub Pages (see "Deployment" above).

## What's not yet built

- **Time-series visualization** — dashboard shows the latest snapshot but
  not how predictions and prices have moved over time. With the cron now
  populating Supabase every 15 minutes, this becomes possible.
- **0.50-0.60 bucket investigation** — the one stubborn miscalibration
  pocket; worth digging into once we have more accumulated runs.
- **ELO as a parallel rating source** — would blend with NRtg in the prior
  and be re-evaluated against the same backtest framework.
- **Tail correction backtest** — needs accumulated live Kalshi data, which
  the cron now starts gathering.
- **Kelly sizing logic** — would compute position sizes if we were actually
  trading. Still simulation-only when added.
- **"Would-have-traded" backtest** — replay against logged prices to
  measure hypothetical PnL.

## Troubleshooting

**`No Kalshi match` for a series** — Kalshi may have closed/settled the
market, or our `kalshi_ticker_match` substring is wrong. The bot logs sample
tickers on each run; eyeball them and update `config.py` if needed.

**`series_state.json` missing some series** — usually means ESPN uses a team
abbreviation we don't recognize. The known mappings are in
`scripts/refresh_series.py` (`ESPN_TO_CONFIG_ABBREV`). Add new ones if a
diagnostic shows a mismatch.

**`ratings_debug.html` was created** — basketball-reference changed their HTML
structure. Open the file, find the ratings table, update the parser in
`scripts/refresh_ratings.py`.

**Model probability looks wrong for a series** — first check if `ratings.json`
and `series_state.json` were loaded (the bot logs this on startup). If both
loaded successfully, look at the `posterior_diff_mean` value. If that number
disagrees with intuition, the issue is upstream of the model.
