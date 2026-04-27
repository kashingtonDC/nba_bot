-- nba-bot schema for Supabase
-- Run this in the Supabase SQL editor.

-- One row per bot invocation. Lets us group observations and track runs.
create table if not exists runs (
    id           bigserial primary key,
    started_at   timestamptz not null default now(),
    finished_at  timestamptz,
    n_markets    int,
    notes        text
);

-- One row per series per run. The core log.
create table if not exists observations (
    id                          bigserial primary key,
    run_id                      bigint references runs(id) on delete cascade,
    observed_at                 timestamptz not null default now(),

    -- Identifiers
    series_key                  text not null,        -- e.g. "DEN_MIN"
    kalshi_ticker               text,                 -- e.g. "KXNBASERIES-26MINDENR1"
    favorite                    text not null,        -- 3-letter abbrev
    underdog                    text not null,

    -- Series state at observation time
    favorite_wins               int  not null,
    underdog_wins               int  not null,
    next_game_home              text,                 -- "favorite" or "underdog"
    next_game_number            int,

    -- Model inputs
    fav_net_rating              numeric,
    und_net_rating              numeric,
    net_rating_diff             numeric,              -- fav - und (after Bayesian updates)
    posterior_uncertainty       numeric,              -- std dev of posterior on diff

    -- Model outputs
    p_fav_home                  numeric,              -- per-game P(fav wins) at home
    p_fav_road                  numeric,              -- per-game P(fav wins) on road
    p_fav_series_raw            numeric,              -- model probability fav wins series
    p_fav_series_tail_adj       numeric,              -- after tail correction

    -- Market data (Kalshi)
    market_yes_bid              numeric,              -- best bid for favorite YES
    market_yes_ask              numeric,
    market_yes_last             numeric,
    market_yes_mid              numeric,              -- (bid + ask) / 2 if both present
    market_volume               numeric,
    market_volume_24h           numeric,

    -- Edges
    edge_raw                    numeric,              -- p_fav_series_raw - market_yes_mid
    edge_tail_adj               numeric,              -- p_fav_series_tail_adj - market_yes_mid

    -- Free-form for debugging
    raw_market_payload          jsonb
);

create index if not exists observations_run_id_idx       on observations(run_id);
create index if not exists observations_series_key_idx   on observations(series_key);
create index if not exists observations_observed_at_idx  on observations(observed_at desc);

-- v0: disable RLS so the anon key can read & write. Tighten later.
alter table runs          disable row level security;
alter table observations  disable row level security;
