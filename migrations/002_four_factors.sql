-- Migration 002: add Four Factors columns to observations.
-- Idempotent — safe to run repeatedly.
-- Run this in the Supabase SQL editor.

alter table observations add column if not exists kappa                  numeric;
alter table observations add column if not exists n_games_with_factors   integer;
alter table observations add column if not exists avg_observed_margin    numeric;
alter table observations add column if not exists avg_expected_margin    numeric;
alter table observations add column if not exists avg_margin_divergence  numeric;
