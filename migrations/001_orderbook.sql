-- Migration 001: add orderbook depth columns to observations.
-- Idempotent — uses IF NOT EXISTS so it's safe to run repeatedly.
-- Run this in the Supabase SQL editor.

alter table observations add column if not exists ob_best_bid       numeric;
alter table observations add column if not exists ob_best_bid_size  integer;
alter table observations add column if not exists ob_best_ask       numeric;
alter table observations add column if not exists ob_best_ask_size  integer;
alter table observations add column if not exists ob_spread         numeric;
alter table observations add column if not exists ob_mid            numeric;
alter table observations add column if not exists ob_depth_top      integer;
