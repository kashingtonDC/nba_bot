-- Migration 003: log the prior_regression coefficient used for each prediction.
-- Idempotent — safe to run repeatedly.
-- Run this in the Supabase SQL editor.

alter table observations add column if not exists prior_regression numeric;
