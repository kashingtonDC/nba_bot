-- Migration 004: Enable Row-Level Security on runs and observations.
--
-- Security model:
--   - service_role: full access (bypasses RLS by design). The bot uses this
--     key for inserts/updates. Stored in .env locally and as a GitHub Actions
--     secret in production.
--   - anon: SELECT only. Safe to embed in client-side code (the dashboard),
--     since visitors can read but cannot write or delete.
--   - authenticated: not used in this project.
--
-- Idempotent — safe to run repeatedly.
-- Run this in the Supabase SQL editor.

-- Enable RLS. Without policies, this would block all access (even from
-- service_role bypasses RLS, but it's defense-in-depth to enable it
-- explicitly so any future role we add starts with "no access" by default).
alter table runs           enable row level security;
alter table observations   enable row level security;

-- Drop existing policies if re-running (idempotent).
drop policy if exists "anon read runs"           on runs;
drop policy if exists "anon read observations"   on observations;

-- Allow anon role to SELECT (read) all rows. No INSERT/UPDATE/DELETE policy
-- means those operations are denied for anon. The dashboard reads using
-- this role.
create policy "anon read runs"
  on runs
  for select
  to anon
  using (true);

create policy "anon read observations"
  on observations
  for select
  to anon
  using (true);

-- service_role bypasses RLS automatically; no policies needed for it.
-- The bot, using SUPABASE_SERVICE_KEY, can read/write/delete freely.
