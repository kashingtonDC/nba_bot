"""
Supabase write client.

The bot writes to two tables:
  - `runs`: one row per bot invocation
  - `observations`: one row per series per run

For v0 we use the anon key with RLS disabled on these tables. When we move to
GitHub Actions, we'll switch to the service_role key (stored as a GH secret)
and re-enable RLS.
"""
from __future__ import annotations
import os
import logging
from typing import Any, Dict, List, Optional

from supabase import create_client, Client

log = logging.getLogger(__name__)


def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set (see .env.example)"
        )
    return create_client(url, key)


def start_run(client: Client, n_markets: Optional[int] = None, notes: Optional[str] = None) -> int:
    """Insert a row into `runs` and return its id."""
    payload = {"n_markets": n_markets, "notes": notes}
    res = client.table("runs").insert(payload).execute()
    if not res.data:
        raise RuntimeError(f"Failed to create run row: {res}")
    return int(res.data[0]["id"])


def finish_run(client: Client, run_id: int, n_markets: int) -> None:
    """Update the run with finish time and final market count."""
    from datetime import datetime, timezone
    client.table("runs").update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_markets": n_markets,
    }).eq("id", run_id).execute()


def insert_observations(client: Client, rows: List[Dict[str, Any]]) -> None:
    """Bulk-insert observations."""
    if not rows:
        return
    res = client.table("observations").insert(rows).execute()
    if not res.data:
        raise RuntimeError(f"Failed to insert observations: {res}")
    log.info(f"Inserted {len(res.data)} observation rows")
