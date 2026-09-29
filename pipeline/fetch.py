"""
STEP 1 — get the raw data.

Pulls the NYC DOE School Quality Reports from NYC Open Data (Socrata).

That dataset is ~1.5 million rows: one row per school, per metric, per year.
We only need a handful of metrics for one report type, so the filtering
happens server-side in the query rather than by downloading everything.

The metrics we ask for are listed in config.json, so changing what the site
measures does not require editing this file.
"""
import io

import pandas as pd
import requests

from common import DATA, load_config

RAW = DATA / "raw" / "source.csv"

PAGE = 50000          # Socrata's maximum rows per request
TIMEOUT = 120


def _where(cfg):
    """SoQL filter: the metrics we want, for the school type we want."""
    metrics = "', '".join(cfg["metrics"])
    return f"report_type = '{cfg['report_type']}' AND metric_variable_name IN ('{metrics}')"


def fetch():
    cfg = load_config()
    url = cfg["source_data_url"]
    RAW.parent.mkdir(parents=True, exist_ok=True)

    frames, offset = [], 0
    while True:
        params = {
            "$select": "dbn, school_name, school_type, report_year, school_year, "
                       "metric_variable_name, metric_value",
            "$where": _where(cfg),
            "$order": "dbn, report_year, metric_variable_name",
            "$limit": PAGE,
            "$offset": offset,
        }
        print(f"fetching rows {offset:,}-{offset + PAGE:,}")
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()

        page = pd.read_csv(io.StringIO(r.text), dtype=str)
        if page.empty:
            break

        frames.append(page)
        if len(page) < PAGE:
            break
        offset += PAGE

    if not frames:
        raise SystemExit(
            "FAIL: the source returned no rows. Check report_type and metrics "
            "in config.json against the dataset."
        )

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RAW, index=False)
    print(f"fetched {len(df):,} rows covering {df['dbn'].nunique():,} schools")
    return RAW


if __name__ == "__main__":
    fetch()
