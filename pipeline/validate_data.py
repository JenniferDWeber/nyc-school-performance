"""
STEP 3 — refuse to publish something broken.

This runs before every commit. If it raises, the workflow stops and the live
site keeps yesterday's good data instead of getting today's bad data.

Most of these checks exist because the thing they test for actually went
wrong at some point, or would have been invisible if it had.
"""
import json
import re
import sys

import pandas as pd

from common import DATA, load_config

DBN = re.compile(r"^\d{2}[MXKQR]\d{3}$")
BOROUGHS = {"Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"}

REQUIRED = [
    "dbn", "school_name", "report_year", "borough", "district",
    "combined_index", "percentile", "is_low", "low_streak",
    "years_low", "years_ranked", "longest_low_streak", "era",
]


def validate():
    cfg = load_config()
    path = DATA / "site_data.csv"

    if not path.exists():
        raise SystemExit("FAIL: data/site_data.csv does not exist")

    df = pd.read_csv(path)
    problems = []

    if df.empty:
        problems.append("the dataset is empty")

    for col in REQUIRED:
        if col not in df.columns:
            problems.append(f"missing required column: {col}")
    if problems:
        _fail(problems)

    # --- identifiers -----------------------------------------------------
    bad_dbn = df.loc[~df["dbn"].astype(str).str.match(DBN), "dbn"].unique()
    if len(bad_dbn):
        problems.append(f"{len(bad_dbn)} malformed DBNs, e.g. {list(bad_dbn[:3])}")

    unknown = df.loc[~df["borough"].isin(BOROUGHS), "borough"].unique()
    if len(unknown):
        problems.append(f"unrecognised borough values: {list(unknown[:5])}")

    # --- the score ------------------------------------------------------
    # "Average Student Proficiency" is derived from scale scores rather than
    # from a mean of the integer levels 1-4, so it can sit slightly above 4.0
    # at schools where students score well into Level 4. Observed range across
    # all years is 1.68 to 4.30, and the handful above 4.0 are NYC's selective
    # gifted programmes. The bound below is the metric's real domain - do not
    # tighten it to 4.0 on the assumption that levels are integers.
    idx = df["combined_index"].dropna()
    if len(idx) and not idx.between(1.0, 4.5).all():
        problems.append(
            f"combined_index outside the 1.0-4.5 proficiency-rating range "
            f"(min {idx.min()}, max {idx.max()})"
        )

    # The pass-rate metrics arrive as proportions, not percentages. If they
    # ever switch to 0-100 the site would silently render "47%" as "4700%".
    for col in ("ela_prof_pct", "math_prof_pct"):
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) and not vals.between(0, 1).all():
                problems.append(
                    f"{col} is not a 0-1 proportion "
                    f"(min {vals.min()}, max {vals.max()})"
                )

    pct = df["percentile"].dropna()
    if len(pct) and not pct.between(0, 100).all():
        problems.append(f"percentile outside 0-100 (min {pct.min()}, max {pct.max()})")

    # --- the ranking ----------------------------------------------------
    # A percentile cutoff should select roughly that share of ranked schools
    # every year. A big miss means the ranking did not partition correctly.
    cutoff = cfg["low_percentile_cutoff"]
    ranked = df[df["combined_index"].notna()]
    if ranked.empty:
        problems.append("no school was ranked in any year")
    else:
        share = ranked.groupby("report_year")["is_low"].apply(lambda s: (s == True).mean() * 100)
        off = share[(share < cutoff - 3) | (share > cutoff + 3)]
        for year, value in off.items():
            problems.append(
                f"{year}: {value:.1f}% of ranked schools flagged low, expected about {cutoff}%"
            )

    # A streak cannot be longer than the number of years a school was ranked.
    impossible = df[df["longest_low_streak"] > df["years_ranked"]]
    if len(impossible):
        problems.append(
            f"{len(impossible)} rows have a low streak longer than their ranked years"
        )

    if (df["years_low"] > df["years_ranked"]).any():
        problems.append("some schools are low in more years than they were ranked")

    # --- the standards break --------------------------------------------
    # The whole method rests on showing the two eras apart. If one vanishes,
    # the page would silently start comparing across the 2023 change.
    eras = set(df["era"].dropna().unique())
    if len(eras) < 2:
        problems.append(f"expected two standards eras in the data, found {eras}")

    # --- guard against a gutted source -----------------------------------
    meta_path = DATA / "meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text()).get("rows")
        if previous and len(df) < previous * 0.5:
            problems.append(
                f"row count fell from {previous} to {len(df)} - "
                "that looks like a broken source, not real change"
            )

    if problems:
        _fail(problems)

    print(f"validation passed: {len(df):,} rows, "
          f"{df['dbn'].nunique():,} schools, {ranked['report_year'].nunique()} ranked years")
    return df


def _fail(problems):
    print("VALIDATION FAILED:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    validate()
