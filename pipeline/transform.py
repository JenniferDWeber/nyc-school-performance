"""
STEP 2 — turn raw data into exactly what the site needs.

The raw feed is long: one row per school, per metric, per year. The site needs
one row per school, per year, with each metric as a column and a ranking
attached.

The ranking is the point of this file. A school's raw score cannot be compared
across the 2023 standards change, but its position relative to other NYC
schools in the same year can be. So every score is converted to a percentile
within its own year, and "low" means the bottom slice of that year's ranking.
"""
from datetime import datetime, timezone

import pandas as pd

from common import DATA, load_config, write_json
from fetch import RAW

BOROUGHS = {
    "M": "Manhattan",
    "X": "Bronx",
    "K": "Brooklyn",
    "Q": "Queens",
    "R": "Staten Island",
}


def _widen(df, cfg):
    """Long (one row per metric) to wide (one row per school-year)."""
    df = df[df["metric_variable_name"].isin(cfg["metrics"])].copy()
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")

    wide = df.pivot_table(
        index=["dbn", "school_name", "report_year", "school_year"],
        columns="metric_variable_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide.rename(columns=cfg["metrics"])


def _geography(wide):
    """District and borough are encoded in the DBN: 01M015 -> district 1, Manhattan."""
    dbn = wide["dbn"].astype(str)
    wide["district"] = pd.to_numeric(dbn.str[:2], errors="coerce").astype("Int64")
    wide["borough"] = dbn.str[2].map(BOROUGHS).fillna("Unknown")
    # District 84 is how the DOE codes charter schools.
    wide["is_charter"] = wide["district"].eq(84)
    return wide


def _pool_columns(wide, cfg, mask, suffix):
    """
    Rank, flag, and count streaks over one pool of schools.

    Run twice: once over every school, once over district schools only.
    Whether charters sit in the comparison pool moves roughly eighteen schools
    across the bottom-decile line, so both answers are published and the
    reader chooses, rather than the choice being buried in this file.

    Schools outside the pool get no value for that pool, not a zero.
    """
    idx = wide["combined_index"].where(mask)

    percentile = (idx.groupby(wide["report_year"]).rank(pct=True) * 100).round(1)
    low = percentile.le(cfg["low_percentile_cutoff"]).astype("boolean")
    low[idx.isna()] = pd.NA

    # Streaks. A year when no test was given anywhere does not break a run;
    # a year when this school alone has no score does.
    tested = sorted(int(y) for y in wide.loc[idx.notna(), "report_year"].unique())
    frame = pd.DataFrame({
        "dbn": wide["dbn"],
        "year": wide["report_year"].astype(int),
        "low": low.fillna(False).astype(bool),
        "scored": idx.notna(),
    })

    streak_at = {}
    for dbn, grp in frame.groupby("dbn", sort=False):
        seen = {y: (l if sc else None)
                for y, l, sc in zip(grp["year"], grp["low"], grp["scored"])}
        run = 0
        for year in tested:
            run = run + 1 if seen.get(year) else 0
            streak_at[(dbn, year)] = run

    streak = pd.Series(
        [streak_at.get((d, y), 0) for d, y in zip(frame["dbn"], frame["year"])],
        index=wide.index,
    )

    out = pd.DataFrame(index=wide.index)
    out[f"percentile{suffix}"] = percentile
    out[f"is_low{suffix}"] = low
    out[f"low_streak{suffix}"] = streak.where(mask)

    by_school = out.groupby(wide["dbn"])
    out[f"years_low{suffix}"] = by_school[f"is_low{suffix}"].transform(
        lambda s: (s == True).sum()
    )
    out[f"longest_low_streak{suffix}"] = by_school[f"low_streak{suffix}"].transform("max")
    out[f"persistently_low{suffix}"] = (
        out[f"longest_low_streak{suffix}"] >= cfg["persistence_years"]
    )
    return out


def _rank(wide, cfg):
    """Percentile within each year. 0 is the bottom of the city, 100 the top."""
    wide["combined_index"] = wide[cfg["index_components"]].mean(axis=1, skipna=False)
    wide["years_ranked"] = wide.groupby("dbn")["combined_index"].transform("count")

    everyone = pd.Series(True, index=wide.index)
    district_only = ~wide["is_charter"]

    return pd.concat(
        [
            wide,
            _pool_columns(wide, cfg, everyone, ""),
            _pool_columns(wide, cfg, district_only, "_district_only"),
        ],
        axis=1,
    )


def transform():
    cfg = load_config()
    df = pd.read_csv(RAW, dtype=str)

    wide = _widen(df, cfg)
    wide["report_year"] = wide["report_year"].astype(int)
    wide["school_year"] = wide["school_year"].astype(int)

    wide = _geography(wide)
    wide = _rank(wide, cfg)

    # The 2023 assessments began a new standards regime. Scores either side of
    # that line are not comparable; percentiles are the bridge.
    wide["era"] = wide["report_year"].ge(cfg["new_standards_report_year"]).map(
        {True: "Next Generation standards", False: "Prior standards"}
    )

    wide = wide.sort_values(["dbn", "report_year"]).reset_index(drop=True)

    out = DATA / "site_data.csv"
    wide.to_csv(out, index=False)

    latest = int(wide["report_year"].max())
    current = wide[wide["report_year"].eq(latest)]

    write_json(DATA / "meta.json", {
        "rows": int(len(wide)),
        "schools": int(wide["dbn"].nunique()),
        "years": sorted(int(y) for y in wide["report_year"].unique()),
        "latest_report_year": latest,
        "schools_latest": int(current["dbn"].nunique()),
        "persistently_low": int(current["persistently_low"].sum()),
        "persistently_low_district_only": int(current["persistently_low_district_only"].sum()),
        "charters": int(current["is_charter"].sum()),
        "updated": datetime.now(timezone.utc).strftime("%B %d, %Y"),
        "updated_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_name": cfg["source_name"],
        "source_url": cfg["source_url"],
    })

    print(f"wrote {len(wide):,} school-year rows "
          f"({wide['dbn'].nunique():,} schools, {wide['report_year'].nunique()} years)")
    return wide


if __name__ == "__main__":
    transform()
