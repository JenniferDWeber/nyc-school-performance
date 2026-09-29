"""
STEP 4 — render the site from the data.

Reads templates/index.html, fills it in, and writes site/. The site folder is
generated output: edit the template, not the result.

The published CSV is one row per school-year. The page wants one record per
school with a short array per measure, which is both smaller to ship and the
shape the charts actually read.
"""
import json
import math
import shutil

import pandas as pd
from jinja2 import Template

from common import DATA, SITE, TEMPLATES, load_config

# CSV column -> short key in the page payload. Short keys because this ships
# inline and the same names repeat once per school.
SERIES = {
    "percentile": "p",
    "percentile_district_only": "q",
    "combined_index": "x",
    "ela_mean": "e",
    "math_mean": "m",
    "attendance": "a",
}

SUMMARY = {
    "years_low": "yl",
    "longest_low_streak": "ls",
    "persistently_low": "pl",
    "years_low_district_only": "yl2",
    "longest_low_streak_district_only": "ls2",
    "persistently_low_district_only": "pl2",
}


def _clean(v):
    """JSON has no NaN. An unranked year must travel as null, never as zero."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (bool, int, str)):
        return v
    return round(float(v), 3)


def _payload(df, cfg):
    tested = sorted(
        int(y) for y in df.loc[df["combined_index"].notna(), "report_year"].unique()
    )

    # Schools that still report in the most recent year. Roughly eighty in the
    # file have closed or stopped reporting since 2016, and counting those in a
    # headline about how schools are doing now would overstate it.
    newest = int(df["report_year"].max())
    still_open = set(df.loc[df["report_year"].eq(newest), "dbn"])

    schools = []
    for dbn, grp in df.groupby("dbn", sort=False):
        grp = grp.set_index("report_year")
        first = grp.iloc[0]

        record = {
            "d": dbn,
            "n": str(first["school_name"]),
            "b": str(first["borough"]),
            "g": None if pd.isna(first["district"]) else int(first["district"]),
            "c": bool(first["is_charter"]),
            "o": dbn in still_open,
        }
        for col, key in SERIES.items():
            record[key] = [
                _clean(grp[col].get(y)) if y in grp.index else None for y in tested
            ]
        for col, key in SUMMARY.items():
            value = first[col]
            # A charter has no district-only summary: it is not in that pool.
            # That must travel as null, never as a zero that reads as "never low".
            if pd.isna(value):
                record[key] = None
            else:
                record[key] = bool(value) if key.startswith("pl") else int(value)
        schools.append(record)

    schools.sort(key=lambda s: s["n"].lower())

    ranked = df[df["combined_index"].notna()]
    median = ranked.groupby("report_year")["combined_index"].median()

    return {
        "years": tested,
        "breakYear": cfg["new_standards_report_year"],
        "cutoff": cfg["low_percentile_cutoff"],
        "persistenceYears": cfg["persistence_years"],
        "medianIndex": [_clean(median.get(y)) for y in tested],
        "schools": schools,
    }


def build():
    cfg = load_config()
    df = pd.read_csv(DATA / "site_data.csv")
    meta = json.loads((DATA / "meta.json").read_text())

    data = _payload(df, cfg)

    html = Template((TEMPLATES / "index.html").read_text(encoding="utf-8")).render(
        cfg=cfg,
        meta=meta,
        data_json=json.dumps(data, separators=(",", ":")),
    )

    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(html, encoding="utf-8")

    # The raw data, downloadable from the site.
    shutil.copy(DATA / "site_data.csv", SITE / "data.csv")

    for asset in ("favicon.svg", "preview.png"):
        src = TEMPLATES / asset
        if src.exists():
            shutil.copy(src, SITE / asset)

    size = (SITE / "index.html").stat().st_size / 1024
    print(f"built site/index.html - {len(data['schools']):,} schools, {size:,.0f} KB")


if __name__ == "__main__":
    build()
