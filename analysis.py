"""Pure data helpers: labelling, age bins, trend anomaly detection and case parsing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fda_client import OUTCOME_LABELS, REPORTER_LABELS, SEX_LABELS

AGE_BINS = [0, 18, 30, 45, 65, 75, 130]
AGE_LABELS = ["0–17", "18–29", "30–44", "45–64", "65–74", "75+"]


def relabel(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["label", "count"])
    out = df.assign(label=df["term"].astype(str).map(mapping).fillna("Other"))
    return out.groupby("label", as_index=False)["count"].sum()


def with_share(df: pd.DataFrame, total: int) -> pd.DataFrame:
    """Add % of reports. A report can list several reactions, so shares don't sum to 100."""
    out = df.copy()
    out["pct_of_reports"] = (out["count"] / total * 100).round(2) if total else 0.0
    return out


def age_groups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"age_group": AGE_LABELS, "count": [0] * len(AGE_LABELS)})
    ages = df.assign(age=pd.to_numeric(df["term"], errors="coerce")).dropna(subset=["age"])
    ages = ages[(ages["age"] >= 0) & (ages["age"] < 130)]
    ages["age_group"] = pd.cut(ages["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    grouped = ages.groupby("age_group", observed=False)["count"].sum().reset_index()
    grouped["age_group"] = grouped["age_group"].astype(str)
    return grouped


def trend_anomalies(trend: pd.DataFrame, window: int = 12, threshold: float = 3.5,
                    seasonal_ratio: float = 1.5) -> pd.DataFrame:
    """Flag months whose volume is far above the recent baseline, separating true anomalies
    from recurring (seasonal) peaks.

    1. Robust z-score against the trailing `window`-month median and MAD (median absolute
       deviation), so one past spike doesn't inflate the baseline the way mean/std would.
    2. FAERS receives periodic batch submissions from manufacturers, so some calendar months
       peak every year. Each month's size is measured *relative to its baseline* (lift =
       reports / baseline), which removes overall growth in a drug's usage. A peak is only an
       *anomaly* if its lift exceeds `seasonal_ratio` x the median lift of the same calendar month
       in earlier years; otherwise it is labelled `seasonal` (a recurring peak).
    """
    out = trend.copy()
    if out.empty:
        return out.assign(baseline=[], robust_z=[], lift=[], seasonal_ref=[], peak=[], anomaly=[], seasonal=[])
    past = out["reports"].shift(1)
    baseline = past.rolling(window, min_periods=6).median()
    mad = past.rolling(window, min_periods=6).apply(
        lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
    mad = mad.where(mad > 0, 1.0)
    out["baseline"] = baseline
    out["robust_z"] = 0.6745 * (out["reports"] - baseline) / mad

    # Lift over baseline removes the drug's overall growth trend.
    out["lift"] = out["reports"] / baseline.clip(lower=1)
    # Median lift of the same calendar month in all earlier years (NaN for the first year).
    month = pd.to_datetime(out["month"]).dt.month
    out["seasonal_ref"] = (
        out.assign(_m=month).groupby("_m")["lift"]
        .transform(lambda s: s.shift(1).expanding().median())
    )
    out["peak"] = out["robust_z"] > threshold
    recurring = out["seasonal_ref"].notna() & (out["lift"] <= seasonal_ratio * out["seasonal_ref"])
    out["seasonal"] = out["peak"] & recurring
    out["anomaly"] = out["peak"] & ~recurring
    return out


def parse_cases(results: list[dict]) -> pd.DataFrame:
    """Flatten raw FAERS case reports into one row per case."""
    rows = []
    for r in results:
        patient = r.get("patient", {}) or {}
        drugs = patient.get("drug", []) or []
        suspect = sorted({d.get("medicinalproduct", "?") for d in drugs if str(d.get("drugcharacterization")) == "1"})
        other = sorted({d.get("medicinalproduct", "?") for d in drugs if str(d.get("drugcharacterization")) != "1"})
        reactions = patient.get("reaction", []) or []
        outcomes = {OUTCOME_LABELS.get(str(x.get("reactionoutcome")), "") for x in reactions} - {""}
        age = patient.get("patientonsetage") if str(patient.get("patientonsetageunit")) == "801" else None
        rows.append({
            "report_id": r.get("safetyreportid"),
            "received": pd.to_datetime(str(r.get("receivedate", "")), format="%Y%m%d", errors="coerce"),
            "country": r.get("occurcountry"),
            "reporter": REPORTER_LABELS.get(str((r.get("primarysource") or {}).get("qualification")), "Unknown"),
            "age": pd.to_numeric(age, errors="coerce"),
            "sex": SEX_LABELS.get(str(patient.get("patientsex")), "Unknown"),
            "serious": str(r.get("serious")) == "1",
            "reactions": ", ".join(x.get("reactionmeddrapt", "") for x in reactions),
            "outcomes": ", ".join(sorted(outcomes)),
            "suspect_drugs": ", ".join(suspect),
            "other_drugs": ", ".join(other[:8]) + (" …" if len(other) > 8 else ""),
        })
    return pd.DataFrame(rows)