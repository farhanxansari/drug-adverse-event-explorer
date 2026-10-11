import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import age_groups, parse_cases, relabel, trend_anomalies  # noqa: E402
from fda_client import SEX_LABELS, build_search, build_url  # noqa: E402
from signals import (  # noqa: E402
    benjamini_hochberg, disproportionality, first_signal_year, signal_table, timeline_table,
)


def test_build_search_and_url():
    s = build_search("metformin", date(2020, 1, 1), date(2020, 12, 31), serious_only=True)
    assert 'generic_name:"metformin"' in s and "receivedate:[20200101 TO 20201231]" in s
    url = build_url(s, count="patient.patientsex", limit=3)
    assert " " not in url and "+AND+" in url and "+TO+" in url
    assert url.endswith("count=patient.patientsex&limit=3")


def test_disproportionality_known_values():
    # a=20, b=80, c=200, d=9800  (drug_total=100, reaction_total=220, db_total=10100)
    m = disproportionality(20, 100, 220, 10_100)
    assert math.isclose(m["prr"], (20 / 100) / (200 / 10_000))          # 10.0
    assert math.isclose(m["ror"], (20 * 9800) / (80 * 200))               # 12.25
    assert math.isclose(m["expected"], 100 * 220 / 10_100)
    assert math.isclose(m["ic"], math.log2(20.5 / (m["expected"] + 0.5)))
    assert m["prr_low"] < m["prr"] < m["prr_high"]
    assert m["ic025"] < m["ic"] < m["ic975"]
    assert m["chi2"] > 4 and m["p_value"] < 0.001


def test_degenerate_tables_return_nan():
    assert math.isnan(disproportionality(0, 100, 50, 1000)["prr"])
    assert math.isnan(disproportionality(5, 100, 5, 1000)["prr"])  # c = 0


def test_signal_table_flags_and_order():
    rx = pd.DataFrame({"term": ["COMMON", "STRONG"], "count": [50, 20]})
    # COMMON: reported at background rate; STRONG: 10x background
    out = signal_table(rx, {"COMMON": 5_050, "STRONG": 220}, drug_total=100, db_total=10_100)
    top = out.iloc[0]
    assert top["reaction"] == "STRONG" and top["methods_agreeing"] == 3
    assert out.set_index("reaction").loc["COMMON", "methods_agreeing"] == 0
    assert out["q_value"].between(0, 1).all()


def test_benjamini_hochberg():
    q = benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, None]))
    assert math.isclose(q[0], 0.03) and math.isclose(q[1], 0.04) and math.isclose(q[2], 0.04)
    assert math.isnan(q[3])


def test_timeline_first_signal_year():
    rows = [
        {"year": 2018, "a": 1, "drug_total": 50, "reaction_total": 100, "db_total": 10_000},
        {"year": 2019, "a": 15, "drug_total": 100, "reaction_total": 200, "db_total": 12_000},
    ]
    tl = timeline_table(rows)
    assert first_signal_year(tl) == 2019


def test_trend_anomalies_flags_spike():
    months = pd.date_range("2020-01-01", periods=24, freq="MS")
    reports = [100 + (i % 3) for i in range(24)]
    reports[20] = 400
    out = trend_anomalies(pd.DataFrame({"month": months, "reports": reports}))
    assert out["anomaly"].sum() == 1 and bool(out.loc[20, "anomaly"])


def test_recurring_yearly_peaks_are_seasonal_not_anomalies():
    months = pd.date_range("2018-01-01", periods=60, freq="MS")
    reports = [100 + (i % 3) for i in range(60)]
    for i in range(6, 60, 12):          # a July batch every year
        reports[i] = 500
    reports[54] = 2000                   # July of the final year is far bigger than usual
    out = trend_anomalies(pd.DataFrame({"month": months, "reports": reports}))
    assert out["seasonal"].sum() >= 2                       # years 2–4 recognised as recurring
    assert bool(out.loc[54, "anomaly"])                     # the unusually large July still flagged
    assert not out.loc[[30, 42], "anomaly"].any()


def test_seasonal_peaks_recognised_despite_growth():
    """Twice-yearly batch peaks on a drug whose volume doubles each year (like a new blockbuster)."""
    months = pd.date_range("2018-01-01", periods=96, freq="MS")
    reports = []
    for i, m in enumerate(months):
        base = 50 * 2 ** (i / 12)                       # usage doubles every year
        reports.append(round(base * (4 if m.month in (1, 7) else 1) + (i % 3)))
    reports[90] = round(50 * 2 ** (90 / 12) * 12)      # one genuinely unusual month (lift ~12x)
    out = trend_anomalies(pd.DataFrame({"month": months, "reports": reports}))
    later = out[out["month"] >= "2020-01-01"]
    peaks = later[later["peak"]]
    assert peaks["seasonal"].sum() >= 0.8 * (len(peaks) - 1)   # batch peaks recognised as recurring
    assert bool(out.loc[90, "anomaly"])                        # the real outlier is still flagged


def test_age_groups_and_relabel():
    ages = pd.DataFrame({"term": [5, 25, 70, 80, 999], "count": [1, 2, 3, 4, 5]})
    g = age_groups(ages).set_index("age_group")["count"]
    assert g["0–17"] == 1 and g["18–29"] == 2 and g["65–74"] == 3 and g["75+"] == 4
    sex = relabel(pd.DataFrame({"term": [1, 2, 0], "count": [10, 20, 1]}), SEX_LABELS)
    assert set(sex["label"]) == {"Male", "Female", "Unknown"}


def test_parse_cases():
    raw = [{
        "safetyreportid": "123", "receivedate": "20240115", "serious": "1", "occurcountry": "US",
        "primarysource": {"qualification": "1"},
        "patient": {
            "patientonsetage": "67", "patientonsetageunit": "801", "patientsex": "2",
            "reaction": [{"reactionmeddrapt": "LACTIC ACIDOSIS", "reactionoutcome": "5"}],
            "drug": [{"medicinalproduct": "METFORMIN", "drugcharacterization": "1"},
                     {"medicinalproduct": "LISINOPRIL", "drugcharacterization": "2"}],
        },
    }]
    row = parse_cases(raw).iloc[0]
    assert row["age"] == 67 and row["sex"] == "Female" and row["reporter"] == "Physician"
    assert row["suspect_drugs"] == "METFORMIN" and row["other_drugs"] == "LISINOPRIL"
    assert row["outcomes"] == "Fatal" and row["serious"]