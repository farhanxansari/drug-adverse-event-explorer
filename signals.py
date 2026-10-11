"""Disproportionality analysis for spontaneous adverse event reports.

For drug D and reaction R, every report falls into one cell of a 2x2 table:

                        reaction R    other reactions
    drug D                  a               b
    all other drugs         c               d

Three standard methods are computed from it:
- PRR  (Proportional Reporting Ratio), flagged with the Evans criteria: PRR >= 2, chi² >= 4, a >= 3
- ROR  (Reporting Odds Ratio), flagged when the lower 95% CI > 1 and a >= 3
- IC   (Information Component, Bayesian shrinkage as used by WHO-UMC), flagged when IC025 > 0
p-values from the Yates-corrected chi² are adjusted for multiple testing with Benjamini–Hochberg.
"""
from __future__ import annotations

import math

import pandas as pd

Z = 1.96
NAN = math.nan


def contingency(a: int, drug_total: int, reaction_total: int, db_total: int) -> tuple[int, int, int, int]:
    b = drug_total - a
    c = reaction_total - a
    d = db_total - drug_total - c
    return a, b, c, d


def _chi2_p(x: float) -> float:
    """Upper-tail p-value of a chi² statistic with 1 degree of freedom."""
    return math.erfc(math.sqrt(x / 2)) if x > 0 else 1.0


def disproportionality(a: int, drug_total: int, reaction_total: int, db_total: int) -> dict:
    a, b, c, d = contingency(a, drug_total, reaction_total, db_total)
    out = {k: NAN for k in ("expected", "prr", "prr_low", "prr_high", "chi2", "p_value",
                            "ror", "ror_low", "ror_high", "ic", "ic025", "ic975")}
    out["a"] = a
    if a <= 0 or min(b, c, d) < 0:
        return out
    n = a + b + c + d

    expected = (a + b) * (a + c) / n
    out["expected"] = expected

    # PRR
    if c > 0 and (a + b) > 0 and (c + d) > 0:
        prr = (a / (a + b)) / (c / (c + d))
        se = math.sqrt(max(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d), 0))
        out.update(prr=prr, prr_low=math.exp(math.log(prr) - Z * se),
                   prr_high=math.exp(math.log(prr) + Z * se))

    # ROR
    if b > 0 and c > 0 and d > 0:
        ror = (a * d) / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        out.update(ror=ror, ror_low=math.exp(math.log(ror) - Z * se),
                   ror_high=math.exp(math.log(ror) + Z * se))

    # Yates-corrected chi²
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom > 0:
        chi2 = n * max(abs(a * d - b * c) - n / 2, 0) ** 2 / denom
        out.update(chi2=chi2, p_value=_chi2_p(chi2))

    # Information Component with Norén et al. (2013) credibility-interval approximation
    ic = math.log2((a + 0.5) / (expected + 0.5))
    out.update(ic=ic,
               ic025=ic - 3.3 * (a + 0.5) ** -0.5 - 2 * (a + 0.5) ** -1.5,
               ic975=ic + 2.4 * (a + 0.5) ** -0.5 - 0.5 * (a + 0.5) ** -1.5)
    return out


def flags(m: dict) -> dict:
    def ok(x):
        return x == x  # not NaN

    prr_sig = m["a"] >= 3 and ok(m["prr"]) and m["prr"] >= 2 and ok(m["chi2"]) and m["chi2"] >= 4
    ror_sig = m["a"] >= 3 and ok(m["ror_low"]) and m["ror_low"] > 1
    ic_sig = ok(m["ic025"]) and m["ic025"] > 0
    return {"prr_signal": prr_sig, "ror_signal": ror_sig, "ic_signal": ic_sig,
            "methods_agreeing": int(prr_sig) + int(ror_sig) + int(ic_sig)}


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """False-discovery-rate adjusted p-values (q-values). NaNs are left as NaN."""
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return pd.Series(NAN, index=p.index)
    ranked = valid * m / pd.Series(range(1, m + 1), index=valid.index)
    q = ranked[::-1].cummin()[::-1].clip(upper=1.0)
    return q.reindex(p.index)


def signal_table(reactions: pd.DataFrame, reaction_totals: dict[str, int],
                 drug_total: int, db_total: int) -> pd.DataFrame:
    rows = []
    for term, a in zip(reactions["term"], reactions["count"]):
        m = disproportionality(int(a), drug_total, reaction_totals.get(term, 0), db_total)
        rows.append({"reaction": term, **m, **flags(m)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_value"] = benjamini_hochberg(df["p_value"])
    return df.sort_values(["methods_agreeing", "ic025"], ascending=[False, False],
                          na_position="last").reset_index(drop=True)


def timeline_table(rows: list[dict]) -> pd.DataFrame:
    """rows: [{year, a, drug_total, reaction_total, db_total}] using cumulative counts."""
    out = []
    for r in rows:
        m = disproportionality(r["a"], r["drug_total"], r["reaction_total"], r["db_total"])
        out.append({"year": r["year"], **m, **flags(m)})
    return pd.DataFrame(out)


def first_signal_year(timeline: pd.DataFrame, min_methods: int = 2) -> int | None:
    hit = timeline[timeline["methods_agreeing"] >= min_methods]
    return int(hit["year"].iloc[0]) if not hit.empty else None
