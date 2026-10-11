"""Standalone HTML signal report (opens in any browser, prints to PDF)."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1f2933}
h1{margin-bottom:4px} .muted{color:#616e7c;font-size:14px}
.kpis{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}
.kpi{border:1px solid #e4e7eb;border-radius:8px;padding:12px 16px;min-width:150px}
.kpi b{display:block;font-size:22px}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{border-bottom:1px solid #e4e7eb;padding:6px 8px;text-align:right}
th:first-child,td:first-child{text-align:left} th{background:#f5f7fa}
.note{background:#fffbea;border-left:4px solid #f0b429;padding:10px 14px;font-size:13px;margin-top:24px}
"""


def build_report(drug: str, filters: str, kpis: dict[str, str], signals: pd.DataFrame,
                 figures: list) -> str:
    kpi_html = "".join(f"<div class='kpi'>{html.escape(k)}<b>{html.escape(v)}</b></div>" for k, v in kpis.items())
    cols = ["reaction", "a", "expected", "prr", "ror", "ror_low", "ic", "ic025", "q_value", "methods_agreeing"]
    table = signals[[c for c in cols if c in signals.columns]].to_html(
        index=False, float_format=lambda x: f"{x:.3g}", border=0, na_rep="—")
    figs = "".join(
        f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(figures)
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Safety signal report: {html.escape(drug)}</title><style>{CSS}</style></head><body>
<h1>Safety signal report: {html.escape(drug)}</h1>
<div class="muted">{html.escape(filters)} · generated {datetime.now():%Y-%m-%d %H:%M} · source: openFDA / FAERS</div>
<div class="kpis">{kpi_html}</div>
{figs}
<h2>Disproportionality results</h2>
{table}
<div class="note"><b>Interpretation.</b> a = reports with the drug and reaction; expected = count expected if the
drug and reaction were independent. A signal is a statistical association warranting review, not proof of causation.
FAERS data contain duplicates, reporting bias and confounding by indication.</div>
</body></html>"""
