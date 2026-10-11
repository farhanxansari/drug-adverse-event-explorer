"""Drug Adverse Event Explorer: pharmacovigilance dashboard over openFDA FAERS data."""
from __future__ import annotations

import math
import os
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import age_groups, parse_cases, relabel, trend_anomalies, with_share
from fda_client import (
    AGE_FIELD, COMED_FIELD, COUNTRY_FIELD, INDICATION_FIELD, OUTCOME_FIELD, OUTCOME_LABELS,
    REACTION_FIELD, REPORTER_FIELD, REPORTER_LABELS, SERIOUSNESS_CRITERIA, SEX_FIELD, SEX_LABELS,
    OpenFDAClient, OpenFDAError, build_search, reaction_clause,
)
from report import build_report
from signals import first_signal_year, signal_table, timeline_table

st.set_page_config(page_title="Adverse Event Explorer", layout="wide")

CACHE_TTL = 24 * 60 * 60
SIGNAL_COLORS = {"0": "#9aa5b1", "1": "#f0b429", "2": "#f97316", "3": "#dc2626"}
client: OpenFDAClient  # created after the sidebar collects the key


def _configured_key() -> str | None:
    try:
        key = st.secrets.get("OPENFDA_API_KEY")
        if key:
            return str(key).strip()
    except Exception:  # no secrets file configured
        pass
    key = os.getenv("OPENFDA_API_KEY", "").strip()
    return key or None


# ---------- data loading (cached, concurrent) ----------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def drug_profile(drug: str, start: date, end: date, serious_only: bool) -> dict:
    s = build_search(drug, start, end, serious_only)
    total = client.total(s)
    if total == 0:
        return {"total": 0}

    def sub(extra: list[str]) -> str:
        return build_search(drug, start, end, serious_only, extra)

    tasks = {
        "serious": lambda: client.total(sub(["serious:1"])),
        "reactions": lambda: client.count(s, REACTION_FIELD, 100),
        "trend": lambda: client.monthly_trend(s),
        "sex": lambda: client.count(s, SEX_FIELD, 3),
        "ages": lambda: client.count(sub(["patient.patientonsetageunit:801"]), AGE_FIELD, 1000),
        "outcomes": lambda: client.count(s, OUTCOME_FIELD, 6),
        "countries": lambda: client.count(s, COUNTRY_FIELD, 10),
        "reporters": lambda: client.count(s, REPORTER_FIELD, 6),
        "comeds": lambda: client.count(s, COMED_FIELD, 25),
        "indications": lambda: client.count(s, INDICATION_FIELD, 12),
    }
    for label, field in SERIOUSNESS_CRITERIA.items():
        tasks[f"sc:{label}"] = lambda f=field: client.total(sub([f"{f}:1"]))
    res = client.gather(tasks)
    res["seriousness"] = pd.DataFrame(
        [{"criterion": label, "reports": res.pop(f"sc:{label}")} for label in SERIOUSNESS_CRITERIA]
    )
    res["total"] = total
    return res


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def background_totals(terms: tuple[str, ...], start: date, end: date, serious_only: bool) -> tuple[int, dict]:
    """Report counts across ALL drugs, the comparison group for disproportionality."""
    searches = {"__all__": build_search(None, start, end, serious_only)}
    searches.update({t: build_search(None, start, end, serious_only, [reaction_clause(t)]) for t in terms})
    res = client.totals(searches)
    return res.pop("__all__"), res


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def signal_timeline(drug: str, reaction: str, start: date, end: date, serious_only: bool) -> list[dict]:
    """Cumulative 2x2 counts from `start` to the end of each year."""
    tasks = {}
    for y in range(start.year, end.year + 1):
        stop = min(date(y, 12, 31), end)
        tasks[(y, "a")] = build_search(drug, start, stop, serious_only, [reaction_clause(reaction)])
        tasks[(y, "drug_total")] = build_search(drug, start, stop, serious_only)
        tasks[(y, "reaction_total")] = build_search(None, start, stop, serious_only, [reaction_clause(reaction)])
        tasks[(y, "db_total")] = build_search(None, start, stop, serious_only)
    res = client.totals(tasks)
    return [{"year": y, **{k: res[(y, k)] for k in ("a", "drug_total", "reaction_total", "db_total")}}
            for y in range(start.year, end.year + 1)]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def case_reports(drug: str, reaction: str, start: date, end: date, serious_only: bool) -> pd.DataFrame:
    return parse_cases(client.cases(build_search(drug, start, end, serious_only, [reaction_clause(reaction)]), 20))


# ---------- sidebar ----------
today = date.today()
with st.sidebar:
    st.header("Filters")
    drug = st.text_input("Drug (generic or brand name)", value="metformin")
    compare = st.text_input("Compare with (optional)", value="", placeholder="e.g. sitagliptin")
    years = st.slider("Report received between", 2004, today.year, (2015, today.year))
    serious_only = st.checkbox("Serious reports only")
    top_n = st.slider("Top reactions to chart", 5, 25, 10)
    scan_depth = st.slider("Reactions to scan for signals", 10, 50, 25,
                           help="Each scanned reaction costs one extra API call (cached for 24h).")
    st.caption("Try: atorvastatin, ozempic, ibuprofen, keytruda, gardasil")
    api_key = _configured_key()
    if not api_key:
        st.divider()
        api_key = st.text_input("openFDA API key", type="password",
                                help="Free, instant: https://open.fda.gov/apis/authentication/").strip()

start, end = date(years[0], 1, 1), min(date(years[1], 12, 31), today)
filters_text = f"{start:%Y}–{end:%Y}" + (" · serious reports only" if serious_only else "")

st.title("Drug Adverse Event Explorer")
st.caption(
    "Post-market safety surveillance on the FDA Adverse Event Reporting System (FAERS) via openFDA: "
    "disproportionality signal detection, signal emergence over time, and case-level review."
)

if not api_key:
    st.info(
        "openFDA requires a free API key.\n\n"
        "1. Get one instantly at https://open.fda.gov/apis/authentication/ (just enter your email).\n"
        "2. Paste it in the sidebar, or set `OPENFDA_API_KEY` as an environment variable or in "
        "`.streamlit/secrets.toml`."
    )
    st.stop()

client = OpenFDAClient(api_key=api_key)

if not drug.strip():
    st.info("Enter a drug name in the sidebar to begin.")
    st.stop()

try:
    with st.spinner(f"Fetching reports for {drug}…"):
        p = drug_profile(drug, start, end, serious_only)
    if p["total"] == 0:
        st.warning(f"No reports found for **{drug}** in this window. Check the spelling or widen the years.")
        st.stop()
    scanned = p["reactions"].head(scan_depth)
    with st.spinner("Scanning reactions against the whole FAERS database…"):
        db_total, per_reaction = background_totals(tuple(scanned["term"]), start, end, serious_only)
except OpenFDAError as exc:
    st.error(str(exc))
    st.stop()

total = p["total"]
sig = signal_table(scanned, per_reaction, total, db_total)
n_signals = int((sig["methods_agreeing"] >= 2).sum()) if not sig.empty else 0

# ---------- KPIs ----------
sex = relabel(p["sex"], SEX_LABELS).set_index("label")["count"]
known_sex = sex.get("Male", 0) + sex.get("Female", 0)
deaths = int(p["seriousness"].set_index("criterion").loc["Death", "reports"])
kpis = {
    "Reports": f"{total:,}",
    "Serious": f"{p['serious'] / total:.1%}",
    "Reports listing death": f"{deaths:,}",
    "Female (of known sex)": f"{sex.get('Female', 0) / known_sex:.0%}" if known_sex else "—",
    "Signals (≥2 methods)": f"{n_signals} of {len(sig)}",
}
for col, (k, v) in zip(st.columns(len(kpis)), kpis.items()):
    col.metric(k, v)

tab_names = ["Overview", "Reactions", "Signals", "Signal timeline", "Demographics",
             "Co-medication", "Case explorer"]
if compare.strip():
    tab_names.append(f"Compare: {compare}")
tabs = st.tabs(tab_names)

# ---------- Overview ----------
with tabs[0]:
    trend = trend_anomalies(p["trend"])
    if trend.empty:
        st.info("No dated reports in this window.")
    else:
        fig_trend = go.Figure()
        fig_trend.add_scatter(x=trend["month"], y=trend["reports"], mode="lines", name="Reports")
        fig_trend.add_scatter(x=trend["month"], y=trend["baseline"], mode="lines", name="12-month baseline",
                              line={"dash": "dot"})
        seasonal = trend[trend["seasonal"]]
        fig_trend.add_scatter(x=seasonal["month"], y=seasonal["reports"], mode="markers",
                              name="Recurring peak", marker={"size": 9, "color": "#9aa5b1",
                                                             "symbol": "circle-open", "line": {"width": 2}})
        spikes = trend[trend["anomaly"]]
        fig_trend.add_scatter(x=spikes["month"], y=spikes["reports"], mode="markers", name="Anomalous spike",
                              marker={"size": 11, "color": "#dc2626"})
        fig_trend.update_layout(title="Reports received per month, with seasonality-aware spike detection",
                                yaxis_title="Reports", legend={"orientation": "h", "y": -0.15})
        st.plotly_chart(fig_trend, width="stretch")
        st.caption(
            f"{len(spikes)} anomalous month(s), {len(seasonal)} recurring peak(s). A peak is a robust "
            "z-score above 3.5 against the trailing 12-month median and MAD; it counts as anomalous only if its "
            "lift over baseline is more than 1.5× the usual lift for that calendar month in earlier years "
            "(so a drug's overall growth doesn't count). Recurring peaks usually reflect "
            "periodic batch submissions by manufacturers. Anomalies often follow label changes, media coverage "
            "or litigation, not only new harms."
        )
    c1, c2 = st.columns(2)
    with c1:
        sc = p["seriousness"].sort_values("reports")
        st.plotly_chart(px.bar(sc, x="reports", y="criterion", orientation="h",
                               title="Seriousness criteria (a report can meet several)",
                               labels={"reports": "Reports", "criterion": ""}), width="stretch")
    with c2:
        out = relabel(p["outcomes"], OUTCOME_LABELS).sort_values("count", ascending=False)
        st.plotly_chart(px.bar(out, x="label", y="count", title="Reaction outcomes",
                               labels={"label": "", "count": "Reports"}), width="stretch")

# ---------- Reactions ----------
with tabs[1]:
    rx = with_share(p["reactions"].head(top_n), total)
    fig = px.bar(
        rx.sort_values("count"), x="count", y="term", orientation="h",
        hover_data={"pct_of_reports": ":.2f"},
        labels={"count": "Reports", "term": "", "pct_of_reports": "% of reports"},
        title=f"Top {len(rx)} reported reactions: {drug}",
    )
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download as CSV", with_share(p["reactions"], total).to_csv(index=False).encode(),
                       file_name=f"{drug}_reactions.csv", mime="text/csv")

# ---------- Signals ----------
with tabs[2]:
    with st.expander("How signals are detected", expanded=False):
        st.markdown(
            "Each reaction is placed in a 2×2 table: reports with this drug vs all other drugs, "
            "with vs without the reaction. Three standard methods are applied:\n\n"
            "| Method | Used by | Flagged when |\n|---|---|---|\n"
            "| **PRR** (Proportional Reporting Ratio) | EMA, MHRA | PRR ≥ 2, χ² ≥ 4 and ≥ 3 reports (Evans criteria) |\n"
            "| **ROR** (Reporting Odds Ratio) | Lareb, many studies | lower 95% CI > 1 and ≥ 3 reports |\n"
            "| **IC** (Bayesian Information Component) | WHO Uppsala Monitoring Centre | IC025 > 0 |\n\n"
            "Because many reactions are tested at once, χ² p-values are corrected with "
            "**Benjamini–Hochberg** (q-value = false discovery rate). Reactions where more methods agree "
            "are stronger signals."
        )
    if sig.empty:
        st.info("Not enough data for signal analysis.")
    else:
        min_methods = st.radio("Show reactions flagged by at least", [0, 1, 2, 3], index=0,
                               format_func=lambda k: "all reactions" if k == 0 else f"{k} method(s)",
                               horizontal=True)
        view = sig[sig["methods_agreeing"] >= min_methods]

        plot = sig.dropna(subset=["ror", "q_value"]).copy()
        plot["log2_ror"] = plot["ror"].apply(math.log2)
        plot["neg_log10_q"] = plot["q_value"].clip(lower=1e-300).apply(lambda q: -math.log10(q))
        plot["agree"] = plot["methods_agreeing"].astype(str)
        fig_volcano = px.scatter(
            plot, x="log2_ror", y="neg_log10_q", size="a", color="agree", hover_name="reaction",
            color_discrete_map=SIGNAL_COLORS, size_max=28,
            category_orders={"agree": ["0", "1", "2", "3"]},
            labels={"log2_ror": "log₂ ROR (effect size)", "neg_log10_q": "−log₁₀ q (FDR-adjusted significance)",
                    "agree": "Methods agreeing", "a": "Reports"},
            title="Signal volcano plot: top-right = disproportionate and significant",
        )
        fig_volcano.add_vline(x=0, line_dash="dash", line_color="#9aa5b1")
        fig_volcano.add_hline(y=-math.log10(0.05), line_dash="dot", line_color="#9aa5b1",
                              annotation_text="q = 0.05", annotation_position="top left")
        st.plotly_chart(fig_volcano, width="stretch")

        show_cols = ["reaction", "a", "expected", "prr", "chi2", "ror", "ror_low", "ror_high",
                     "ic", "ic025", "q_value", "prr_signal", "ror_signal", "ic_signal", "methods_agreeing"]
        st.dataframe(
            view[show_cols].style.format(
                {c: "{:.2f}" for c in ["expected", "prr", "chi2", "ror", "ror_low", "ror_high", "ic", "ic025"]}
                | {"q_value": "{:.1e}"}, na_rep="—"),
            width="stretch", hide_index=True,
        )
        st.caption("a = reports with drug and reaction · expected = count if drug and reaction were independent. "
                   "Watch for confounding by indication (e.g. a diabetes drug flags 'blood glucose increased').")

        report_html = build_report(drug, filters_text, kpis, sig,
                                   [fig_volcano] + ([fig_trend] if not trend.empty else []))
        d1, d2 = st.columns(2)
        d1.download_button("Download signal table (CSV)", sig.to_csv(index=False).encode(),
                           f"{drug}_signals.csv", "text/csv")
        d2.download_button("Download full signal report (HTML)", report_html.encode(),
                           f"{drug}_signal_report.html", "text/html")

# ---------- Signal timeline ----------
with tabs[3]:
    st.markdown("When did a signal first emerge? Disproportionality is recomputed on **cumulative** data "
                "up to the end of each year, the way surveillance teams monitor a drug after launch.")
    if sig.empty:
        st.info("Run a drug with reactions first.")
    else:
        reaction = st.selectbox("Reaction", sig["reaction"].tolist(), key="timeline_reaction")
        try:
            with st.spinner("Recomputing year by year…"):
                tl = timeline_table(signal_timeline(drug, reaction, start, end, serious_only))
        except OpenFDAError as exc:
            st.error(str(exc))
        else:
            tl = tl.dropna(subset=["ic"])
            if tl.empty:
                st.info("Not enough reports to build a timeline.")
            else:
                first = first_signal_year(tl)
                fig = go.Figure()
                fig.add_scatter(x=tl["year"], y=tl["ic975"], mode="lines", line={"width": 0},
                                showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=tl["year"], y=tl["ic025"], mode="lines", line={"width": 0}, fill="tonexty",
                                fillcolor="rgba(249,115,22,0.18)", name="95% credibility interval")
                fig.add_scatter(x=tl["year"], y=tl["ic"], mode="lines+markers", name="IC",
                                line={"color": "#f97316"})
                fig.add_hline(y=0, line_dash="dash", line_color="#9aa5b1")
                if first:
                    fig.add_vline(x=first, line_color="#dc2626",
                                  annotation_text=f"Signal from {first}", annotation_position="top")
                fig.update_layout(title=f"Information Component over time: {reaction}",
                                  xaxis_title="Data up to end of year", yaxis_title="IC (log₂ observed/expected)")
                st.plotly_chart(fig, width="stretch")
                st.markdown(
                    f"**First year flagged by ≥2 methods:** {first}" if first
                    else "**No year where ≥2 methods agree** in this window."
                )
                st.dataframe(tl[["year", "a", "expected", "prr", "ror_low", "ic", "ic025", "methods_agreeing"]]
                             .style.format({"expected": "{:.1f}", "prr": "{:.2f}", "ror_low": "{:.2f}",
                                            "ic": "{:.2f}", "ic025": "{:.2f}"}, na_rep="—"),
                             hide_index=True, width="stretch")

# ---------- Demographics ----------
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(relabel(p["sex"], SEX_LABELS), names="label", values="count",
                               title="Patient sex", hole=0.4), width="stretch")
    with c2:
        st.plotly_chart(px.bar(age_groups(p["ages"]), x="age_group", y="count", title="Patient age (years)",
                               labels={"age_group": "", "count": "Reports"}), width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        if not p["countries"].empty:
            st.plotly_chart(px.bar(p["countries"].sort_values("count"), x="count", y="term", orientation="h",
                                   title="Top reporting countries", labels={"term": "", "count": "Reports"}),
                            width="stretch")
    with c4:
        st.plotly_chart(px.pie(relabel(p["reporters"], REPORTER_LABELS), names="label", values="count",
                               title="Who reported", hole=0.4), width="stretch")

# ---------- Co-medication ----------
with tabs[5]:
    c1, c2 = st.columns(2)
    name = drug.strip().upper()
    comeds = p["comeds"]
    comeds = comeds[~comeds["term"].astype(str).str.upper().str.contains(name, regex=False)].head(15)
    with c1:
        st.plotly_chart(px.bar(comeds.sort_values("count"), x="count", y="term", orientation="h",
                               title="Most frequently co-reported drugs",
                               labels={"term": "", "count": "Reports"}), width="stretch")
    with c2:
        ind = p["indications"]
        ind = ind[~ind["term"].astype(str).str.upper().isin(["PRODUCT USED FOR UNKNOWN INDICATION"])]
        st.plotly_chart(px.bar(ind.sort_values("count"), x="count", y="term", orientation="h",
                               title="Reported indications (all drugs on the report)",
                               labels={"term": "", "count": "Reports"}), width="stretch")
    st.caption("Co-reported drugs suggest polypharmacy patterns and candidate interactions to review. "
               "If you searched a brand name, its generic may appear here.")

# ---------- Case explorer ----------
with tabs[6]:
    st.markdown("Drill into individual case reports, as a safety reviewer would before confirming a signal.")
    if not sig.empty:
        case_rx = st.selectbox("Reaction", sig["reaction"].tolist(), key="case_reaction")
        try:
            cases = case_reports(drug, case_rx, start, end, serious_only)
        except OpenFDAError as exc:
            st.error(str(exc))
        else:
            if cases.empty:
                st.info("No case reports returned.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Cases shown", len(cases))
                c2.metric("Serious", f"{cases['serious'].mean():.0%}")
                c3.metric("Median age", f"{cases['age'].median():.0f}" if cases["age"].notna().any() else "—")
                st.dataframe(cases, hide_index=True, width="stretch",
                             column_config={"received": st.column_config.DateColumn("received")})
                st.caption("Showing up to 20 matching reports. Suspect drugs are those the reporter "
                           "believed caused the reaction.")

# ---------- Compare ----------
if compare.strip():
    with tabs[7]:
        try:
            with st.spinner(f"Fetching {compare}…"):
                q = drug_profile(compare, start, end, serious_only)
        except OpenFDAError as exc:
            st.error(str(exc))
        else:
            if q["total"] == 0:
                st.warning(f"No reports found for **{compare}**.")
            else:
                q_deaths = int(q["seriousness"].set_index("criterion").loc["Death", "reports"])
                summary = pd.DataFrame({
                    "": ["Reports", "Serious %", "Reports listing death"],
                    drug: [f"{total:,}", f"{p['serious'] / total:.1%}", f"{deaths:,}"],
                    compare: [f"{q['total']:,}", f"{q['serious'] / q['total']:.1%}", f"{q_deaths:,}"],
                })
                st.dataframe(summary, hide_index=True, width="stretch")

                a_rx, b_rx = p["reactions"], q["reactions"]
                top_terms = list(dict.fromkeys(list(a_rx["term"][:8]) + list(b_rx["term"][:8])))

                def pct(df: pd.DataFrame, n: int) -> list[float]:
                    m = df.set_index("term")["count"]
                    return [m.get(t, 0) / n * 100 for t in top_terms]

                long = pd.DataFrame({"reaction": top_terms, drug: pct(a_rx, total),
                                     compare: pct(b_rx, q["total"])}).melt(
                    id_vars="reaction", var_name="drug", value_name="pct")
                fig = px.bar(long, x="pct", y="reaction", color="drug", barmode="group", orientation="h",
                             labels={"pct": "% of that drug's reports", "reaction": ""},
                             title="Reaction profile, normalised by report volume")
                fig.update_layout(height=600)
                st.plotly_chart(fig, width="stretch")
                st.caption("Normalising by each drug's total makes drugs with very different usage comparable.")

st.divider()
st.caption("Data: openFDA drug/event endpoint (FAERS). Not for medical decisions. "
           "openFDA terms: https://open.fda.gov/terms/")