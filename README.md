# Drug Adverse Event Explorer

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drug-adverse-event-explorer.streamlit.app/)

**Live app:** https://drug-adverse-event-explorer.streamlit.app/

A pharmacovigilance dashboard for post-market drug safety surveillance, built on the FDA Adverse Event Reporting System (FAERS) via the [openFDA API](https://open.fda.gov/apis/drug/event/).

Search any drug by generic or brand name to see:
- which adverse reactions are reported **disproportionately** often, using three regulatory-standard statistical methods
- **when** each safety signal first emerged
- unusual reporting spikes
- individual case reports behind each signal

## Demo

https://github.com/user-attachments/assets/02f6f597-6ed6-41f9-bca2-abb1d041871e

## Features

| Tab | What it does |
|---|---|
| **Overview** | Monthly report volume with **seasonality-aware anomaly detection** (rolling median + MAD, separating recurring batch peaks from true anomalies, even as volume grows), seriousness criteria (death, hospitalisation, life-threatening…), reaction outcomes |
| **Reactions** | Most-reported MedDRA reaction terms as a share of reports, CSV export |
| **Signals** | **PRR, ROR and Bayesian IC** for each reaction with confidence intervals, **Benjamini–Hochberg FDR correction**, a consensus score (how many methods agree), a volcano plot, and a downloadable **HTML signal report** |
| **Signal timeline** | Recomputes disproportionality on **cumulative yearly data** to find the first year a reaction became a signal |
| **Demographics** | Patient sex, age groups, reporting countries, reporter type (physician, pharmacist, consumer…) |
| **Co-medication** | Most frequently co-reported drugs (polypharmacy and interaction hypotheses) and reported indications |
| **Case explorer** | Individual case reports for any reaction: suspect vs concomitant drugs, outcomes, reporter |
| **Compare** | Side-by-side KPIs and reaction profiles of two drugs, normalised by report volume |

## Signal detection methodology

For drug D and reaction R, every FAERS report falls into one cell of a 2×2 table:

|                 | Reaction R | Other reactions |
|-----------------|-----------:|----------------:|
| Drug D          | a          | b               |
| All other drugs | c          | d               |

| Method | Formula | Signal criterion | Used by |
|---|---|---|---|
| **PRR** | [a/(a+b)] / [c/(c+d)] | PRR ≥ 2, Yates χ² ≥ 4, a ≥ 3 (Evans 2001) | EMA, MHRA |
| **ROR** | ad / bc | lower 95% CI > 1, a ≥ 3 | Lareb, academic studies |
| **IC** | log₂[(a + 0.5) / (E + 0.5)], where E = (a+b)(a+c)/N | IC025 > 0 (Norén 2013 approximation) | WHO Uppsala Monitoring Centre |

- **Multiple testing:** dozens of reactions are tested at once, so χ² p-values are adjusted with Benjamini–Hochberg to control the false discovery rate.
- **Consensus:** reactions flagged by 2–3 methods are treated as stronger signals.
- **Signal timeline:** the IC with its 95% credibility interval is recomputed on cumulative data at the end of each year. This mirrors how safety teams monitor a drug after launch.
- **Spike detection:** a month is flagged when its robust z-score is above 3.5, where the score is 0.6745 × (x − median) / MAD against the trailing 12 months. Median and MAD are used instead of mean and standard deviation so that one past spike doesn't hide the next.
  FAERS volumes also have recurring peaks from manufacturers' periodic batch submissions, often twice a year. Each month is measured by its **lift** (reports ÷ baseline), which removes the drug's overall growth. A peak is labelled an **anomaly** only if its lift is more than 1.5× the median lift of the same calendar month in earlier years. Otherwise it is labelled a **recurring peak**.

These are screening methods for prioritising reactions for medical review. They do **not** establish causation.

## Engineering

- **Concurrent API calls:** a thread pool runs independent queries in parallel, which makes signal scans several times faster.
- **Resilient client:** exponential-backoff retries on rate limits (429) and server errors (5xx), with clear errors for missing or invalid keys.
- **Caching:** 24-hour response caching keeps usage within openFDA rate limits.
- **Testable design:** statistics and data logic live in pure modules (`signals.py`, `analysis.py`), separate from the UI.
- **Offline tests:** a full app render runs against a mocked openFDA.
- **Deployment:** hosted on Streamlit Community Cloud, with the API key stored in Streamlit secrets.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

openFDA requires a free [API key](https://open.fda.gov/apis/authentication/) (issued instantly by email). Provide it with any of these:
- the API key box in the app's sidebar
- environment variable `OPENFDA_API_KEY`
- `.streamlit/secrets.toml` containing `OPENFDA_API_KEY = "your-key"`

## Tests

```bash
pytest -q
```

The tests cover:
- PRR/ROR/IC values checked against hand-computed 2×2 tables
- Benjamini–Hochberg adjustment
- first-signal-year detection
- spike detection, including telling recurring batch peaks from true anomalies on fast-growing drugs
- case parsing
- query building
- full rendering of all 8 tabs against a mocked API

## Project structure

```
app.py          Streamlit UI (8 tabs)
fda_client.py   openFDA client: query building, concurrency, retries with backoff
signals.py      PRR / ROR / IC, χ², Benjamini–Hochberg, signal timeline
analysis.py     Relabelling, age bins, seasonality-aware spike detection, case parsing
report.py       Standalone HTML signal report
tests/          Unit tests and mocked full-app tests
```

## Limitations

FAERS is a spontaneous reporting system. It contains duplicate reports, under-reporting and stimulated reporting (for example after media coverage), and is subject to confounding by indication. A diabetes drug will flag "blood glucose increased" because of the disease it treats. Report counts are not incidence rates. Not for medical decisions.

## Tech

Python · pandas · NumPy · Plotly · Streamlit · openFDA REST API · concurrent.futures · pytest