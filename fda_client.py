"""Client for the openFDA drug adverse event (FAERS) endpoint.

Docs: https://open.fda.gov/apis/drug/event/
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, Callable
from urllib.parse import quote

import pandas as pd
import requests

BASE_URL = "https://api.fda.gov/drug/event.json"
TIMEOUT = 30
RETRY_STATUSES = {429, 500, 502, 503, 504}

# Field names used across the app
REACTION_FIELD = "patient.reaction.reactionmeddrapt.exact"
SEX_FIELD = "patient.patientsex"
AGE_FIELD = "patient.patientonsetage"
OUTCOME_FIELD = "patient.reaction.reactionoutcome"
COUNTRY_FIELD = "occurcountry.exact"
COMED_FIELD = "patient.drug.openfda.generic_name.exact"
INDICATION_FIELD = "patient.drug.drugindication.exact"
REPORTER_FIELD = "primarysource.qualification"

SEX_LABELS = {"0": "Unknown", "1": "Male", "2": "Female"}
OUTCOME_LABELS = {
    "1": "Recovered",
    "2": "Recovering",
    "3": "Not recovered",
    "4": "Recovered with sequelae",
    "5": "Fatal",
    "6": "Unknown",
}
REPORTER_LABELS = {
    "1": "Physician",
    "2": "Pharmacist",
    "3": "Other health professional",
    "4": "Lawyer",
    "5": "Consumer",
}
SERIOUSNESS_CRITERIA = {
    "Death": "seriousnessdeath",
    "Hospitalisation": "seriousnesshospitalization",
    "Life-threatening": "seriousnesslifethreatening",
    "Disability": "seriousnessdisabling",
    "Congenital anomaly": "seriousnesscongenitalanomali",
    "Other serious": "seriousnessother",
}


class OpenFDAError(RuntimeError):
    """Raised when openFDA returns an error we can't treat as 'no results'."""


def build_search(
    drug: str | None = None,
    start: date | None = None,
    end: date | None = None,
    serious_only: bool = False,
    extra: list[str] | None = None,
) -> str:
    """Build an openFDA search expression (spaces are encoded later)."""
    clauses: list[str] = []
    if drug and drug.strip():
        name = drug.strip().replace('"', "")
        clauses.append(
            f'(patient.drug.openfda.generic_name:"{name}" OR '
            f'patient.drug.openfda.brand_name:"{name}")'
        )
    if start and end:
        clauses.append(f"receivedate:[{start:%Y%m%d} TO {end:%Y%m%d}]")
    if serious_only:
        clauses.append("serious:1")
    if extra:
        clauses.extend(extra)
    return " AND ".join(clauses)


def reaction_clause(term: str) -> str:
    return f'{REACTION_FIELD}:"{term.replace(chr(34), "")}"'


def build_url(search: str = "", count: str | None = None, limit: int | None = None,
              api_key: str | None = None) -> str:
    """openFDA wants spaces as '+', so the query string is built by hand."""
    params: list[str] = []
    if search:
        params.append("search=" + quote(search.replace(" ", "+"), safe='+:"[]()*.-'))
    if count:
        params.append(f"count={count}")
    if limit is not None:
        params.append(f"limit={limit}")
    if api_key:
        params.append(f"api_key={quote(api_key)}")
    return BASE_URL + ("?" + "&".join(params) if params else "")


class OpenFDAClient:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None,
                 max_retries: int = 3, backoff: float = 1.0, workers: int = 6):
        self.api_key = api_key
        self.session = session or requests.Session()
        self.max_retries, self.backoff, self.workers = max_retries, backoff, workers

    # ---------- transport ----------
    def _get(self, search: str = "", count: str | None = None, limit: int | None = None) -> dict:
        url = build_url(search, count, limit, self.api_key)
        for attempt in range(self.max_retries):
            last = attempt == self.max_retries - 1
            try:
                resp = self.session.get(url, timeout=TIMEOUT)
            except requests.RequestException as exc:
                if last:
                    raise OpenFDAError(f"Could not reach openFDA: {exc}") from exc
                time.sleep(self.backoff * 2 ** attempt)
                continue

            if resp.status_code in RETRY_STATUSES and not last:
                time.sleep(self.backoff * 2 ** attempt)  # exponential backoff
                continue
            if resp.status_code == 404:  # openFDA's way of saying "no matches"
                return {"meta": {"results": {"total": 0}}, "results": []}
            if resp.status_code in (401, 403):
                raise OpenFDAError(
                    "openFDA rejected the request: the API key is missing or invalid. "
                    "Get a free key at https://open.fda.gov/apis/authentication/ and set OPENFDA_API_KEY."
                )
            if resp.status_code == 429:
                raise OpenFDAError("openFDA rate limit hit. Wait a minute and try again.")
            if not resp.ok:
                try:
                    msg = resp.json().get("error", {}).get("message", resp.text)
                except ValueError:
                    msg = resp.text
                raise OpenFDAError(f"openFDA error {resp.status_code}: {msg}")
            return resp.json()
        raise OpenFDAError("openFDA request failed after retries.")

    def gather(self, tasks: dict[str, Callable[[], Any]]) -> dict[str, Any]:
        """Run independent API calls concurrently; any exception propagates."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {name: pool.submit(fn) for name, fn in tasks.items()}
            return {name: f.result() for name, f in futures.items()}

    # ---------- queries ----------
    def total(self, search: str = "") -> int:
        data = self._get(search, limit=1)
        return int(data.get("meta", {}).get("results", {}).get("total", 0))

    def totals(self, searches: dict[str, str]) -> dict[str, int]:
        return self.gather({k: (lambda s=s: self.total(s)) for k, s in searches.items()})

    def count(self, search: str, field: str, limit: int = 10) -> pd.DataFrame:
        data = self._get(search, count=field, limit=limit)
        rows = data.get("results", [])
        if not rows:
            return pd.DataFrame(columns=["term", "count"])
        return pd.DataFrame(rows)[["term", "count"]]

    def cases(self, search: str, limit: int = 20) -> list[dict]:
        return self._get(search, limit=limit).get("results", [])

    def monthly_trend(self, search: str) -> pd.DataFrame:
        data = self._get(search, count="receivedate")
        rows = data.get("results", [])
        if not rows:
            return pd.DataFrame(columns=["month", "reports"])
        df = pd.DataFrame(rows)
        key = "time" if "time" in df.columns else "term"
        df["date"] = pd.to_datetime(df[key].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"])
        monthly = df.set_index("date")["count"].resample("MS").sum().reset_index()
        return monthly.rename(columns={"date": "month", "count": "reports"})
