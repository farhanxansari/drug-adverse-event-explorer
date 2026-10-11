"""Smoke tests: render the whole Streamlit app against a fake openFDA."""
from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

CASE = {
    "safetyreportid": "1", "receivedate": "20240101", "serious": "1", "occurcountry": "US",
    "primarysource": {"qualification": "5"},
    "patient": {"patientsex": "1", "patientonsetage": "60", "patientonsetageunit": "801",
                "reaction": [{"reactionmeddrapt": "NAUSEA", "reactionoutcome": "1"}],
                "drug": [{"medicinalproduct": "METFORMIN", "drugcharacterization": "1"}]},
}


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status
        self.ok = status < 400
        self.text = ""

    def json(self):
        return self._p


def fake_get(self, url, timeout=None):
    if "count=receivedate" in url:
        rows = [{"time": f"{y}{m:02d}15", "count": 100 + (400 if (y, m) == (2024, 6) else m)}
                for y in (2023, 2024) for m in range(1, 13)]
    elif "count=patient.patientsex" in url:
        rows = [{"term": 1, "count": 400}, {"term": 2, "count": 550}, {"term": 0, "count": 50}]
    elif "count=patient.patientonsetage" in url:
        rows = [{"term": a, "count": 5} for a in range(10, 90, 3)]
    elif "count=patient.reaction.reactionoutcome" in url:
        rows = [{"term": i, "count": 100 - i * 10} for i in range(1, 7)]
    elif "count=occurcountry" in url:
        rows = [{"term": c, "count": n} for c, n in [("US", 600), ("IN", 120), ("GB", 80)]]
    elif "count=primarysource.qualification" in url:
        rows = [{"term": i, "count": 50 * i} for i in range(1, 6)]
    elif "count=patient.drug.openfda.generic_name" in url:
        rows = [{"term": t, "count": n} for t, n in [("METFORMIN", 1000), ("LISINOPRIL", 200), ("INSULIN", 150)]]
    elif "count=patient.drug.drugindication" in url:
        rows = [{"term": "TYPE 2 DIABETES MELLITUS", "count": 700}]
    elif "count=patient.reaction.reactionmeddrapt" in url:
        rows = [{"term": t, "count": n} for t, n in
                [("NAUSEA", 300), ("DIARRHOEA", 250), ("LACTIC ACIDOSIS", 90), ("FATIGUE", 60)]]
    elif "limit=20" in url:
        return FakeResp({"meta": {"results": {"total": 1}}, "results": [CASE]})
    else:  # total queries
        has_drug, has_rx = "generic_name" in url, "reactionmeddrapt" in url
        total = 120 if has_drug and has_rx else 1000 if has_drug else 20_000 if has_rx else 1_000_000
        return FakeResp({"meta": {"results": {"total": total}}, "results": []})
    return FakeResp({"meta": {}, "results": rows})


def test_app_renders_all_tabs(monkeypatch):
    monkeypatch.setenv("OPENFDA_API_KEY", "test")
    with patch.object(requests.Session, "get", fake_get):
        at = AppTest.from_file(APP, default_timeout=90)
        at.run()
        at.sidebar.text_input[1].input("sitagliptin")
        at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 8
    assert at.metric[0].value == "1,000"
    assert any("Signal from" in str(m.value) or "First year" in str(m.value) for m in at.markdown)


def test_app_asks_for_key_when_missing(monkeypatch):
    monkeypatch.delenv("OPENFDA_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("API key" in i.value for i in at.info)
