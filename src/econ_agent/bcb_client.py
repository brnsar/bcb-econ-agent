"""Thin client for the public Banco Central do Brasil APIs (Copom documents and SGS series)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date

COPOM_API = "https://www.bcb.gov.br/api/servico/sitebcb/copom"
SGS_API = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
USER_AGENT = "bcb-econ-agent/0.1 (+https://github.com/brnsar/bcb-econ-agent)"


@dataclass(frozen=True)
class CopomDocument:
    meeting: int
    doc_type: str  # "statement" | "minutes"
    date: str  # ISO date of the meeting
    title: str
    html: str
    url: str


def _get_json(url: str, retries: int = 3, timeout: float = 30.0):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def list_statement_meetings(quantity: int = 40) -> list[int]:
    data = _get_json(f"{COPOM_API}/comunicados?quantidade={quantity}")
    return [int(item["nro_reuniao"]) for item in data["conteudo"]]


def list_minutes_meetings(quantity: int = 40) -> list[int]:
    data = _get_json(f"{COPOM_API}/atas?quantidade={quantity}")
    return [int(item["nroReuniao"]) for item in data["conteudo"]]


def get_statement(meeting: int) -> CopomDocument:
    url = f"{COPOM_API}/comunicados_detalhes?nro_reuniao={meeting}"
    item = _get_json(url)["conteudo"][0]
    return CopomDocument(
        meeting=meeting,
        doc_type="statement",
        date=item["dataReferencia"],
        title=item["titulo"],
        html=item["textoComunicado"],
        url=url,
    )


def get_minutes(meeting: int) -> CopomDocument:
    url = f"{COPOM_API}/atas_detalhes?nro_reuniao={meeting}"
    item = _get_json(url)["conteudo"][0]
    return CopomDocument(
        meeting=meeting,
        doc_type="minutes",
        date=item["dataReferencia"],
        title=item["titulo"],
        html=item["textoAta"],
        url=item.get("urlPdfAta") or url,
    )


def get_sgs_series(code: int, start: date, end: date) -> list[tuple[date, float]]:
    """Fetch an SGS series. Daily series are limited to 10-year windows, so the range is split."""
    points: list[tuple[date, float]] = []
    window_start = start
    while window_start <= end:
        window_end = min(date(window_start.year + 9, 12, 31), end)
        query = urllib.parse.urlencode(
            {
                "formato": "json",
                "dataInicial": window_start.strftime("%d/%m/%Y"),
                "dataFinal": window_end.strftime("%d/%m/%Y"),
            }
        )
        for row in _get_json(f"{SGS_API.format(code=code)}?{query}"):
            day, month, year = (int(x) for x in row["data"].split("/"))
            points.append((date(year, month, day), float(row["valor"])))
        window_start = date(window_end.year + 1, 1, 1)
    return points
