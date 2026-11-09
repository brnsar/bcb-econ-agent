"""Monthly indicator table built from SGS series, stored in SQLite."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean

SERIES = {"usd": 1, "selic": 432, "ipca": 433}

INDICATORS = {
    "usd_avg": "USD/BRL monthly average (BRL per USD)",
    "usd_close": "USD/BRL last quote of the month (BRL per USD)",
    "selic_target": "Copom Selic target at month end (% per year)",
    "ipca_month": "IPCA inflation in the month (%)",
    "ipca_12m": "IPCA accumulated over the last 12 months (%)",
    "real_rate_12m": "Ex-post real rate: Selic target deflated by 12-month IPCA (% per year)",
}

_COLUMNS = ["month", *INDICATORS]


def _month_key(day: date) -> str:
    return f"{day.year:04d}-{day.month:02d}"


def build_monthly_rows(
    usd: list[tuple[date, float]],
    selic: list[tuple[date, float]],
    ipca: list[tuple[date, float]],
) -> list[dict]:
    """Aggregate daily/monthly SGS points into one row per month."""
    usd_by_month: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for day, value in usd:
        usd_by_month[_month_key(day)].append((day, value))
    selic_by_month: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for day, value in selic:
        selic_by_month[_month_key(day)].append((day, value))
    ipca_by_month = {_month_key(day): value for day, value in ipca}

    months = sorted(set(usd_by_month) | set(selic_by_month) | set(ipca_by_month))
    rows: list[dict] = []
    for i, month in enumerate(months):
        row: dict = {"month": month}
        if usd_by_month.get(month):
            points = sorted(usd_by_month[month])
            row["usd_avg"] = round(mean(v for _, v in points), 4)
            row["usd_close"] = round(points[-1][1], 4)
        if selic_by_month.get(month):
            row["selic_target"] = sorted(selic_by_month[month])[-1][1]
        if month in ipca_by_month:
            row["ipca_month"] = ipca_by_month[month]
            window = months[max(0, i - 11) : i + 1]
            if len(window) == 12 and all(m in ipca_by_month for m in window) and _consecutive(window):
                acc = 1.0
                for m in window:
                    acc *= 1 + ipca_by_month[m] / 100
                row["ipca_12m"] = round((acc - 1) * 100, 2)
        if row.get("selic_target") is not None and row.get("ipca_12m") is not None:
            row["real_rate_12m"] = round(((1 + row["selic_target"] / 100) / (1 + row["ipca_12m"] / 100) - 1) * 100, 2)
        rows.append(row)
    return rows


def _consecutive(months: list[str]) -> bool:
    def as_int(m: str) -> int:
        year, mon = m.split("-")
        return int(year) * 12 + int(mon)

    values = [as_int(m) for m in months]
    return all(b - a == 1 for a, b in zip(values, values[1:]))


class IndicatorStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def write(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cols = ", ".join(f"{c} REAL" for c in INDICATORS)
        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS monthly_indicators")
            conn.execute(f"CREATE TABLE monthly_indicators (month TEXT PRIMARY KEY, {cols})")
            conn.executemany(
                f"INSERT INTO monthly_indicators ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)})",
                [tuple(row.get(c) for c in _COLUMNS) for row in rows],
            )

    def exists(self) -> bool:
        return self.path.exists()

    def month_range(self) -> tuple[str, str]:
        with self._connect() as conn:
            row = conn.execute("SELECT MIN(month) AS lo, MAX(month) AS hi FROM monthly_indicators").fetchone()
        return row["lo"], row["hi"]

    def query(self, start_month: str, end_month: str, indicators: list[str]) -> list[dict]:
        unknown = [i for i in indicators if i not in INDICATORS]
        if unknown:
            raise ValueError(f"unknown indicators: {unknown}")
        cols = ", ".join(["month", *indicators])  # safe: validated against a fixed allow-list
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {cols} FROM monthly_indicators WHERE month BETWEEN ? AND ? ORDER BY month",
                (start_month, end_month),
            ).fetchall()
        return [dict(r) for r in rows]

    def value(self, month: str, indicator: str) -> float | None:
        rows = self.query(month, month, [indicator])
        return rows[0][indicator] if rows else None
