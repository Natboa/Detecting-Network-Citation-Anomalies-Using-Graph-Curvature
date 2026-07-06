"""OpenAlex pull harness (task S1.1).

Pulls work records matching `filters`, writes works.parquet and edges.parquet
per the contracts in IMPLEMENTATION_PLAN.md §3, and returns PullStats.
Edge direction: the pulled work cites each entry of its referenced_works, so
src = pulled work, dst = referenced work, t = the citing work's publication
date in days since 1970-01-01. Works without a publication date are kept in
works.parquet but contribute no edges (NFR8; counted in works_no_date).

The API key is required (env OPENALEX_API_KEY): pulls must never fall back to
keyless mode silently (task S0.4). Keyed list pages cost ~$0.0001 each.
pyalex 0.21 has no built-in 429 handling (upstream issue #100), so every page
fetch goes through _with_backoff.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import os
import time

_EPOCH = date(1970, 1, 1)
_RETRYABLE = {429, 500, 502, 503, 504}
_COST_PER_PAGE_USD = 0.0001

_WORK_COLUMNS = [
    "work_id", "doi_norm", "title", "abstract", "pub_date",
    "concepts", "venue_id", "is_retracted_oa",
]
_EDGE_COLUMNS = ["src_id", "dst_id", "t", "doi_pair_key"]


class ApiKeyMissing(RuntimeError):
    pass


@dataclass
class PullStats:
    works: int = 0
    edges: int = 0
    pages: int = 0
    works_no_date: int = 0
    est_cost_usd: float = 0.0


def _short_id(url_or_id):
    return url_or_id.rsplit("/", 1)[-1] if url_or_id else None


def _days_since_epoch(iso_date):
    if not iso_date:
        return None
    return (date.fromisoformat(iso_date) - _EPOCH).days


def _default_pager(filters, data_cfg):
    import pyalex

    key = os.environ.get("OPENALEX_API_KEY")
    if not key:
        raise ApiKeyMissing(
            "OPENALEX_API_KEY is not set; refusing a silent keyless pull (task S0.4)"
        )
    pyalex.config.api_key = key
    if data_cfg.get("mailto"):
        pyalex.config.email = data_cfg["mailto"]
    return pyalex.Works().filter(**filters).paginate(
        per_page=data_cfg["page_size"], n_max=None
    )


def _with_backoff(fn, data_cfg):
    for attempt in range(data_cfg["max_retries"] + 1):
        try:
            return fn()
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status not in _RETRYABLE or attempt == data_cfg["max_retries"]:
                raise
            time.sleep(data_cfg["backoff_base_s"] * 2 ** attempt)


def _work_row(w):
    source = ((w.get("primary_location") or {}).get("source")) or {}
    from citecurv.data.abstracts import invert_abstract
    from citecurv.data.doi import normalize_doi

    return {
        "work_id": _short_id(w.get("id")),
        "doi_norm": normalize_doi(w.get("doi")),
        "title": w.get("title") or w.get("display_name"),
        "abstract": invert_abstract(w.get("abstract_inverted_index")),
        "pub_date": w.get("publication_date"),
        "concepts": [_short_id(c.get("id")) for c in (w.get("concepts") or [])],
        "venue_id": _short_id(source.get("id")),
        "is_retracted_oa": bool(w.get("is_retracted")),
    }


def pull_works(filters, out_dir, max_works=None, _pager=None, config=None):
    import pandas as pd

    from citecurv.config import load_config

    data_cfg = (config or load_config())["data"]
    pager = _pager if _pager is not None else _default_pager(filters, data_cfg)
    pages = iter(pager)

    stats = PullStats()
    work_rows, edge_rows = [], []
    done = False
    while not done:
        page = _with_backoff(lambda: next(pages, None), data_cfg)
        if page is None:
            break
        stats.pages += 1
        for w in page:
            row = _work_row(w)
            work_rows.append(row)
            stats.works += 1
            t = _days_since_epoch(row["pub_date"])
            if t is None:
                stats.works_no_date += 1
            else:
                for ref in w.get("referenced_works") or []:
                    dst = _short_id(ref)
                    edge_rows.append({
                        "src_id": row["work_id"],
                        "dst_id": dst,
                        "t": t,
                        "doi_pair_key": f"{row['work_id']}|{dst}",
                    })
                    stats.edges += 1
            if max_works is not None and stats.works >= max_works:
                done = True
                break

    stats.est_cost_usd = stats.pages * _COST_PER_PAGE_USD
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(work_rows, columns=_WORK_COLUMNS).to_parquet(
        out / "works.parquet", index=False
    )
    pd.DataFrame(edge_rows, columns=_EDGE_COLUMNS).to_parquet(
        out / "edges.parquet", index=False
    )
    return stats
