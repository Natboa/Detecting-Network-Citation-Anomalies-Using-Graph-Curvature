"""Cited works outside the pulled set: their ids, their text, and the cold-start count.

A pull (`citecurv.data.openalex.pull_works`) stores a record only for the works that match the
slice filter. The works they cite appear in the event table by id alone: no title, no abstract,
no date. `docs/colab_runbook.md` left "embed out-of-slice cited works?" open at S1.3; on the
pilot (2026-11-12) 299,539 of 309,425 graph nodes were such id-only works, and every one of them
with no earlier event got the same TGN embedding. This module supplies what is needed to test
giving them text:

- `cited_only_ids(events, works)`: the cited ids that have no work record.
- `pull_works_by_ids(ids, out_path)`: title, abstract and publication date of those works, fetched
  by id in batches of at most 100 (OpenAlex combines up to 100 values with `|` in one filter and
  needs `per_page=100` to return them all; help.openalex.org, filter-entity-lists, read
  2026-11-12). The result has the columns `citecurv.embed.specter2` reads (work_id, title,
  abstract) plus pub_date.
- `resolve_id_filter(sample_ids)`: the documentation read on 2026-11-12 shows the multi-value
  form for DOIs only, not the attribute name for OpenAlex ids, so the name is found by a probe
  on the live API (two known ids must come back) before a long pull.
- `prior_citation_counts(events)` and `cold_start_summary(events, k)`: how many events the
  cold-start rule (docs/score_components.md section 7: fewer than k earlier citations of the cited
  work, strict ts < t) leaves out, overall and among labelled events.

Resumable pulls. The sorted id list is cut into parts of `ids_per_part`; each finished part is a
parquet file in `<out stem>_parts/` next to the output, with a manifest holding the hash of the
id list. A rerun with the same ids asks only for the parts that are missing; a folder written
for another id list is refused. When every part exists they are merged into `out_path` and the
folder is removed (unless keep_parts).

Leak note: this module acquires text and counts events; it computes no feature. A cited work's
title and abstract are fixed at its publication, which precedes every citation of it, so using
them at event time reads nothing from the future (OpenAlex may still hold a later revision of an
abstract; the same holds for the citing works' text).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

MAX_IDS_PER_REQUEST = 100
ID_FILTER_CANDIDATES = ("openalex", "ids.openalex", "openalex_id")
DEFAULT_SELECT = ("id", "title", "display_name", "abstract_inverted_index", "publication_date")
COLUMNS = ["work_id", "title", "abstract", "pub_date"]
# what="references" (2026-11-12, after the S2.2 pilot sweep): the Forman screen needs the cited work's
# reference count (its out-degree at any time after its publication) and the Ollivier/coupling part
# needs the reference ids; a pull stores neither for works outside the slice.
REFERENCE_SELECT = ("id", "publication_date", "referenced_works_count", "referenced_works")
REFERENCE_COLUMNS = ["work_id", "pub_date", "n_refs", "ref_ids"]


# --------------------------------------------------------------------------- ids and counts

def cited_only_ids(events, works) -> np.ndarray:
    """Sorted unique `dst_id` values of `events` that are not a `work_id` of `works` (str array)."""
    # set difference in Python: np.setdiff1d on pandas 3 string arrays took minutes on the pilot (2026-11-13)
    have = set(works["work_id"].astype(str))
    return np.array(sorted(set(events["dst_id"].astype(str)) - have), dtype=object)


def prior_citation_counts(events) -> np.ndarray:
    """Per event (input row order): events of the same cited work with a strictly earlier t."""
    dst = pd.factorize(events["dst_id"])[0]
    t = events["t"].to_numpy()
    n = len(dst)
    order = np.lexsort((t, dst))
    d, tt = dst[order], t[order]
    pos = np.arange(n)
    new_dst = np.ones(n, dtype=bool)
    new_dst[1:] = d[1:] != d[:-1]
    new_day = new_dst.copy()
    new_day[1:] |= tt[1:] != tt[:-1]
    dst_start = np.maximum.accumulate(np.where(new_dst, pos, 0))
    day_start = np.maximum.accumulate(np.where(new_day, pos, 0))
    out = np.empty(n, dtype=np.int64)
    out[order] = day_start - dst_start
    return out


def cold_start_summary(events, k: int = 3, event_labels=None) -> dict:
    """Events with fewer than k prior citations of the cited work; with labels, the same for positives.

    event_labels: optional frame event_id, label (0/1), e.g. event_labels.parquet.
    """
    cold = prior_citation_counts(events) < int(k)
    out = {"k": int(k), "n_events": int(len(cold)), "n_cold": int(cold.sum()),
           "cold_share": float(cold.mean()) if len(cold) else None}
    if event_labels is not None:
        lab = event_labels.set_index("event_id")["label"].reindex(events["event_id"].to_numpy())
        if lab.isna().any():
            raise ValueError(f"{int(lab.isna().sum())} events have no row in event_labels")
        pos = lab.to_numpy() == 1
        n_pos = int(pos.sum())
        out.update(n_positive=n_pos, n_positive_cold=int((pos & cold).sum()),
                   positive_cold_share=float((pos & cold).sum() / n_pos) if n_pos else None)
    return out


# --------------------------------------------------------------------------- pull by id

def _short_id(url_or_id):
    return url_or_id.rsplit("/", 1)[-1] if url_or_id else None


def _row(w) -> dict:
    from citecurv.data.abstracts import invert_abstract

    return {"work_id": _short_id(w.get("id")),
            "title": w.get("title") or w.get("display_name"),
            "abstract": invert_abstract(w.get("abstract_inverted_index")),
            "pub_date": w.get("publication_date")}


def _reference_row(w) -> dict:
    refs = [_short_id(r) for r in (w.get("referenced_works") or []) if r]
    n = w.get("referenced_works_count")
    return {"work_id": _short_id(w.get("id")), "pub_date": w.get("publication_date"),
            "n_refs": len(refs) if n is None else int(n), "ref_ids": refs}


def _text_stats(df) -> dict:
    has_abstract = df["abstract"].fillna("").astype(str).str.strip().ne("")
    has_title = df["title"].fillna("").astype(str).str.strip().ne("")
    return {"n_with_abstract": int(has_abstract.sum()), "n_no_title": int((~has_title).sum())}


def _reference_stats(df) -> dict:
    n_listed = df["ref_ids"].map(len)
    return {"n_with_refs": int((n_listed > 0).sum()), "n_ref_ids": int(n_listed.sum()),
            "n_count_mismatch": int((n_listed != df["n_refs"].astype("int64")).sum())}


# what -> (default select, output columns, record -> row, frame -> extra stats)
_PROFILES = {"text": (DEFAULT_SELECT, COLUMNS, _row, _text_stats),
             "references": (REFERENCE_SELECT, REFERENCE_COLUMNS, _reference_row, _reference_stats)}


def _part_frame(rows, columns):
    """Part/output frame with stable column types: a list column stays a list of strings even when
    every list in the part is empty (an all-empty list column would otherwise be written as list<null>)."""
    df = pd.DataFrame(rows, columns=columns).astype(object)
    if "ref_ids" in columns:
        import pyarrow as pa

        table = pa.table({"work_id": pa.array(df["work_id"], pa.string()), "pub_date": pa.array(df["pub_date"], pa.string()),
                          "n_refs": pa.array(df["n_refs"], pa.int64()),
                          "ref_ids": pa.array([list(x) for x in df["ref_ids"]], pa.list_(pa.string()))})
        return table
    return df


def _write_frame(frame, path):
    path = Path(path)
    if isinstance(frame, pd.DataFrame):
        _atomic_parquet(frame, path)
        return
    import pyarrow.parquet as pq

    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(frame, tmp)
    tmp.replace(path)


def _id_filter(name: str, value: str) -> dict:
    """pyalex filter kwargs for a possibly dotted attribute: 'ids.openalex' -> {'ids': {'openalex': v}}."""
    out = value
    for key in reversed(name.split(".")):
        out = {key: out}
    return out


def _client_fetch_for(data_cfg, select):
    from citecurv.data.openalex import make_client

    client = make_client(data_cfg)

    def fetch_for(name):
        def fetch(ids):
            return client.list_page("works", _id_filter(name, "|".join(ids)), cursor=None,
                                    per_page=MAX_IDS_PER_REQUEST, select=select)
        return fetch

    return fetch_for, client.rate_limit


def resolve_id_filter(sample_ids, config=None, *, candidates=ID_FILTER_CANDIDATES, select=DEFAULT_SELECT,
                      _fetch_for=None) -> str:
    """The first filter attribute in `candidates` whose request returns exactly `sample_ids`.

    Costs one list call per candidate tried. Raises RuntimeError naming what each candidate gave.
    """
    from citecurv.config import load_config

    sample = [str(i) for i in sample_ids]
    if _fetch_for is None:
        _fetch_for, _ = _client_fetch_for((config or load_config())["data"], select)
    seen = {}
    for name in candidates:
        try:
            got = {_short_id(w.get("id")) for w in (_fetch_for(name)(sample) or {}).get("results") or []}
        except Exception as e:  # an unknown attribute answers with an HTTP error; try the next name
            seen[name] = f"{type(e).__name__}: {e}"
            continue
        if got == set(sample):
            return name
        seen[name] = f"returned {len(got & set(sample))} of {len(sample)} ids"
    raise RuntimeError(f"no id filter returned the sample ids; tried {seen}")


def _ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _atomic_parquet(df, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def pull_works_by_ids(ids, out_path, *, config=None, batch_size: int = MAX_IDS_PER_REQUEST,
                      ids_per_part: int = 5000, id_filter: str = "openalex", select=None, what: str = "text",
                      keep_parts: bool = False, log=None, _fetch=None, _rate_limit=None, _sleep=None, _now=None) -> dict:
    """Fetch work_id, title, abstract, pub_date for `ids` into `out_path` (parquet); returns stats.

    what="references" fetches the reference lists instead: columns work_id, pub_date, n_refs
    (OpenAlex `referenced_works_count`) and ref_ids (short ids of `referenced_works`); its stats
    carry n_with_refs, n_ref_ids and n_count_mismatch (rows whose list length differs from n_refs)
    in place of n_with_abstract and n_no_title. select: fields to ask for (default: by `what`).

    ids: OpenAlex short ids (W...); duplicates are dropped and the list is sorted. batch_size: ids
    per request, at most 100. ids_per_part: ids per resumable part (see the module docstring).
    id_filter: the works filter attribute for an OpenAlex id (see resolve_id_filter). log: optional
    callable that gets one progress line per fetched part (e.g. print).
    Stats: n_ids, n_found, n_missing (ids OpenAlex did not return), n_with_abstract, n_no_title,
    n_other_id (returned records whose id was not asked for in that request, dropped), n_duplicate,
    calls and cost_usd_reported of this call, parts, parts_reused. The same dict is written to
    `<out stem>_stats.json`. Raises ApiKeyMissing without a key and BudgetExhausted on a spent
    daily budget; finished parts stay on disk either way.
    Test seams: _fetch(list of ids) -> response dict, _rate_limit, _sleep, _now.
    """
    from citecurv.data.openalex import _with_backoff

    ids = sorted({str(i) for i in ids})
    if not ids:
        raise ValueError("no ids to fetch")
    if not 1 <= int(batch_size) <= MAX_IDS_PER_REQUEST:
        raise ValueError(f"batch_size must be between 1 and {MAX_IDS_PER_REQUEST}")
    if int(ids_per_part) < 1:
        raise ValueError("ids_per_part must be >= 1")
    batch_size, ids_per_part = int(batch_size), int(ids_per_part)
    if what not in _PROFILES:
        raise ValueError(f"what must be one of {sorted(_PROFILES)}, got {what!r}")
    default_select, columns, make_row, extra_stats = _PROFILES[what]
    select = default_select if select is None else select
    if config is None and _fetch is None:
        from citecurv.config import load_config

        config = load_config()
    data_cfg = (config or {}).get("data", {})
    if _fetch is None:
        fetch_for, _rate_limit = _client_fetch_for(data_cfg, select)
        _fetch = fetch_for(id_filter)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_dir = out_path.parent / f"{out_path.stem}_parts"
    manifest_path = part_dir / "manifest.json"
    manifest = {"ids_sha256": _ids_hash(ids), "n_ids": len(ids), "ids_per_part": ids_per_part}
    if what != "text":  # text manifests keep their earlier form, so part folders written before still resume
        manifest["what"] = what
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"{part_dir} was written for a different id list or part size; "
                             "delete it or use another out_path")
    else:
        part_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    counts = {"calls": 0, "n_other_id": 0, "n_duplicate": 0, "parts_reused": 0}
    cost = None
    part_paths = []
    for p, start in enumerate(range(0, len(ids), ids_per_part)):
        path = part_dir / f"part_{p:05d}.parquet"
        part_paths.append(path)
        if path.exists():
            counts["parts_reused"] += 1
            continue
        chunk = ids[start:start + ids_per_part]
        rows, have = [], set()
        for b in range(0, len(chunk), batch_size):
            batch = chunk[b:b + batch_size]
            resp = _with_backoff(lambda batch=batch: _fetch(batch), data_cfg, rate_limit=_rate_limit,
                                 now=_now, sleep=_sleep) or {}
            counts["calls"] += 1
            c = (resp.get("meta") or {}).get("cost_usd")
            if isinstance(c, (int, float)):
                cost = (cost or 0.0) + float(c)
            asked = set(batch)
            for w in resp.get("results") or []:
                row = make_row(w)
                if row["work_id"] not in asked:
                    counts["n_other_id"] += 1
                elif row["work_id"] in have:
                    counts["n_duplicate"] += 1
                else:
                    have.add(row["work_id"])
                    rows.append(row)
        _write_frame(_part_frame(rows, columns), path)
        if log is not None:
            log(f"part {p + 1} of {-(-len(ids) // ids_per_part)}: {len(rows)} of {len(chunk)} ids found, "
                f"{counts['calls']} calls so far")

    df = pd.concat([pd.read_parquet(p) for p in part_paths], ignore_index=True)
    df = df.sort_values("work_id", kind="mergesort").reset_index(drop=True)[columns]
    _write_frame(_part_frame(df, columns), out_path)
    stats = {"what": what, "n_ids": len(ids), "n_found": int(len(df)), "n_missing": int(len(ids) - len(df)),
             **extra_stats(df), **counts, "cost_usd_reported": cost, "parts": len(part_paths),
             "batch_size": batch_size, "id_filter": id_filter}
    out_path.with_name(f"{out_path.stem}_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    if not keep_parts:
        shutil.rmtree(part_dir, ignore_errors=True)
    return stats


# --------------------------------------------------------------------------- reference lists as context rows

def reference_events(refs, *, exclude_src=None, start_event_id: int = 0):
    """Context rows for the event table from a what="references" frame; returns (frame, stats).

    One row per (work, referenced work): event_id (consecutive from `start_event_id`), src_id = the
    work, dst_id = the referenced work, t = the work's publication day (int days since 1970-01-01,
    the unit of events.t). Appended to the events, these rows give the Forman screen and the
    time-sliced graph the out-edges of cited works that the pull did not store; only the original
    event ids are scored. A row enters a count at time t only if its own t <= t, so a cited work
    whose recorded publication day is later than the citing event contributes nothing to it.

    exclude_src: work ids whose references are already event rows (the pulled works). Works without
    a parseable pub_date are dropped (their rows cannot be placed in time) and counted.
    Stats: n_works, n_excluded, n_no_date, n_without_refs (dated, kept works with an empty list),
    n_rows, n_ref_ids_dropped_no_date.
    """
    df = refs[["work_id", "pub_date", "ref_ids"]].copy()
    n_works = len(df)
    if exclude_src is not None:
        df = df[~df["work_id"].astype(str).isin({str(x) for x in exclude_src})]
    n_excluded = n_works - len(df)
    day = (pd.to_datetime(df["pub_date"], format="%Y-%m-%d", errors="coerce") - pd.Timestamp("1970-01-01")).dt.days
    n_listed = df["ref_ids"].map(len)
    no_date = day.isna()
    stats = {"n_works": int(n_works), "n_excluded": int(n_excluded), "n_no_date": int(no_date.sum()),
             "n_without_refs": int((~no_date & (n_listed == 0)).sum())}
    kept = df[~no_date]
    reps = n_listed[~no_date].to_numpy()
    src = np.repeat(kept["work_id"].astype(str).to_numpy(), reps)
    t = np.repeat(day[~no_date].to_numpy(dtype=np.int64), reps)
    dst = np.array([str(r) for lst in kept["ref_ids"] for r in lst], dtype=object)
    out = pd.DataFrame({"event_id": np.arange(int(start_event_id), int(start_event_id) + len(src), dtype=np.int64),
                        "src_id": src, "dst_id": dst, "t": t})
    stats.update(n_rows=int(len(out)), n_ref_ids_dropped_no_date=int(n_listed[no_date].sum()))
    return out, stats
