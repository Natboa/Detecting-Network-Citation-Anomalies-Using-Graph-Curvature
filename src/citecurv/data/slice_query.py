"""Label-density analysis and slice measurement (task S1.3, decision L11).

Two halves:

1. RWDB side (offline). `prepare_records` turns the Retraction Watch CSV into one
   row per record with parsed subject, country and reason lists, original and
   retraction years, and one boolean column per label family. `rwdb_density`,
   `mass_events`, `cooccurrence` and `candidate_table` count *papers* (distinct
   normalized OriginalPaperDOI values: a paper without a DOI can never join to
   OpenAlex, so it never counts) after a RetractionNature filter (default:
   "Retraction" only; expressions of concern, corrections and reinstatements are
   not retractions, see docs/slice_shortlist.md).

2. OpenAlex side (needs the API key; topics only, concepts are deprecated).
   `lookup_labeled_works` fetches the labeled DOIs once to see which topics
   hold them; `openalex_count_queries` and `label_doi_queries` build the cheap
   grouped/count calls for one candidate slice; `measure_slice` runs them
   through an injectable `pager` (same seam and backoff as the S1.1 harness)
   and checks the S1.3 criteria; `pick_topics` + `topic_candidate` narrow a
   subfield-level candidate to ~300K works; `write_slice_yaml` records the
   chosen filters, expected counts and OpenAlex provenance (corpus, topic
   classifier, measurement date).

Candidate format (plain dict, YAML-friendly):
    name: str
    subjects: [RWDB subject literal, ...]      any-match (optional)
    subject_groups: [RWDB group code, ...]     any-match (optional; union with subjects)
    publishers / journals: [literal, ...]      optional restriction (AND)
    years: [lo, hi]                            original publication years, inclusive
    test_years: [lo, hi]                       candidate test window, inclusive
    openalex: {filter: {...}}                  pyalex-style nested filter, string values
                                               only ("a|b" for OR); publication_year is
                                               added from `years`.
The RWDB selectors and the OpenAlex filter describe the same slice in two
taxonomies; how well they agree is exactly what measure_slice measures.

Leak argument: everything here is slice selection on outcome labels and
corpus-level counts, done once before any model sees data. It produces no
per-event feature, so it cannot leak future graph state into a score. The
choice of slice does use label density by design (L11); the time split
(S4.1) is chosen later and separately.
"""

from datetime import date
from math import ceil
import os
import re

FAMILIES = {
    "ref_attr": ("referencing/attributions",),
    "paper_mill": ("paper mill",),
    "peer_review": ("peer review",),
    "broadened": ("referencing/attributions", "paper mill", "peer review"),
}
DEFAULT_NATURES = ("Retraction",)
LIST_COLUMNS = ("subjects", "subject_groups", "countries", "reasons")

_OA = "https://api.openalex.org"
_IDS_CHECKED = ("subfield IDs and names checked against OpenAlex's published topic hierarchy "
                "(github.com/ourresearch/openalex-topic-classification, v2/changes/subfields.csv, "
                "commit 98df3b96, 2026-10-06); live filter behaviour unverified because "
                "api.openalex.org was unreachable when this was written")


def _verify_subfields(*ids):
    """Documented topics-endpoint calls that echo each subfield's name (one topic each)."""
    return [f"{_OA}/topics?filter=subfield.id:{i}&per-page=1&select=id,display_name,subfield,field"
            for i in ids]


# PROVISIONAL(G3) shortlist from the RWDB 2026-07-06 snapshot (docs/slice_shortlist.md,
# results/rwdb_density_2026-07-06_*). RWDB selectors are exact literals from the CSV.
# OpenAlex side: topic subfields only (concepts are deprecated); each subfield set is a
# starting point that the S1.3 measurement narrows to ~300K works.
SHORTLIST = [
    {
        "name": "computing",
        "description": "Computer science and data science papers",
        "subjects": ["(B/T) Computer Science", "(B/T) Data Science"],
        "years": [2018, 2023], "test_years": [2022, 2023],
        "openalex": {
            "filter": {"primary_topic": {"subfield": {"id": "1702|1705"}}},
            "labels": "1702 Artificial Intelligence | 1705 Computer Networks and Communications",
            "resolve": _verify_subfields(1702, 1705),
            "mapping_status": _IDS_CHECKED,
        },
    },
    {
        "name": "env_energy",
        "description": "Environmental science and energy papers",
        "subjects": ["(ENV) Environmental Sciences", "(PHY) Energy"],
        "years": [2018, 2023], "test_years": [2022, 2023],
        "openalex": {
            "filter": {"primary_topic": {"subfield": {"id": "2310|2105"}}},
            "labels": "2310 Pollution | 2105 Renewable Energy, Sustainability and the Environment",
            "resolve": _verify_subfields(2310, 2105),
            "mapping_status": _IDS_CHECKED,
        },
    },
    {
        "name": "cell_mol_bio",
        "description": "Cell biology, molecular biology, genetics and biochemistry papers",
        "subjects": ["(BLS) Biology - Cellular", "(BLS) Biology - Molecular",
                     "(BLS) Genetics", "(BLS) Biochemistry"],
        "years": [2018, 2023], "test_years": [2022, 2023],
        "openalex": {
            "filter": {"primary_topic": {"subfield": {"id": "1307|1312|1311|1303"}}},
            "labels": "1307 Cell Biology | 1312 Molecular Biology | 1311 Genetics | 1303 Biochemistry",
            "resolve": _verify_subfields(1307, 1312, 1311, 1303),
            "mapping_status": _IDS_CHECKED,
        },
    },
    {
        "name": "oncology",
        "description": "Oncology and cancer biology papers",
        "subjects": ["(HSC) Medicine - Oncology", "(BLS) Biology - Cancer"],
        "years": [2018, 2023], "test_years": [2022, 2023],
        "openalex": {
            "filter": {"primary_topic": {"subfield": {"id": "2730|1306"}}},
            "labels": "2730 Oncology | 1306 Cancer Research",
            "resolve": _verify_subfields(2730, 1306),
            "mapping_status": _IDS_CHECKED,
        },
    },
    {
        "name": "venues_cs",
        "description": "The ten journals with the most Referencing/Attributions retractions "
                       "among computing papers (venue-anchored fallback)",
        "journals": ["Journal of Intelligent & Fuzzy Systems",
                     "Computational Intelligence and Neuroscience", "Soft Computing",
                     "Wireless Communications and Mobile Computing",
                     "Security and Communication Networks", "Journal of Sensors",
                     "Mobile Information Systems", "Mathematical Problems in Engineering",
                     "Complexity", "Scientific Programming"],
        "years": [2018, 2023], "test_years": [2022, 2023],
        "openalex": {
            "needs_resolution": True,
            "filter": {"primary_location": {"source": {"id": "<ten S-ids from the resolve step>"}}},
            "resolve": [f"{_OA}/sources?search=Computational%20Intelligence%20and%20Neuroscience"
                        "&select=id,display_name,issn_l,works_count",
                        "(one /sources?search= call per journal name in `journals`)"],
            "mapping_status": "source IDs not resolved yet (needs the live API)",
        },
    },
]

_GROUP_RE = re.compile(r"^\(([^)]+)\)")
_DATE_FORMATS = ("%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y")
_PLACEHOLDER_BEFORE_YEAR = 1900   # RWDB writes 1/1/1753 (SQL datetime minimum) for "unknown"
_COST_PER_CALL_USD = 0.0001       # same ledger rate as the S1.1 harness (one list page)
_UNSAFE_DOI_CHARS = (",", "|")    # would break OpenAlex filter syntax
_MAX_PER_PAGE = 100               # per_page 200 is deprecated
_MAX_OR_VALUES = 100              # documented limit of '|' values in one filter
CORPUS = "core"                   # default OpenAlex corpus; xpac only with include_xpac=true
TOPIC_ASSIGNMENTS = ("OpenAlex topics, classifier version 2: every work's topics were replaced "
                     "on 2026-10-06; concepts are deprecated and no longer assigned to new works")


# ---------- parsing ----------

def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and x != x)


def _split_list(raw) -> list[str]:
    if _is_missing(raw):
        return []
    parts = (" ".join(p.split()) for p in str(raw).split(";"))
    return list(dict.fromkeys(p for p in parts if p))


def parse_subjects(raw) -> list[str]:
    """'(BLS) Biology - Cellular;(HSC) Medicine - Oncology;' -> two clean subjects.

    Whitespace inside a subject is collapsed (the CSV has '(PHY) Mathematics ' and
    '(PHY) Mathematics'), empty parts dropped, duplicates removed in order.
    """
    return _split_list(raw)


def subject_group(subject: str) -> str | None:
    """'(B/T) Computer Science' -> 'B/T'; None when there is no leading code."""
    m = _GROUP_RE.match(subject or "")
    return m.group(1) if m else None


def parse_rwdb_date(raw):
    """RWDB 'M/D/YYYY H:MM' -> Timestamp; missing, unparseable or placeholder -> NaT."""
    import pandas as pd
    from datetime import datetime

    if _is_missing(raw):
        return pd.NaT
    s = str(raw).strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
        except ValueError:
            continue
        if dt.year < _PLACEHOLDER_BEFORE_YEAR:
            return pd.NaT
        return pd.Timestamp(dt.date())
    return pd.NaT


def prepare_records(raw_df, families=None):
    """One row per RWDB record with parsed columns (blank rows dropped).

    Output columns: record_id, doi_norm, journal, publisher, nature, subjects,
    subject_groups, countries, reasons (lists), orig_date, orig_year, ret_date,
    ret_year (nullable Int64), and fam_<name> (bool) for each family in
    `families` (default FAMILIES). A family matches when any reason contains
    any of its lowercase substrings (same rule as rwdb.load_labels).
    """
    import pandas as pd

    from citecurv.data.doi import normalize_doi
    from citecurv.data.rwdb import parse_reasons

    families = FAMILIES if families is None else families
    df = raw_df[raw_df["Record ID"].notna()]
    out = pd.DataFrame({
        "record_id": df["Record ID"].astype(str).str.strip(),
        "doi_norm": df["OriginalPaperDOI"].map(normalize_doi),
        "journal": df["Journal"].fillna("").str.strip(),
        "publisher": df["Publisher"].fillna("").str.strip(),
        "nature": df["RetractionNature"].fillna("").str.strip(),
        "subjects": df["Subject"].map(parse_subjects),
        "countries": df["Country"].map(_split_list),
        "reasons": df["Reason"].map(parse_reasons),
        "orig_date": df["OriginalPaperDate"].map(parse_rwdb_date),
        "ret_date": df["RetractionDate"].map(parse_rwdb_date),
    }).reset_index(drop=True)
    out["subject_groups"] = out["subjects"].map(
        lambda ss: list(dict.fromkeys(g for g in map(subject_group, ss) if g))
    )
    out["orig_date"] = pd.to_datetime(out["orig_date"])
    out["ret_date"] = pd.to_datetime(out["ret_date"])
    out["orig_year"] = out["orig_date"].dt.year.astype("Int64")
    out["ret_year"] = out["ret_date"].dt.year.astype("Int64")
    lowered = out["reasons"].map(lambda rs: [r.lower() for r in rs])
    for name, subs in families.items():
        out[f"fam_{name}"] = lowered.map(lambda rs, subs=subs: any(s in r for r in rs for s in subs))
    return out


def load_records(csv_path, families=None):
    """Read an RWDB snapshot CSV (schema-checked) and run prepare_records."""
    from citecurv.data.rwdb import _read_csv

    return prepare_records(_read_csv(csv_path), families=families)


# ---------- density tables ----------

def _papers(records, natures):
    """Records with a DOI and an allowed nature (natures=None keeps all)."""
    r = records[records["doi_norm"].notna()]
    if natures is not None:
        r = r[r["nature"].isin(tuple(natures))]
    return r


def _explode(df, keys):
    for k in keys:
        if k in LIST_COLUMNS:
            df = df.explode(k)
            df = df[df[k].notna()]
    return df


def rwdb_density(labels_df, by, families=("ref_attr",), natures=DEFAULT_NATURES):
    """Distinct labeled papers per group, per family.

    labels_df: prepare_records output. by: a column name or a list of them;
    list-valued columns (subjects, subject_groups, countries, reasons) are
    exploded, so a paper counts once in every group it belongs to and column
    totals can exceed the number of papers. Groups are every value seen among
    papers that pass the nature filter (any reason), so zero counts appear.
    Returns columns: <by...>, all_papers, <family>_papers for each family,
    sorted by the first family (descending), then by the keys.
    """
    keys = [by] if isinstance(by, str) else list(by)
    base = _explode(_papers(labels_df, natures), keys)
    grouped = base.groupby(keys, dropna=True)
    out = grouped["doi_norm"].nunique().rename("all_papers").to_frame()
    for fam in families:
        col = f"fam_{fam}"
        out[f"{fam}_papers"] = (
            base[base[col]].groupby(keys)["doi_norm"].nunique()
            .reindex(out.index, fill_value=0).astype(int)
        )
    out = out.reset_index()
    first = f"{families[0]}_papers"
    return out.sort_values([first] + keys, ascending=[False] + [True] * len(keys),
                           kind="mergesort").reset_index(drop=True)


def mass_events(labels_df, family="ref_attr", min_size=50, natures=DEFAULT_NATURES):
    """Same-day retraction batches: family papers grouped by (publisher, ret_date).

    Returns publisher, ret_date, papers, journals (distinct), top_journal for
    every batch with at least `min_size` papers, largest first.
    """
    r = _papers(labels_df, natures)
    r = r[r[f"fam_{family}"] & r["ret_date"].notna()]
    g = r.groupby(["publisher", "ret_date"])
    out = g["doi_norm"].nunique().rename("papers").to_frame()
    out["journals"] = g["journal"].nunique()
    out["top_journal"] = g["journal"].agg(lambda s: s.value_counts().index[0])
    out = out[out["papers"] >= min_size].reset_index()
    return out.sort_values(["papers", "publisher", "ret_date"], ascending=[False, True, True],
                           kind="mergesort").reset_index(drop=True)


def cooccurrence(labels_df, family="ref_attr", natures=DEFAULT_NATURES):
    """Reason literals carried by the family's papers: papers and share per reason."""
    r = _papers(labels_df, natures)
    r = r[r[f"fam_{family}"]]
    n = r["doi_norm"].nunique()
    ex = r.explode("reasons")
    out = ex.groupby("reasons")["doi_norm"].nunique().rename("papers").reset_index()
    out = out.rename(columns={"reasons": "reason"})
    out["share"] = out["papers"] / n if n else 0.0
    return out.sort_values(["papers", "reason"], ascending=[False, True],
                           kind="mergesort").reset_index(drop=True)


def concentration(labels_df, family="ref_attr", natures=DEFAULT_NATURES, event_min=50, top=5) -> dict:
    """How concentrated a family is: top publishers/journals, same-day batches, lag.

    events_ge_min counts distinct papers inside same-day publisher batches of at
    least `event_min` papers. lag_years = retraction date minus original
    publication date, over the family's papers (quantiles 0.5/0.75/0.9).
    """
    r = _papers(labels_df, natures)
    r = r[r[f"fam_{family}"]]
    n = int(r["doi_norm"].nunique())

    def top_list(col, k):
        c = r.groupby(col)["doi_norm"].nunique().reset_index(name="papers")
        c = c.sort_values(["papers", col], ascending=[False, True], kind="mergesort").head(k)
        return [{col: v, "papers": int(p), "share": (p / n) if n else 0.0}
                for v, p in zip(c[col], c["papers"])]

    ev = mass_events(labels_df, family=family, min_size=event_min, natures=natures)
    keys = set(zip(ev["publisher"], ev["ret_date"]))
    in_ev = r[[k in keys for k in zip(r["publisher"], r["ret_date"])]]
    n_ev = int(in_ev["doi_norm"].nunique())
    lag = ((r["ret_date"] - r["orig_date"]).dt.days / 365.25).dropna()
    journals = top_list("journal", 10)
    return {
        "papers": n,
        "top_publishers": top_list("publisher", top),
        "top10_journals_share": sum(j["papers"] for j in journals) / n if n else 0.0,
        "events_ge_min": {"min_size": event_min, "events": int(len(ev)), "papers": n_ev,
                          "share": (n_ev / n) if n else 0.0},
        "largest_event": (
            {"publisher": ev.iloc[0]["publisher"], "ret_date": str(ev.iloc[0]["ret_date"].date()),
             "papers": int(ev.iloc[0]["papers"])} if len(ev) else None
        ),
        "lag_years": ({q: round(float(lag.quantile(float(q))), 2) for q in ("0.5", "0.75", "0.9")}
                      if len(lag) else None),
    }


# ---------- candidates (RWDB side) ----------

def select_records(labels_df, candidate):
    """Records matching a candidate's RWDB selectors and original-year range."""
    subj = set(candidate.get("subjects") or ())
    grps = set(candidate.get("subject_groups") or ())
    r = labels_df
    if subj or grps:
        hit = r["subjects"].map(lambda ss: bool(subj.intersection(ss))) | \
            r["subject_groups"].map(lambda gs: bool(grps.intersection(gs)))
        r = r[hit]
    if candidate.get("publishers"):
        r = r[r["publisher"].isin(candidate["publishers"])]
    if candidate.get("journals"):
        r = r[r["journal"].isin(candidate["journals"])]
    lo, hi = candidate["years"]
    yr = r["orig_year"]
    return r[yr.notna() & (yr >= lo) & (yr <= hi)]


def _top_share(papers_df, keys):
    n = papers_df["doi_norm"].nunique()
    if n == 0:
        return None, 0.0
    counts = papers_df.groupby(keys)["doi_norm"].nunique().sort_values(ascending=False, kind="mergesort")
    return counts.index[0], float(counts.iloc[0] / n)


def candidate_table(labels_df, candidates, families=("ref_attr", "broadened"),
                    natures=DEFAULT_NATURES, min_labeled=300, min_labeled_test=100):
    """One row per candidate: label totals, test-window totals, per-year counts.

    Columns: name, years, test_years, <fam>_papers, <fam>_test_papers for each
    family, <first family>_<year> for every year in the union of candidate
    ranges (NaN outside a candidate's own range), top_publisher,
    top_publisher_share, top_event_share (largest same-day publisher batch),
    n_journals, and passes_label_criteria (first family:
    >= min_labeled papers and >= min_labeled_test in the test window; the S1.3
    thresholds). These are RWDB-side upper bounds: only papers that OpenAlex
    also places inside the slice filter will count after measure_slice.
    """
    import pandas as pd

    primary = families[0]
    all_years = sorted({y for c in candidates for y in range(c["years"][0], c["years"][1] + 1)})
    rows = []
    for c in candidates:
        sel = _papers(select_records(labels_df, c), natures)
        tlo, thi = c["test_years"]
        row = {"name": c["name"], "years": f"{c['years'][0]}-{c['years'][1]}",
               "test_years": f"{tlo}-{thi}"}
        for fam in families:
            f = sel[sel[f"fam_{fam}"]]
            in_test = f[(f["orig_year"] >= tlo) & (f["orig_year"] <= thi)]
            row[f"{fam}_papers"] = int(f["doi_norm"].nunique())
            row[f"{fam}_test_papers"] = int(in_test["doi_norm"].nunique())
        fp = sel[sel[f"fam_{primary}"]]
        per_year = fp.groupby("orig_year")["doi_norm"].nunique()
        for y in all_years:
            in_range = c["years"][0] <= y <= c["years"][1]
            row[f"{primary}_{y}"] = int(per_year.get(y, 0)) if in_range else float("nan")
        row["top_publisher"], row["top_publisher_share"] = _top_share(fp, "publisher")
        _, row["top_event_share"] = _top_share(fp[fp["ret_date"].notna()], ["publisher", "ret_date"])
        row["n_journals"] = int(fp["journal"].nunique())
        row["passes_label_criteria"] = bool(
            row[f"{primary}_papers"] >= min_labeled and row[f"{primary}_test_papers"] >= min_labeled_test
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- OpenAlex queries ----------

def filter_to_string(filters: dict, prefix: str | None = None) -> str:
    """Flatten a pyalex-style nested filter dict into the REST `filter=` string.

    Values must be str/int/bool; OR goes inside the string ("a|b"). Lists are
    rejected on purpose: pyalex joins a plain list with '+' (AND), which would
    silently turn an OR-set into an empty result.
    """
    parts = []
    for k, v in filters.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            parts.append(filter_to_string(v, key))
        elif isinstance(v, (list, tuple, set)):
            raise TypeError(f"filter {key!r}: use an 'a|b' string for OR, not a {type(v).__name__}")
        elif isinstance(v, bool):
            parts.append(f"{key}:{str(v).lower()}")
        else:
            parts.append(f"{key}:{v}")
    return ",".join(parts)


def slice_filter(candidate) -> dict:
    """The candidate's OpenAlex filter plus publication_year from `years` (copy).

    Raises ValueError when a value still holds a '<...>' placeholder (an ID that
    has to be resolved live first, e.g. venue source IDs).
    """
    import copy

    f = copy.deepcopy((candidate.get("openalex") or {}).get("filter") or {})
    lo, hi = candidate["years"]
    f.setdefault("publication_year", f"{lo}-{hi}")
    flat = filter_to_string(f)
    if "<" in flat:
        raise ValueError(f"candidate {candidate.get('name')!r} has an unresolved filter value: {flat}")
    return f


def _query(name, filters, group_by=None, sample=None, seed=None, select=None, per_page=None, **extra):
    params = [f"filter={filter_to_string(filters)}"]
    for k, v in (("group_by", group_by), ("sample", sample), ("seed", seed),
                 ("select", select), ("per_page", per_page)):
        if v is not None:
            params.append(f"{k}={v}")
    q = {"name": name, "endpoint": "works", "filter": filters, "group_by": group_by,
         "sample": sample, "seed": seed, "select": select, "per_page": per_page,
         "url": "https://api.openalex.org/works?" + "&".join(params)}
    q.update(extra)
    return q


def openalex_count_queries(candidate, sample_n=100, seed=0) -> list[dict]:
    """The cheap calls that size one candidate (no DOI lists; see label_doi_queries).

    works_by_year: works per publication year (sum = slice size).
    with_refs_by_year: same, with the documented `has_references:true` filter.
    retracted_oa_by_year: OpenAlex's own is_retracted flag (cross-check only).
    refcount_sample: a seeded random sample (one page, per_page <= 100) with
        `referenced_works` selected; mean length x works = edge estimate (all
        references, as pull_works keeps them).
    works_by_subfield / works_by_topic: the slice split by primary-topic subfield
        and topic, for narrowing to ~300K works (at most 200 groups per page:
        enough for every shortlist subfield set, which hold <= 199 topics).
    Each query: name, endpoint, filter (nested dict), group_by, sample, seed,
    select, per_page, url (human-readable, unencoded). Corpus: OpenAlex core
    (no include_xpac).
    """
    f = slice_filter(candidate)
    return [
        _query("works_by_year", f, group_by="publication_year"),
        _query("with_refs_by_year", {**f, "has_references": True}, group_by="publication_year"),
        _query("retracted_oa_by_year", {**f, "is_retracted": True}, group_by="publication_year"),
        _query("refcount_sample", f, sample=sample_n, seed=seed,
               select="id,publication_year,referenced_works", per_page=min(sample_n, _MAX_PER_PAGE)),
        _query("works_by_subfield", f, group_by="primary_topic.subfield.id"),
        _query("works_by_topic", f, group_by="primary_topic.id"),
    ]


def label_doi_queries(candidate, dois, batch_size=50) -> list[dict]:
    """Count RWDB-labeled DOIs inside the slice, per publication year, in OR-batches.

    DOIs containing ',' or '|' are skipped (they would break the filter syntax);
    each query records n_dois so the caller can report the skipped count.
    batch_size 50 keeps URLs short; the documented OR limit is 100 values.
    """
    if batch_size > _MAX_OR_VALUES:
        raise ValueError(f"batch_size {batch_size} exceeds the OpenAlex OR limit of {_MAX_OR_VALUES}")
    f = slice_filter(candidate)
    safe = [d for d in dict.fromkeys(dois) if d and not any(ch in d for ch in _UNSAFE_DOI_CHARS)]
    out = []
    for i in range(0, len(safe), batch_size):
        batch = safe[i:i + batch_size]
        doi_filter = "|".join(f"https://doi.org/{d}" for d in batch)
        out.append(_query(f"labeled_by_year_{i // batch_size:04d}", {**f, "doi": doi_filter},
                          group_by="publication_year", n_dois=len(batch)))
    return out


def _default_fetch(data_cfg):
    """pyalex-backed pager: query dict -> {'meta', 'group_by' | 'results'}."""
    import pyalex

    from citecurv.data.openalex import ApiKeyMissing

    key = os.environ.get("OPENALEX_API_KEY")
    if not key:
        raise ApiKeyMissing("OPENALEX_API_KEY is not set; refusing a silent keyless call (task S0.4)")
    pyalex.config.api_key = key
    if data_cfg.get("mailto"):
        pyalex.config.email = data_cfg["mailto"]

    def fetch(query):
        q = pyalex.Works().filter(**query["filter"])
        if query.get("group_by"):
            q = q.group_by(query["group_by"])
        if query.get("sample"):
            q = q.sample(query["sample"], seed=query.get("seed"))
        if query.get("select"):
            q = q.select(query["select"])
        resp = q.get(per_page=query.get("per_page"))
        field = "group_by" if query.get("group_by") else "results"
        return {"meta": dict(resp.meta or {}), field: [dict(r) for r in resp]}

    return fetch


def _year_counts(resp) -> dict[int, int]:
    out = {}
    for g in resp.get("group_by") or []:
        try:
            y = int(str(g["key"]).rsplit("/", 1)[-1])
        except ValueError:
            continue
        out[y] = out.get(y, 0) + int(g["count"])
    return dict(sorted(out.items()))


def _n_refs(work) -> int:
    if work.get("referenced_works_count") is not None:
        return int(work["referenced_works_count"])
    return len(work.get("referenced_works") or [])


def _short(x):
    return str(x).rstrip("/").rsplit("/", 1)[-1] if x not in (None, "") else None


def doi_lookup_queries(dois, batch_size=50,
                       select="id,doi,publication_year,primary_topic,primary_location") -> list[dict]:
    """List calls that fetch the OpenAlex records of labeled DOIs (no slice filter).

    One call per batch of `batch_size` DOIs ('|'-joined, at most 100 per the
    OpenAlex docs); per_page 100 leaves room for DOIs that map to two works.
    Used to see, in OpenAlex's own taxonomy, which topics and subfields hold
    the labeled papers before any slice is fixed.
    """
    if batch_size > _MAX_OR_VALUES:
        raise ValueError(f"batch_size {batch_size} exceeds the OpenAlex OR limit of {_MAX_OR_VALUES}")
    safe = [d for d in dict.fromkeys(dois) if d and not any(ch in d for ch in _UNSAFE_DOI_CHARS)]
    out = []
    for i in range(0, len(safe), batch_size):
        batch = safe[i:i + batch_size]
        out.append(_query(f"doi_lookup_{i // batch_size:04d}",
                          {"doi": "|".join(f"https://doi.org/{d}" for d in batch)},
                          select=select, per_page=_MAX_PER_PAGE, n_dois=len(batch)))
    return out


def lookup_labeled_works(dois, pager=None, config=None, batch_size=50):
    """Fetch labeled DOIs from OpenAlex; returns (DataFrame, stats).

    DataFrame columns: work_id, doi_norm, publication_year, topic_id,
    subfield_id, field_id, source_id (short IDs as strings, None when absent).
    stats: dois_queried, works_returned, dois_found (distinct normalized DOIs
    returned), calls. A strict-join found rate over the whole label family
    is dois_found / dois_queried (S1.2 evidence before any pilot pull).
    Same pager seam, key guard and backoff as measure_slice.
    """
    import pandas as pd

    from citecurv.config import load_config
    from citecurv.data.doi import normalize_doi
    from citecurv.data.openalex import _with_backoff

    data_cfg = (config or load_config())["data"]
    fetch = pager if pager is not None else _default_fetch(data_cfg)
    rows, calls, queried = [], 0, 0
    for q in doi_lookup_queries(dois, batch_size=batch_size):
        calls += 1
        queried += q["n_dois"]
        for w in _with_backoff(lambda q=q: fetch(q), data_cfg).get("results") or []:
            topic = w.get("primary_topic") or {}
            source = (w.get("primary_location") or {}).get("source") or {}
            rows.append({
                "work_id": _short(w.get("id")),
                "doi_norm": normalize_doi(w.get("doi")),
                "publication_year": w.get("publication_year"),
                "topic_id": _short(topic.get("id")),
                "subfield_id": _short((topic.get("subfield") or {}).get("id")),
                "field_id": _short((topic.get("field") or {}).get("id")),
                "source_id": _short(source.get("id")),
            })
    cols = ["work_id", "doi_norm", "publication_year", "topic_id", "subfield_id", "field_id", "source_id"]
    df = pd.DataFrame(rows, columns=cols).astype(object)
    df = df.where(df.notna(), None)
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").astype("Int64")
    stats = {"dois_queried": queried, "works_returned": len(df),
             "dois_found": int(df["doi_norm"].dropna().nunique()), "calls": calls}
    return df, stats


def labeled_by_topic(lookup_df, years=None, by="topic_id") -> dict:
    """Distinct labeled DOIs per OpenAlex topic (or subfield/source via `by`).

    lookup_df: lookup_labeled_works output. years: optional [lo, hi] on the
    OpenAlex publication_year. Works without a topic are left out.
    """
    d = lookup_df[lookup_df[by].notna() & lookup_df["doi_norm"].notna()]
    if years is not None:
        y = d["publication_year"]
        d = d[y.notna() & (y >= years[0]) & (y <= years[1])]
    counts = d.groupby(by)["doi_norm"].nunique()
    return {str(k): int(v) for k, v in counts.sort_values(ascending=False, kind="mergesort").items()}


def pick_topics(topic_works: dict, topic_labeled: dict, target_works=300_000, max_works=None) -> dict:
    """Greedy topic set: highest label density first, until target_works is reached.

    Topics are ranked by labeled/works (ties: more labeled papers, then ID);
    topics without labels are never picked. A topic is skipped when adding it
    would pass max_works (default 1.5 x target). Stops once the running total
    reaches target_works. Returns {topics, works, labeled}. The counts are
    corpus-level and label-level only, chosen before any model sees data.
    """
    cap = max_works if max_works is not None else int(target_works * 1.5)
    ranked = sorted(
        (t for t in topic_works if topic_labeled.get(t, 0) > 0 and topic_works[t] > 0),
        key=lambda t: (-topic_labeled[t] / topic_works[t], -topic_labeled[t], t),
    )
    picked, works, labeled = [], 0, 0
    for t in ranked:
        if works >= target_works:
            break
        if works + topic_works[t] > cap:
            continue
        picked.append(t)
        works += topic_works[t]
        labeled += topic_labeled[t]
    return {"topics": picked, "works": works, "labeled": labeled}


def topic_candidate(candidate, topic_ids, name=None) -> dict:
    """Copy of `candidate` whose OpenAlex filter is the primary-topic set `topic_ids`."""
    import copy

    c = copy.deepcopy(candidate)
    c["name"] = name or f"{candidate['name']}_topics"
    c["openalex"] = {"filter": {"primary_topic": {"id": "|".join(topic_ids)}},
                     "narrowed_from": candidate["name"]}
    return c


def measure_slice(candidate, pager=None, dois=None, config=None, batch_size=50,
                  sample_n=100, seed=0, works_target=300_000, works_tol=0.5,
                  min_edges=2_000_000, min_labeled=300, min_labeled_test=100,
                  max_cost_usd=0.5, daily_cap=1_000_000) -> dict:
    """Run the count queries for one candidate and check the S1.3 criteria.

    pager: callable(query dict) -> response dict ({'meta', 'group_by'} or
    {'meta', 'results'}); default = pyalex with OPENALEX_API_KEY (refuses to
    run keyless). Every call goes through the S1.1 retry/backoff wrapper.
    dois: normalized DOIs of labeled papers (e.g. every family DOI with an
    original year near the slice); None skips the labeled counts.
    Returns a dict of measured counts, cost estimates and `criteria` flags:
    works (within works_target x (1 +/- works_tol)), edges (edges_est >=
    min_edges), labeled / labeled_test (>= min_labeled / min_labeled_test;
    False when not measured), budget (pull cost <= max_cost_usd), daily_cap
    (works <= daily_cap results). `passes` = all flags true.
    """
    from citecurv.config import load_config
    from citecurv.data.openalex import _with_backoff

    data_cfg = (config or load_config())["data"]
    fetch = pager if pager is not None else _default_fetch(data_cfg)
    calls = 0

    def run(q):
        nonlocal calls
        calls += 1
        return _with_backoff(lambda: fetch(q), data_cfg)

    resp = {q["name"]: run(q) for q in openalex_count_queries(candidate, sample_n=sample_n, seed=seed)}
    works_by_year = _year_counts(resp["works_by_year"])
    refs_by_year = _year_counts(resp["with_refs_by_year"])
    works_total = sum(works_by_year.values())
    sample = [_n_refs(w) for w in resp["refcount_sample"].get("results") or []]
    mean_refs = sum(sample) / len(sample) if sample else 0.0
    edges_est = int(round(works_total * mean_refs))

    labeled_by_year, n_queried = None, 0
    if dois is not None:
        labeled_by_year = {}
        for q in label_doi_queries(candidate, dois, batch_size=batch_size):
            n_queried += q["n_dois"]
            for y, n in _year_counts(run(q)).items():
                labeled_by_year[y] = labeled_by_year.get(y, 0) + n
        labeled_by_year = dict(sorted(labeled_by_year.items()))
    tlo, thi = candidate["test_years"]
    labeled_total = sum(labeled_by_year.values()) if labeled_by_year is not None else None
    labeled_test = (sum(n for y, n in labeled_by_year.items() if tlo <= y <= thi)
                    if labeled_by_year is not None else None)

    pages = ceil(works_total / data_cfg["page_size"]) if works_total else 0
    pull_cost = pages * _COST_PER_CALL_USD
    criteria = {
        "works": works_target * (1 - works_tol) <= works_total <= works_target * (1 + works_tol),
        "edges": edges_est >= min_edges,
        "labeled": labeled_total is not None and labeled_total >= min_labeled,
        "labeled_test": labeled_test is not None and labeled_test >= min_labeled_test,
        "budget": pull_cost <= max_cost_usd,
        "daily_cap": works_total <= daily_cap,
    }
    return {
        "candidate": candidate["name"],
        "filter_string": filter_to_string(slice_filter(candidate)),
        "measured_on": date.today().isoformat(),
        "works_by_year": works_by_year,
        "works_total": works_total,
        "with_refs_by_year": refs_by_year,
        "with_refs_share": (sum(refs_by_year.values()) / works_total) if works_total else 0.0,
        "retracted_oa_total": sum(_year_counts(resp["retracted_oa_by_year"]).values()),
        "works_by_subfield": {_short(g.get("key")): int(g.get("count", 0))
                              for g in resp["works_by_subfield"].get("group_by") or []},
        "works_by_topic": {_short(g.get("key")): int(g.get("count", 0))
                           for g in resp["works_by_topic"].get("group_by") or []},
        "sample_n": len(sample),
        "mean_refs_sample": mean_refs,
        "edges_est": edges_est,
        "labeled_by_year": labeled_by_year,
        "labeled_total": labeled_total,
        "labeled_test": labeled_test,
        "dois_queried": n_queried,
        "dois_skipped": (len(set(dois)) - n_queried) if dois is not None else 0,
        "pull_pages_est": pages,
        "pull_cost_usd_est": pull_cost,
        "calls": calls,
        "measure_cost_usd_est": calls * _COST_PER_CALL_USD,
        "criteria": criteria,
        "passes": all(criteria.values()),
    }


def write_slice_yaml(path, candidate, measured, rwdb_snapshot=None):
    """Write configs/slice.yaml for the chosen candidate; returns the Path.

    `openalex_filter` is the exact pyalex kwargs dict for pull_works(filters=...);
    `filter_string` is the same filter in REST form. `expected` holds the
    measured counts that S4.0 must reproduce within 20%.
    """
    from pathlib import Path

    import yaml

    m = measured
    body = {"slice": {
        "name": candidate["name"],
        "description": candidate.get("description", ""),
        "openalex_filter": slice_filter(candidate),
        "filter_string": filter_to_string(slice_filter(candidate)),
        "years": list(candidate["years"]),
        "test_years": list(candidate["test_years"]),
        "rwdb_selectors": {k: list(candidate[k]) for k in
                           ("subjects", "subject_groups", "publishers", "journals") if candidate.get(k)},
        "rwdb_snapshot": dict(rwdb_snapshot or {}),
        "expected": {
            "works": m["works_total"],
            "with_refs_share": round(float(m["with_refs_share"]), 4),
            "edges_est": m["edges_est"],
            "labeled_papers": m["labeled_total"],
            "labeled_papers_test": m["labeled_test"],
            "pull_pages": m["pull_pages_est"],
            "pull_cost_usd": round(float(m["pull_cost_usd_est"]), 4),
        },
        "works_by_year": {int(k): int(v) for k, v in m["works_by_year"].items()},
        "criteria": {k: bool(v) for k, v in m["criteria"].items()},
        "openalex": {
            "corpus": CORPUS,
            "topic_assignments": TOPIC_ASSIGNMENTS,
            "measured_on": m.get("measured_on"),
            "pull_date": None,   # filled by the S4.0 manifest; topics can be re-tagged between dates
        },
    }}
    header = (
        "# PROVISIONAL(G3): slice chosen under the Referencing/Attributions label rule (L12).\n"
        "# Written by citecurv.data.slice_query.write_slice_yaml (task S1.3). Do not hand-edit:\n"
        "# re-run the measurement instead. See docs/slice_shortlist.md.\n"
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


# ---------- report ----------

def write_density_tables(records, out_dir, tag, candidates=(),
                         families=("ref_attr", "broadened", "paper_mill", "peer_review"),
                         natures=DEFAULT_NATURES, event_min=50, year_range=(2010, 2025),
                         top_n=40):
    """Write the compact label-density tables for one snapshot; returns the paths.

    Files (prefix rwdb_density_<tag>_): natures (every RetractionNature,
    unfiltered), subjects, subject_groups_year,
    subjects_year (first family, wide), orig_year, ret_year, orig_x_ret (first
    family), publishers, journals, countries (top_n rows each), mass_events,
    cooccurrence, candidates (.csv), and summary.json (definitions,
    concentration, candidate definitions with their OpenAlex count queries).
    Every number in docs/slice_shortlist.md comes from one of these files.
    """
    import json
    from pathlib import Path

    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    primary = families[0]
    lo, hi = year_range
    written = []

    def save(df, name):
        p = out / f"rwdb_density_{tag}_{name}.csv"
        df.to_csv(p, index=False)
        written.append(p)

    def years_only(df, col):
        y = df[col]
        return df[y.notna() & (y >= lo) & (y <= hi)]

    save(rwdb_density(records, "nature", families, natures=None), "natures")
    save(rwdb_density(records, "subjects", families, natures), "subjects")
    save(years_only(rwdb_density(records, ["subject_groups", "orig_year"], families, natures)
                    .sort_values(["subject_groups", "orig_year"]), "orig_year"), "subject_groups_year")
    sy = years_only(rwdb_density(records, ["subjects", "orig_year"], (primary,), natures), "orig_year")
    wide = sy.pivot_table(index="subjects", columns="orig_year", values=f"{primary}_papers",
                          aggfunc="sum", fill_value=0)
    wide.columns = [str(int(c)) for c in wide.columns]
    totals = rwdb_density(records, "subjects", (primary,), natures).set_index("subjects")
    wide.insert(0, f"{primary}_papers_all_years",
                totals[f"{primary}_papers"].reindex(wide.index).fillna(0).astype(int))
    save(wide.sort_values(f"{primary}_papers_all_years", ascending=False, kind="mergesort")
         .reset_index(), "subjects_year")
    save(rwdb_density(records, "orig_year", families, natures).sort_values("orig_year"), "orig_year")
    save(rwdb_density(records, "ret_year", families, natures).sort_values("ret_year"), "ret_year")

    fam_papers = _papers(records, natures)
    fam_papers = years_only(fam_papers[fam_papers[f"fam_{primary}"]], "orig_year")
    xt = (fam_papers.drop_duplicates("doi_norm").groupby(["orig_year", "ret_year"]).size()
          .unstack(fill_value=0))
    xt.columns = [f"ret_{int(c)}" for c in xt.columns]
    save(xt.reset_index(), "orig_x_ret")

    for col, name in (("publisher", "publishers"), ("journal", "journals"), ("countries", "countries")):
        save(rwdb_density(records, col, families, natures).head(top_n), name)
    save(mass_events(records, primary, event_min, natures), "mass_events")
    save(cooccurrence(records, primary, natures), "cooccurrence")
    ct = (candidate_table(records, list(candidates), families=families[:2], natures=natures)
          if candidates else pd.DataFrame())
    save(ct, "candidates")

    rows = json.loads(ct.to_json(orient="records")) if len(ct) else []
    base = _papers(records, natures)
    summary = {
        "tag": tag,
        "natures": list(natures) if natures is not None else None,
        "paper_definition": "distinct normalized OriginalPaperDOI; records without a DOI are excluded",
        "families": {f: list(FAMILIES[f]) for f in families},
        "totals": {f: int(base.loc[base[f"fam_{f}"], "doi_norm"].nunique()) for f in families},
        "concentration": {f: concentration(records, f, natures, event_min) for f in families[:2]},
        "candidates": [
            {**c, "rwdb_counts": row,
             "openalex_queries": [] if (c.get("openalex") or {}).get("needs_resolution") else
             [{k: q[k] for k in ("name", "filter", "group_by", "sample", "seed", "select", "url")}
              for q in openalex_count_queries(c)]}
            for c, row in zip(candidates, rows)
        ],
        "files": [w.name for w in written],
    }
    sp = out / f"rwdb_density_{tag}_summary.json"
    sp.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    written.append(sp)
    return written
