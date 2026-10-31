"""OpenAlex pull harness (tasks S1.1, S0.4; updated for the 2026-10 API changes).

`pull_works(filters, out_dir, ...)` pulls the work records matching `filters` with cursor paging,
writes works.parquet and edges.parquet per the contracts in IMPLEMENTATION_PLAN.md section 3, and
returns PullStats. Edge direction: the pulled work cites each entry of its referenced_works, so
src = pulled work, dst = referenced work, t = the citing work's publication date in days since
1970-01-01. Works without a publication date are kept in works.parquet but contribute no edges
(NFR8; counted in works_no_date).

API facts this module follows (OpenAlex documentation source, ourresearch/openalex-help on GitHub:
content/api/authentication.md updated 2026-10-15, paging.md 2026-10-14, errors.md 2026-10-27,
deprecations.md 2026-10-02; summarised in docs/literature_refresh_2026-10.md, B1 and B2):

- Key. Required here (env OPENALEX_API_KEY); a pull never falls back to keyless mode silently
  (task S0.4). Cursor paging past the first page needs a key anyway (keyless: 401). The key goes
  either in an `Authorization: Bearer` header (key_mode "bearer", the default; this is what
  pyalex 0.21 does, confirmed in its released api.py) or in the `api_key` query parameter
  (key_mode "param"). The docs say both work identically; `check_key()` verifies on the live API
  that keyed calls are counted against the key. Switch: argument > env OPENALEX_KEY_MODE >
  config `data.api_key_mode` > "bearer". The key never appears in error messages (URLs are
  redacted and transport errors are not chained).
- mailto. Ignored by OpenAlex since the polite pool was replaced by keys (February 2026). If
  `data.mailto` is set it is sent only as a `From` header, which is harmless; it is never put in
  the URL and has no effect on limits.
- Rate. A free key allows 30 requests per second. The client spaces requests by
  1 / `data.max_requests_per_s` (default 10, well inside the limit; null or 0 switches it off).
- per_page. 100 is the supported maximum; 200 is deprecated legacy behaviour. `clamp_per_page`
  enforces <= 100 (with a warning when a larger value is configured).
- 429. Returned both for "too fast" and for "daily budget spent". On a 429 the harness calls
  `GET /rate-limit` (key as api_key parameter, as the docs show for this endpoint): remaining
  budget <= 0 -> BudgetExhausted, carrying the next reset time (from the endpoint, else the next
  midnight UTC) and the resume state; otherwise exponential backoff and retry. If the endpoint
  gives no usable answer, the 429 response's X-RateLimit-Remaining header decides; if neither
  says anything, it is treated as "too fast". The /rate-limit field names are parsed tolerantly
  (`parse_rate_limit`), because the current documentation does not show the response body.
- Cost. Each page's `meta.cost_usd` is summed into PullStats.cost_usd_reported when present;
  est_cost_usd stays pages x $0.0001 (list call price). The first page's meta block is kept in
  the state file, so the field names can be checked after the first live pull.

pyalex 0.21 is still used to turn the filter dict into the query string, so `filters` keeps the
pyalex kwargs format used by configs/slice.yaml and slice_query. Requests are made with
`requests` directly: pyalex's paginator cannot send the key as a parameter, exposes no resumable
cursor and cannot tell the two kinds of 429 apart (its retries treat both alike; upstream issue
#100), and its error messages contain the full URL.

Resumable pulls. With `state_path`, the cursor and counts are written atomically to that JSON
file every time a part is committed: by default every page (`pages_per_part=1`). The page rows
go to `<state stem>_parts/works_NNNNN.parquet` and `edges_NNNNN.parquet` next to the state file
before the state is written. Calling pull_works again with the same filters and state_path
resumes at the saved cursor and, when paging is finished, merges the parts into
`out_dir/works.parquet` and `edges.parquet`, marks the state done and deletes the parts. A part
listed in the state but missing or short on disk (e.g. a Drive upload that never finished) rolls
the pull back to that part's starting cursor. A finished state returns its counts without any
request. A state file written for different filters, page size or select list is refused.
Raising pages_per_part writes fewer, larger parts; after a crash at most pages_per_part - 1
pages are fetched again.

Determinism and limits. Rows come in OpenAlex's cursor order. A resumed pull yields exactly the
rows of an uninterrupted pull as long as OpenAlex does not change the matching records between
the sessions; a pull that spans days can see records added or updated in between, which is why
S4.0 freezes the result with a manifest of SHA256 hashes.

Leak note: this module acquires data and computes no feature, so NFR3 does not apply to it; it
does fix the event time t to the citing work's publication date, which every leak argument
downstream relies on.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote_plus

OPENALEX_URL = "https://api.openalex.org"
MAX_PER_PAGE = 100
KEY_ENV = "OPENALEX_API_KEY"
KEY_MODE_ENV = "OPENALEX_KEY_MODE"
KEY_MODES = ("bearer", "param")
DEFAULT_MAX_REQUESTS_PER_S = 10.0
USER_AGENT = "citecurv (+https://github.com/Natboa/Detecting-Network-Citation-Anomalies-Using-Graph-Curvature)"
STATE_VERSION = 1

_EPOCH = date(1970, 1, 1)
_RETRYABLE = {429, 500, 502, 503, 504}
_COST_PER_PAGE_USD = 0.0001
_RATE_HEADERS = ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Credits-Used", "X-RateLimit-Reset")

_WORK_COLUMNS = [
    "work_id", "doi_norm", "title", "abstract", "pub_date",
    "concepts", "venue_id", "is_retracted_oa",
]
_EDGE_COLUMNS = ["src_id", "dst_id", "t", "doi_pair_key"]


# --------------------------------------------------------------------------- errors and stats

class ApiKeyMissing(RuntimeError):
    pass


class OpenAlexHTTPError(RuntimeError):
    """A non-2xx answer (`status` set) or a transport failure (`status` None, `transient` True).

    `response.status_code` and `response.headers` mirror a requests response, so the retry
    wrapper treats this like any HTTP error. Message and attributes never contain the API key.
    """

    def __init__(self, status, url=None, error=None, headers=None, transient=False):
        self.status = status
        self.url = url
        self.error = error
        self.transient = bool(transient)
        self.response = SimpleNamespace(status_code=status, headers=dict(headers or {}))
        head = f"OpenAlex HTTP {status}" if status is not None else "OpenAlex request failed (transport error)"
        super().__init__("; ".join(str(p) for p in (head, error, url and f"url: {url}") if p))


class BudgetExhausted(RuntimeError):
    """The daily OpenAlex budget is spent (a 429 confirmed by /rate-limit or by the 429 headers).

    reset_at: timezone-aware UTC datetime of the next reset. state: the resume state (cursor,
    counts, parts) as saved; state_path: where it is saved, or None when the pull was not
    resumable. rate_limit: what the 429 check read.
    """

    def __init__(self, reset_at, state=None, state_path=None, rate_limit=None):
        super().__init__()
        self.reset_at = reset_at
        self.state = state
        self.state_path = None if state_path is None else Path(state_path)
        self.rate_limit = rate_limit

    def __str__(self):
        when = self.reset_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        source = (self.rate_limit or {}).get("source", "the API")
        msg = f"OpenAlex daily budget is spent (429 confirmed by {source}); it resets at {when}."
        st = self.state or {}
        if self.state_path is not None:
            return (f"{msg} {st.get('pages', 0)} pages ({st.get('works', 0)} works) are saved. Run the same "
                    f"pull again with state_path='{self.state_path}' after the reset to resume at the saved cursor.")
        return (f"{msg} No state_path was given, so the rows of this call were not saved; pass "
                "state_path=... to make the pull resumable.")


@dataclass
class PullStats:
    works: int = 0
    edges: int = 0
    pages: int = 0
    works_no_date: int = 0
    est_cost_usd: float = 0.0
    cost_usd_reported: float | None = None   # sum of meta.cost_usd over pages that report it
    pages_resumed: int = 0                   # pages taken from a saved state instead of fetched


# --------------------------------------------------------------------------- small helpers

def clamp_per_page(per_page) -> int:
    """OpenAlex page size, at most 100 (per_page=200 is deprecated); warns when clamping."""
    n = int(per_page)
    if n < 1:
        raise ValueError(f"per_page must be >= 1, got {per_page}")
    if n > MAX_PER_PAGE:
        warnings.warn(f"per_page={n} is deprecated by OpenAlex (100 is the supported maximum); using "
                      f"{MAX_PER_PAGE}", UserWarning, stacklevel=2)
        return MAX_PER_PAGE
    return n


def _utcnow():
    return datetime.now(timezone.utc)


def next_midnight_utc(now=None) -> datetime:
    """The next daily budget reset: midnight UTC after `now`."""
    now = (now or _utcnow()).astimezone(timezone.utc)
    return datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(days=1)


def _num(x):
    if isinstance(x, bool) or x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _header(headers, name):
    """Case-insensitive header lookup on a plain dict or a requests header map."""
    if not headers:
        return None
    if name in headers:
        return headers[name]
    low = name.lower()
    for k, v in headers.items():
        if str(k).lower() == low:
            return v
    return None


def _parse_time(x):
    if not isinstance(x, str) or not x:
        return None
    try:
        t = datetime.fromisoformat(x.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _pick_number(d, names, contains):
    for n in names:
        v = _num(d.get(n))
        if v is not None:
            return v
    for k in sorted(d):
        if contains in k and "second" not in k and "reset" not in k:
            v = _num(d[k])
            if v is not None:
                return v
    return None


def parse_rate_limit(payload, now=None) -> dict:
    """Read a `GET /rate-limit` answer into remaining / used / limit / reset_at / exhausted.

    Tolerant of field names: the older OpenAlex documentation shows {"rate_limit": {
    "credits_limit", "credits_used", "credits_remaining", "resets_at", "resets_in_seconds"}};
    USD-named fields (any key containing "remaining", "used" or "limit") are read as well.
    exhausted is True when remaining <= 0, None when no remaining figure was found. reset_at
    comes from resets_at, else now + resets_in_seconds, else the next midnight UTC.
    `fields` lists the keys seen (never the key itself), to check the live schema.
    """
    now = now or _utcnow()
    rl = payload.get("rate_limit") if isinstance(payload, dict) and isinstance(payload.get("rate_limit"), dict) \
        else payload
    rl = rl if isinstance(rl, dict) else {}
    remaining = _pick_number(rl, ("credits_remaining", "remaining_usd", "budget_remaining_usd", "remaining"),
                             "remaining")
    used = _pick_number(rl, ("credits_used", "used_usd", "usage_usd", "used"), "used")
    limit = _pick_number(rl, ("credits_limit", "daily_budget_usd", "limit"), "limit")
    reset_at = _parse_time(rl.get("resets_at"))
    if reset_at is None:
        secs = _num(rl.get("resets_in_seconds"))
        reset_at = now + timedelta(seconds=secs) if secs is not None else next_midnight_utc(now)
    return {"remaining": remaining, "used": used, "limit": limit, "reset_at": reset_at,
            "exhausted": None if remaining is None else remaining <= 0,
            "fields": sorted(k for k in rl if k != "api_key")}


def _classify_429(exc, rate_limit, now) -> dict:
    """Decide what a 429 means: {'exhausted': True|False|None, 'reset_at', 'source', ...}."""
    info = None
    if rate_limit is not None:
        try:
            info = parse_rate_limit(rate_limit(), now=now)
            info["source"] = "GET /rate-limit"
        except Exception as err:  # the check must never hide the original 429
            info = {"exhausted": None, "rate_limit_error": type(err).__name__}
    if info is None or info.get("exhausted") is None:
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        rem = _num(_header(headers, "X-RateLimit-Remaining"))
        if rem is not None:
            reset_s = _num(_header(headers, "X-RateLimit-Reset"))
            reset_at = now + timedelta(seconds=reset_s) if reset_s is not None else next_midnight_utc(now)
            info = {**(info or {}), "remaining": rem, "reset_at": reset_at, "exhausted": rem <= 0,
                    "source": "429 response headers"}
    return info or {"exhausted": None}


def _with_backoff(fn, data_cfg, *, rate_limit=None, now=None, sleep=None):
    """Call fn() with retries: exponential backoff on 429/5xx and on transient transport errors.

    A 429 is first classified (see _classify_429): a spent daily budget raises BudgetExhausted
    at once (its state is filled in by the caller), anything else backs off. Without
    `rate_limit` the 429 headers are the only evidence. Other errors propagate immediately.
    Knobs: data.max_retries, data.backoff_base_s (sleep base * 2**attempt).
    """
    sleep = sleep or time.sleep
    now = now or _utcnow
    retries = int(data_cfg.get("max_retries", 5))
    base = float(data_cfg.get("backoff_base_s", 2.0))
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429:
                verdict = _classify_429(e, rate_limit, now())
                if verdict.get("exhausted"):
                    raise BudgetExhausted(verdict["reset_at"], rate_limit=verdict) from e
            elif status not in _RETRYABLE and not getattr(e, "transient", False):
                raise
            if attempt == retries:
                raise
            sleep(base * 2 ** attempt)


# --------------------------------------------------------------------------- HTTP client

def _filter_query(entity, filters, select=None) -> str:
    """Encoded `filter=...&select=...` query string, built by pyalex from its kwargs format."""
    import pyalex

    q = getattr(pyalex, entity.capitalize())()
    if filters:
        q = q.filter(**filters)
    if select:
        q = q.select(list(select))
    url = q.url
    return url.split("?", 1)[1] if "?" in url else ""


def _transport_errors():
    try:
        import requests

        return (requests.RequestException, OSError)
    except ImportError:  # pragma: no cover - requests is in every data-stage env
        return (OSError,)


class OpenAlexClient:
    """Minimal keyed OpenAlex client: throttled GETs, key as Bearer header or api_key parameter.

    `session` needs only `.get(url, headers=..., timeout=...)` (a requests.Session by default);
    `clock`/`sleep` drive the throttle. Methods return parsed JSON; non-2xx answers raise
    OpenAlexHTTPError with the key redacted.
    """

    def __init__(self, api_key, key_mode="bearer", *, mailto=None,
                 max_requests_per_s=DEFAULT_MAX_REQUESTS_PER_S, timeout_s=60.0, base_url=OPENALEX_URL,
                 session=None, clock=None, sleep=None):
        if not api_key:
            raise ApiKeyMissing(f"{KEY_ENV} is not set; refusing a silent keyless call (task S0.4)")
        if key_mode not in KEY_MODES:
            raise ValueError(f"key_mode must be one of {KEY_MODES}, got {key_mode!r}")
        self._key = str(api_key)
        self.key_mode = key_mode
        self.mailto = mailto
        self.min_interval_s = 1.0 / float(max_requests_per_s) if max_requests_per_s else 0.0
        self.timeout_s = timeout_s
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._last = None

    def __repr__(self):
        return f"OpenAlexClient(key=...{self.key_suffix}, key_mode={self.key_mode!r})"

    @property
    def key_suffix(self) -> str:
        return self._key[-4:] if len(self._key) > 8 else "?"

    @property
    def session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _throttle(self):
        if self.min_interval_s <= 0:
            return
        now = self._clock()
        if self._last is not None:
            wait = self.min_interval_s - (now - self._last)
            if wait > 0:
                self._sleep(wait)
                now += wait
        self._last = now

    def _redact(self, text):
        return str(text).replace(quote_plus(self._key), "***").replace(self._key, "***")

    def get(self, path, query="", *, key_in_url=None):
        """GET base_url/path?query -> (json, headers). `query` is an already encoded string."""
        url = f"{self.base_url}/{path.lstrip('/')}" + (f"?{query}" if query else "")
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.mailto:
            headers["From"] = self.mailto   # ignored by OpenAlex since 2026-02; harmless
        if (self.key_mode == "param") if key_in_url is None else key_in_url:
            url += ("&" if "?" in url else "?") + "api_key=" + quote_plus(self._key)
        else:
            headers["Authorization"] = f"Bearer {self._key}"
        shown = self._redact(url)
        self._throttle()
        try:
            resp = self.session.get(url, headers=headers, timeout=self.timeout_s)
        except _transport_errors() as e:
            # not chained: requests puts the full URL (and so a parameter key) in its messages
            raise OpenAlexHTTPError(None, shown, error=type(e).__name__, transient=True) from None
        if not 200 <= resp.status_code < 300:
            raise OpenAlexHTTPError(resp.status_code, shown, error=self._error_text(resp),
                                    headers=dict(resp.headers or {}))
        return resp.json(), dict(resp.headers or {})

    def _error_text(self, resp):
        try:
            body = resp.json()
        except Exception:
            body = None
        text = None
        if isinstance(body, dict):
            text = " - ".join(str(body[k]) for k in ("error", "message") if body.get(k))
        hints = {401: "a valid key is needed (cursor paging past the first page always needs one)",
                 403: "the key may not be activated yet (click the emailed activation link)"}
        if resp.status_code in hints:
            text = f"{text} ({hints[resp.status_code]})" if text else hints[resp.status_code]
        return self._redact(text) if text else None

    def list_page_raw(self, entity, filters, cursor="*", per_page=MAX_PER_PAGE, select=None):
        """One list page -> (json, headers). cursor None sends no cursor (basic first page)."""
        parts = [p for p in (_filter_query(entity, filters, select),) if p]
        parts.append(f"per-page={clamp_per_page(per_page)}")
        if cursor is not None:
            parts.append(f"cursor={quote_plus(str(cursor))}")
        return self.get(entity, "&".join(parts))

    def list_page(self, entity, filters, cursor="*", per_page=MAX_PER_PAGE, select=None) -> dict:
        return self.list_page_raw(entity, filters, cursor=cursor, per_page=per_page, select=select)[0]

    def rate_limit(self) -> dict:
        """GET /rate-limit (key as api_key parameter, the form the docs give for this endpoint)."""
        return self.get("rate-limit", key_in_url=True)[0]


def make_client(data_cfg=None, *, key=None, key_mode=None, session=None) -> OpenAlexClient:
    """Client from the `data` config block and the environment; refuses to run keyless.

    Key: `key` or env OPENALEX_API_KEY. Mode: `key_mode` > env OPENALEX_KEY_MODE >
    data.api_key_mode > "bearer". Throttle: data.max_requests_per_s (default 10; null/0 = off).
    """
    data_cfg = data_cfg or {}
    key = key or os.environ.get(KEY_ENV)
    if not key:
        raise ApiKeyMissing(f"{KEY_ENV} is not set; refusing a silent keyless pull (task S0.4)")
    mode = str(key_mode or os.environ.get(KEY_MODE_ENV) or data_cfg.get("api_key_mode") or "bearer").strip().lower()
    if mode not in KEY_MODES:
        raise ValueError(f"OpenAlex key mode must be one of {KEY_MODES}, got {mode!r}")
    return OpenAlexClient(key, mode, mailto=data_cfg.get("mailto"),
                          max_requests_per_s=data_cfg.get("max_requests_per_s", DEFAULT_MAX_REQUESTS_PER_S),
                          timeout_s=data_cfg.get("timeout_s", 60.0), session=session)


def _usage_delta(a, b):
    if a.get("used") is not None and b.get("used") is not None:
        return b["used"] - a["used"]
    if a.get("remaining") is not None and b.get("remaining") is not None:
        return a["remaining"] - b["remaining"]
    return None


def _jsonable_rl(rl):
    out = dict(rl)
    if isinstance(out.get("reset_at"), datetime):
        out["reset_at"] = out["reset_at"].isoformat()
    return out


def check_key(config=None, *, key=None, key_mode=None, probe_filters=None, _client=None, _now=None) -> dict:
    """S0.4 verify: is the key accepted, and are keyed list calls counted against it?

    Calls GET /rate-limit twice (the difference is what a /rate-limit call itself adds), makes
    one keyed list call (per_page=1, about $0.0001) in the configured key mode, then calls
    /rate-limit again. counted_against_key = the probe raised the key's usage by more than the
    baseline. If usage is booked with a delay this reads False: wait a minute and run again.
    Returns a JSON-ready dict (never the key itself); `ok` = probe answered 200 and counted.
    """
    if _client is None:
        if config is None:
            from citecurv.config import load_config

            config = load_config()
        _client = make_client(config.get("data", {}), key=key, key_mode=key_mode)
    now = _now or _utcnow
    rl0 = parse_rate_limit(_client.rate_limit(), now=now())
    rl1 = parse_rate_limit(_client.rate_limit(), now=now())
    try:
        body, headers = _client.list_page_raw("works", probe_filters or {"publication_year": "2020"},
                                              cursor=None, per_page=1)
        status, error = 200, None
    except OpenAlexHTTPError as e:
        body, headers, status, error = {}, e.response.headers, e.status, str(e)
    rl2 = parse_rate_limit(_client.rate_limit(), now=now())
    baseline, delta = _usage_delta(rl0, rl1), _usage_delta(rl1, rl2)
    counted = None if baseline is None or delta is None else delta > baseline
    if status != 200:
        verdict = f"FAIL: the keyed probe call was refused ({error})"
    elif counted is None:
        verdict = "UNKNOWN: /rate-limit reported no usage figure; compare rate_limit_before/after by hand"
    elif counted:
        verdict = f"PASS: the probe call was counted against the key (key mode {_client.key_mode})"
    else:
        verdict = (f"FAIL: the probe call was not counted against the key (key mode {_client.key_mode}); "
                   "try the other key mode (OPENALEX_KEY_MODE=param or bearer), or rerun in a minute")
    return {
        "ok": status == 200 and counted is True,
        "verdict": verdict,
        "key_mode": _client.key_mode,
        "key_suffix": _client.key_suffix,
        "counted_against_key": counted,
        "baseline_delta": baseline,
        "probe_delta": delta,
        "probe_status": status,
        "probe_error": error,
        "probe_meta": {k: v for k, v in ((body or {}).get("meta") or {}).items() if k != "next_cursor"},
        "probe_rate_headers": {h: v for h in _RATE_HEADERS if (v := _header(headers, h)) is not None},
        "rate_limit_before": _jsonable_rl(rl1),
        "rate_limit_after": _jsonable_rl(rl2),
    }


# --------------------------------------------------------------------------- rows

def _short_id(url_or_id):
    return url_or_id.rsplit("/", 1)[-1] if url_or_id else None


def _days_since_epoch(iso_date):
    if not iso_date:
        return None
    return (date.fromisoformat(iso_date) - _EPOCH).days


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


class _Pending:
    """Rows of pages fetched since the last commit."""

    def __init__(self, cursor_in):
        self.cursor_in = cursor_in
        self.pages = 0
        self.works, self.edges = [], []
        self.no_date = 0
        self.cost = None

    def add(self, results, works_left, cost):
        self.pages += 1
        c = _num(cost)
        if c is not None:
            self.cost = (self.cost or 0.0) + c
        for w in results:
            if works_left is not None and works_left <= 0:
                break
            row = _work_row(w)
            self.works.append(row)
            if works_left is not None:
                works_left -= 1
            t = _days_since_epoch(row["pub_date"])
            if t is None:
                self.no_date += 1
                continue
            for ref in w.get("referenced_works") or []:
                dst = _short_id(ref)
                self.edges.append({"src_id": row["work_id"], "dst_id": dst, "t": t,
                                   "doi_pair_key": f"{row['work_id']}|{dst}"})


def _frames(work_rows, edge_rows):
    import pandas as pd

    works = pd.DataFrame(work_rows, columns=_WORK_COLUMNS)
    edges = pd.DataFrame(edge_rows, columns=_EDGE_COLUMNS)
    return _typed(works, edges)


def _typed(works, edges):
    works = works.astype({"is_retracted_oa": bool})
    edges = edges.astype({"t": "int64"})
    return works, edges


def _atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_parquet(df, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _write_outputs(out, works, edges):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(works, out / "works.parquet")
    _atomic_parquet(edges, out / "edges.parquet")


# --------------------------------------------------------------------------- resumable state

def _canonical(query) -> str:
    return json.dumps(query, sort_keys=True, default=str)


def _recount(st):
    parts = st["parts"]
    for k in ("pages", "works", "edges", "works_no_date"):
        st[k] = sum(p[k] for p in parts)
    costs = [p["cost_usd"] for p in parts if p.get("cost_usd") is not None]
    st["cost_usd_reported"] = sum(costs) if costs else None
    st["est_cost_usd"] = st["pages"] * _COST_PER_PAGE_USD


class _PartStore:
    """State file + its part folder (`<stem>_parts/` next to the state file)."""

    def __init__(self, state_path):
        self.state_path = Path(state_path)
        self.dir = self.state_path.parent / f"{self.state_path.stem}_parts"

    def paths(self, i):
        return self.dir / f"works_{i:05d}.parquet", self.dir / f"edges_{i:05d}.parquet"

    def save(self, st):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.state_path, json.dumps(st, indent=1))

    def write_part(self, i, work_rows, edge_rows):
        self.dir.mkdir(parents=True, exist_ok=True)
        works, edges = _frames(work_rows, edge_rows)
        wp, ep = self.paths(i)
        _atomic_parquet(works, wp)
        _atomic_parquet(edges, ep)

    def part_ok(self, part) -> bool:
        import pyarrow.parquet as pq

        wp, ep = self.paths(part["index"])
        try:
            return (pq.read_metadata(wp).num_rows == part["works"]
                    and pq.read_metadata(ep).num_rows == part["edges"])
        except (OSError, ValueError):
            return False

    def load(self, query_key):
        """Saved state or None; refuses a state of another query; rolls back missing parts."""
        if not self.state_path.exists():
            return None
        st = json.loads(self.state_path.read_text(encoding="utf-8"))
        if st.get("query_key") != query_key:
            raise ValueError(f"{self.state_path} belongs to a different query (filters, page size or select); "
                             "use another state_path or delete that file")
        if st.get("done"):
            return st
        for i, part in enumerate(st["parts"]):
            if not self.part_ok(part):
                warnings.warn(f"{self.state_path.name}: part {i} is missing or incomplete on disk; resuming from "
                              "its starting cursor (later pages are fetched again)", UserWarning, stacklevel=3)
                st["parts"] = st["parts"][:i]
                st["cursor"] = part["cursor_in"]
                st["paging_complete"] = False
                _recount(st)
                break
        return st

    def read_parts(self, parts):
        import pandas as pd

        if not parts:
            return _frames([], [])
        works = pd.concat([pd.read_parquet(self.paths(p["index"])[0]) for p in parts], ignore_index=True)
        edges = pd.concat([pd.read_parquet(self.paths(p["index"])[1]) for p in parts], ignore_index=True)
        return _typed(works, edges)

    def remove_parts(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def _new_state(query_key, now):
    stamp = now().isoformat()
    st = {"version": STATE_VERSION, "query": json.loads(query_key), "query_key": query_key,
          "cursor": "*", "paging_complete": False, "done": False,
          "pages": 0, "works": 0, "edges": 0, "works_no_date": 0,
          "cost_usd_reported": None, "est_cost_usd": 0.0,
          "parts": [], "first_meta": None, "created_at": stamp, "updated_at": stamp}
    return st


def _stats(st, pages_resumed):
    return PullStats(works=st["works"], edges=st["edges"], pages=st["pages"], works_no_date=st["works_no_date"],
                     est_cost_usd=st["pages"] * _COST_PER_PAGE_USD, cost_usd_reported=st["cost_usd_reported"],
                     pages_resumed=pages_resumed)


# --------------------------------------------------------------------------- pulls

def pull_works(filters, out_dir, max_works=None, _pager=None, config=None, *, state_path=None,
               pages_per_part=None, keep_parts=False, select=None,
               _fetch=None, _rate_limit=None, _now=None, _sleep=None) -> PullStats:
    """Pull works matching `filters` (pyalex kwargs) into out_dir/works.parquet + edges.parquet.

    max_works: stop after this many works (None = all). config: full config dict (default:
    configs/default.yaml); reads data.page_size (clamped to 100), max_retries, backoff_base_s,
    api_key_mode, max_requests_per_s, timeout_s, mailto. state_path: JSON file that makes the
    pull resumable (see the module docstring); without it nothing is saved until the end.
    pages_per_part: pages per committed part, state written after each (default
    data.pages_per_part, else 1 = after every page). keep_parts: keep the
    part files after the merge. select: optional OpenAlex `select` field list.
    Raises ApiKeyMissing before any request when no key is set, and BudgetExhausted (with the
    reset time and the saved state) when a 429 turns out to be a spent daily budget.
    Test seams: _pager (legacy iterable of pages, no cursor, no state), _fetch(cursor, per_page)
    -> response dict, _rate_limit() -> /rate-limit payload, _now() -> aware datetime, _sleep(s).
    """
    from citecurv.config import load_config

    data_cfg = (config or load_config())["data"]
    out = Path(out_dir)
    if _pager is not None:
        if state_path is not None:
            raise ValueError("state_path needs the cursor-aware fetcher; the legacy _pager has no cursor to save")
        return _pull_pages(_pager, out, max_works, data_cfg)

    per_page = clamp_per_page(data_cfg.get("page_size", MAX_PER_PAGE))
    if _fetch is None:
        client = make_client(data_cfg)

        def fetch(cursor, n):
            return client.list_page("works", filters, cursor=cursor, per_page=n, select=select)

        rate_limit = client.rate_limit
    else:
        fetch, rate_limit = _fetch, _rate_limit
    if pages_per_part is None:
        pages_per_part = data_cfg.get("pages_per_part") or 1
    if int(pages_per_part) < 1:
        raise ValueError("pages_per_part must be >= 1")
    return _pull_cursor(filters, out, max_works, data_cfg, per_page, fetch, rate_limit, select=select,
                        state_path=state_path, pages_per_part=int(pages_per_part), keep_parts=keep_parts,
                        now=_now or _utcnow, sleep=_sleep)


def _pull_pages(pager, out, max_works, data_cfg):
    """Legacy path: iterate pre-built pages (lists of work dicts), retrying failed fetches."""
    pages = iter(pager)
    acc = _Pending(None)
    while True:
        page = _with_backoff(lambda: next(pages, None), data_cfg)
        if page is None:
            break
        meta = getattr(page, "meta", None)
        acc.add(list(page), None if max_works is None else max_works - len(acc.works),
                meta.get("cost_usd") if isinstance(meta, dict) else None)
        if max_works is not None and len(acc.works) >= max_works:
            break
    _write_outputs(out, *_frames(acc.works, acc.edges))
    return PullStats(works=len(acc.works), edges=len(acc.edges), pages=acc.pages, works_no_date=acc.no_date,
                     est_cost_usd=acc.pages * _COST_PER_PAGE_USD, cost_usd_reported=acc.cost)


def _pull_cursor(filters, out, max_works, data_cfg, per_page, fetch, rate_limit, *, select, state_path,
                 pages_per_part, keep_parts, now, sleep):
    query = {"entity": "works", "filters": filters, "per_page": per_page, "select": select}
    query_key = _canonical(query)
    store = _PartStore(state_path) if state_path is not None else None
    st = store.load(query_key) if store else None
    if st is None:
        st = _new_state(query_key, now)
    pages_resumed = st["pages"]

    if st["done"]:
        if (out / "works.parquet").exists() and (out / "edges.parquet").exists():
            return _stats(st, pages_resumed)
        raise RuntimeError(f"{state_path} says the pull is finished but {out} has no works/edges parquet; "
                           "delete the state file to pull again")

    memory = []           # parts of a non-resumable pull: (work_rows, edge_rows)
    cursor = st["cursor"]
    pend = _Pending(cursor)

    def commit(complete):
        nonlocal pend
        if pend.pages:
            part = {"index": len(st["parts"]), "cursor_in": pend.cursor_in, "pages": pend.pages,
                    "works": len(pend.works), "edges": len(pend.edges), "works_no_date": pend.no_date,
                    "cost_usd": pend.cost}
            if store:
                store.write_part(part["index"], pend.works, pend.edges)
            else:
                memory.append((pend.works, pend.edges))
            st["parts"].append(part)
        st["cursor"] = None if complete else cursor
        st["paging_complete"] = complete
        _recount(st)
        st["updated_at"] = now().isoformat()
        if store:
            store.save(st)
        pend = _Pending(st["cursor"])

    while not st["paging_complete"]:
        try:
            resp = _with_backoff(lambda c=cursor: fetch(c, per_page), data_cfg, rate_limit=rate_limit, now=now,
                                 sleep=sleep)
        except BaseException as e:
            if pend.pages:
                commit(False)        # keep what was fetched; `cursor` is the page that failed
            if isinstance(e, BudgetExhausted):
                e.state, e.state_path = copy.deepcopy(st), (None if store is None else store.state_path)
            raise
        resp = resp or {}
        results = list(resp.get("results") or [])
        meta = resp.get("meta") or {}
        if st["first_meta"] is None:
            st["first_meta"] = {k: v for k, v in meta.items() if k != "next_cursor"}
        have = st["works"] + len(pend.works)
        pend.add(results, None if max_works is None else max_works - have, meta.get("cost_usd"))
        cursor = meta.get("next_cursor")
        complete = (not results or cursor is None
                    or (max_works is not None and st["works"] + len(pend.works) >= max_works))
        if complete or pend.pages >= pages_per_part:
            commit(complete)

    if store:
        works, edges = store.read_parts(st["parts"])
    else:
        works, edges = _frames([r for w, _ in memory for r in w], [r for _, e in memory for r in e])
    _write_outputs(out, works, edges)
    st["done"] = True
    st["updated_at"] = now().isoformat()
    if store:
        store.save(st)
        if not keep_parts:
            store.remove_parts()
    return _stats(st, pages_resumed)
