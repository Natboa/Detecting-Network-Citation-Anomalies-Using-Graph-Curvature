"""S1.1: pull harness — pagination, retry/backoff, cost ledger, parquet contracts.

Spec under test (S1.1 plus the 2026-10 OpenAlex changes, docs/literature_refresh_2026-10.md B1):
- legacy page iterator (_pager): pagination, 429/5xx retry, cost ledger, contracted parquet;
- per_page is clamped to 100 (per_page=200 is deprecated);
- a 429 is ambiguous: GET /rate-limit decides between "budget spent" (BudgetExhausted
  carrying the next reset at midnight UTC and the resume state) and "too fast" (back off and
  retry); the 429 response headers are the fallback when the endpoint gives no answer;
- resumable cursor state: after every committed page the cursor and counts are written
  atomically; a resumed pull yields exactly the rows of an uninterrupted pull;
- the key is never optional and never leaks into error messages; it travels as a Bearer
  header or an api_key parameter (config/env switch); check_key() tells whether calls are
  counted against the key.

No network anywhere: pages come from the _pager / _fetch seams, HTTP from a fake session.
"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from citecurv.data.openalex import (
    ApiKeyMissing,
    BudgetExhausted,
    OpenAlexClient,
    OpenAlexHTTPError,
    PullStats,
    check_key,
    clamp_per_page,
    make_client,
    parse_rate_limit,
    pull_works,
)

CFG = {
    "data": {
        "page_size": 100,
        "mailto": None,
        "max_retries": 3,
        "backoff_base_s": 0.0,
    }
}

W1 = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1/A",
    "title": "Paper one",
    "abstract_inverted_index": {"Deep": [0], "learning": [1]},
    "publication_date": "2020-01-01",
    "concepts": [{"id": "https://openalex.org/C41008148"}],
    "primary_location": {"source": {"id": "https://openalex.org/S123"}},
    "is_retracted": False,
    "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
}
W4_NO_DATE = {
    "id": "https://openalex.org/W4",
    "referenced_works": ["https://openalex.org/W5"],
}
W6 = {"id": "https://openalex.org/W6", "publication_date": "1970-01-02"}


class FakeHTTP(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class FlakyPager:
    """Raises the given error on the first __next__, then serves one page."""

    def __init__(self, page, status):
        self.page, self.status, self.calls, self.served = page, status, 0, False

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        if self.calls == 1:
            raise FakeHTTP(self.status)
        if self.served:
            raise StopIteration
        self.served = True
        return self.page


def test_writes_contracted_parquet_schemas(tmp_path):
    stats = pull_works({}, tmp_path, _pager=[[W1], [W4_NO_DATE, W6]], config=CFG)

    assert isinstance(stats, PullStats)
    assert (stats.works, stats.edges, stats.pages, stats.works_no_date) == (3, 2, 2, 1)
    assert stats.est_cost_usd == pytest.approx(2 * 0.0001)

    works = pd.read_parquet(tmp_path / "works.parquet")
    edges = pd.read_parquet(tmp_path / "edges.parquet")
    assert list(works.columns) == [
        "work_id", "doi_norm", "title", "abstract", "pub_date",
        "concepts", "venue_id", "is_retracted_oa",
    ]
    assert list(edges.columns) == ["src_id", "dst_id", "t", "doi_pair_key"]

    w1 = works[works.work_id == "W1"].iloc[0]
    assert w1.doi_norm == "10.1/a"
    assert w1.abstract == "Deep learning"
    assert w1.venue_id == "S123"
    assert list(w1.concepts) == ["C41008148"]

    assert set(edges.dst_id) == {"W2", "W3"}          # W4's ref dropped: no timestamp
    assert set(edges.t) == {18262}                     # days(2020-01-01)
    assert set(edges.doi_pair_key) == {"W1|W2", "W1|W3"}


def test_retries_retryable_status_then_succeeds(tmp_path):
    pager = FlakyPager([W6], status=429)
    stats = pull_works({}, tmp_path, _pager=pager, config=CFG)
    assert (stats.pages, stats.works) == (1, 1)
    assert pager.calls == 3  # fail, page, StopIteration


def test_non_retryable_status_raises_immediately(tmp_path):
    pager = FlakyPager([W6], status=403)
    with pytest.raises(FakeHTTP):
        pull_works({}, tmp_path, _pager=pager, config=CFG)
    assert pager.calls == 1


def test_max_works_stops_early(tmp_path):
    stats = pull_works({}, tmp_path, _pager=[[W1, W6, W6]], config=CFG, max_works=2)
    assert stats.works == 2
    assert len(pd.read_parquet(tmp_path / "works.parquet")) == 2


def test_missing_api_key_refuses_keyless_pull(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    with pytest.raises(ApiKeyMissing):
        pull_works({}, tmp_path, config=CFG)


# --------------------------------------------------------------------------- cursor-aware harness

CFG_C = {"data": {"page_size": 5, "mailto": None, "max_retries": 3, "backoff_base_s": 0.0}}
NOW = datetime(2026, 10, 31, 15, 20, tzinfo=timezone.utc)
NEXT_MIDNIGHT = datetime(2026, 11, 1, tzinfo=timezone.utc)


def make_works(n):
    """n works W1..Wn; Wi cites up to three earlier works; every 4th work has no date."""
    out = []
    for i in range(1, n + 1):
        w = {
            "id": f"https://openalex.org/W{i}",
            "doi": f"https://doi.org/10.9/{i}",
            "title": f"Title {i}",
            "concepts": [{"id": f"https://openalex.org/C{i % 3}"}],
            "referenced_works": [f"https://openalex.org/W{j}" for j in range(max(1, i - 3), i)],
        }
        if i % 4:
            w["publication_date"] = f"2020-01-{1 + i % 28:02d}"
        out.append(w)
    return out


class FakeOpenAlex:
    """Cursor server over a fixed work list. Cursor '*' starts; 'c<k>' resumes at offset k.

    fail maps a 1-based call number to the exception raised on that call.
    """

    def __init__(self, works, fail=None, cost=0.0001):
        self.works, self.fail, self.cost, self.calls = works, dict(fail or {}), cost, []

    def __call__(self, cursor, per_page):
        self.calls.append((cursor, per_page))
        exc = self.fail.pop(len(self.calls), None)
        if exc is not None:
            raise exc
        start = 0 if cursor == "*" else int(cursor[1:])
        end = start + per_page
        nxt = f"c{end}" if end < len(self.works) else None
        meta = {"count": len(self.works), "per_page": per_page, "next_cursor": nxt}
        if self.cost is not None:
            meta["cost_usd"] = self.cost
        return {"meta": meta, "results": self.works[start:end]}


def read_out(d):
    return pd.read_parquet(d / "works.parquet"), pd.read_parquet(d / "edges.parquet")


def assert_same_output(d1, d2):
    w1, e1 = read_out(d1)
    w2, e2 = read_out(d2)
    pd.testing.assert_frame_equal(w1, w2)
    pd.testing.assert_frame_equal(e1, e2)


def reference_pull(tmp_path, works, cfg=CFG_C):
    """Uninterrupted pull, in memory (no state file)."""
    ref = tmp_path / "ref"
    stats = pull_works({"x": "1"}, ref, config=cfg, _fetch=FakeOpenAlex(works))
    return ref, stats


def test_clamp_per_page():
    assert clamp_per_page(100) == 100
    assert clamp_per_page(25) == 25
    with pytest.warns(UserWarning, match="deprecated"):
        assert clamp_per_page(200) == 100
    with pytest.raises(ValueError):
        clamp_per_page(0)


def test_pull_clamps_per_page_to_100(tmp_path):
    fake = FakeOpenAlex(make_works(3))
    cfg = {"data": {**CFG_C["data"], "page_size": 200}}
    with pytest.warns(UserWarning, match="deprecated"):
        stats = pull_works({}, tmp_path, config=cfg, _fetch=fake)
    assert {pp for _, pp in fake.calls} == {100}
    assert stats.works == 3


def test_cursor_pull_matches_hand_counts_and_meta_cost(tmp_path):
    works = make_works(12)
    fake = FakeOpenAlex(works)
    stats = pull_works({}, tmp_path, config=CFG_C, _fetch=fake)
    # 12 works on pages of 5 -> 3 pages; W4, W8, W12 have no date; Wi cites min(3, i - 1) works
    dated_refs = sum(min(3, i - 1) for i in range(1, 13) if i % 4)
    assert (stats.works, stats.pages, stats.works_no_date, stats.edges) == (12, 3, 3, dated_refs)
    assert [c for c, _ in fake.calls] == ["*", "c5", "c10"]
    assert stats.cost_usd_reported == pytest.approx(3 * 0.0001)
    assert stats.est_cost_usd == pytest.approx(3 * 0.0001)
    works_df, edges_df = read_out(tmp_path)
    assert list(works_df.work_id) == [f"W{i}" for i in range(1, 13)]
    assert edges_df["t"].dtype == "int64"


def test_429_budget_spent_raises_budget_exhausted_with_resume_state(tmp_path):
    works = make_works(12)
    ref, _ = reference_pull(tmp_path, works)
    state_path = tmp_path / "run" / "pull_state.json"
    fake = FakeOpenAlex(works, fail={2: OpenAlexHTTPError(429)})
    rl_calls = []

    def rate_limit():
        rl_calls.append(1)
        return {"rate_limit": {"credits_limit": 10000, "credits_used": 10000, "credits_remaining": 0}}

    with pytest.raises(BudgetExhausted) as info:
        pull_works({"x": "1"}, tmp_path / "run", config=CFG_C, state_path=state_path,
                   _fetch=fake, _rate_limit=rate_limit, _now=lambda: NOW)
    e = info.value
    assert e.reset_at == NEXT_MIDNIGHT
    assert "2026-11-01 00:00 UTC" in str(e)
    assert e.state_path == state_path
    assert e.state["cursor"] == "c5" and e.state["pages"] == 1 and e.state["works"] == 5
    assert json.loads(state_path.read_text()) == e.state
    assert len(rl_calls) == 1 and len(fake.calls) == 2      # no blind retries once the budget is known spent
    assert not (tmp_path / "run" / "works.parquet").exists()

    resumed = FakeOpenAlex(works)
    stats = pull_works({"x": "1"}, tmp_path / "run", config=CFG_C, state_path=state_path, _fetch=resumed)
    assert [c for c, _ in resumed.calls] == ["c5", "c10"]
    assert (stats.pages, stats.pages_resumed, stats.works) == (3, 1, 12)
    assert_same_output(ref, tmp_path / "run")


def test_budget_reset_time_from_rate_limit_payload():
    rl = parse_rate_limit({"rate_limit": {"credits_remaining": 0, "resets_in_seconds": 3600}}, now=NOW)
    assert rl["exhausted"] is True
    assert rl["reset_at"] == datetime(2026, 10, 31, 16, 20, tzinfo=timezone.utc)
    rl = parse_rate_limit({"rate_limit": {"credits_remaining": 0, "resets_at": "2026-11-01T00:00:00.000Z"}}, now=NOW)
    assert rl["reset_at"] == NEXT_MIDNIGHT
    rl = parse_rate_limit({"remaining_usd": 0.42}, now=NOW)
    assert rl["exhausted"] is False and rl["remaining"] == pytest.approx(0.42)
    assert rl["reset_at"] == NEXT_MIDNIGHT
    assert parse_rate_limit({"something": "else"}, now=NOW)["exhausted"] is None


def test_429_rate_limit_ok_backs_off_and_retries(tmp_path):
    works = make_works(7)
    fake = FakeOpenAlex(works, fail={2: OpenAlexHTTPError(429)})
    rl_calls, sleeps = [], []

    def rate_limit():
        rl_calls.append(1)
        return {"rate_limit": {"credits_remaining": 5000}}

    cfg = {"data": {**CFG_C["data"], "backoff_base_s": 1.5}}
    stats = pull_works({}, tmp_path, config=cfg, _fetch=fake, _rate_limit=rate_limit, _sleep=sleeps.append)
    assert stats.works == 7 and stats.pages == 2
    assert [c for c, _ in fake.calls] == ["*", "c5", "c5"]
    assert len(rl_calls) == 1 and sleeps == [1.5]


def test_429_falls_back_to_response_headers_when_rate_limit_endpoint_fails(tmp_path):
    fake = FakeOpenAlex(make_works(7), fail={1: OpenAlexHTTPError(429, headers={"x-ratelimit-remaining": "0"})})

    def broken():
        raise OSError("endpoint down")

    with pytest.raises(BudgetExhausted) as info:
        pull_works({}, tmp_path, config=CFG_C, _fetch=fake, _rate_limit=broken, _now=lambda: NOW)
    assert info.value.reset_at == NEXT_MIDNIGHT
    assert info.value.state_path is None and info.value.state["cursor"] == "*"
    assert "state_path" in str(info.value)          # tells the user how to make pulls resumable


def test_429_of_unknown_kind_retries_then_raises(tmp_path):
    fake = FakeOpenAlex(make_works(3), fail={i: OpenAlexHTTPError(429) for i in range(1, 10)})
    with pytest.raises(OpenAlexHTTPError):
        pull_works({}, tmp_path, config=CFG_C, _fetch=fake)
    assert len(fake.calls) == CFG_C["data"]["max_retries"] + 1


def test_resume_from_saved_cursor_reproduces_uninterrupted_pull(tmp_path):
    works = make_works(23)
    ref, ref_stats = reference_pull(tmp_path, works)
    state_path = tmp_path / "run" / "pull_state.json"
    dying = FakeOpenAlex(works, fail={3: RuntimeError("session died")})
    with pytest.raises(RuntimeError, match="session died"):
        pull_works({"x": "1"}, tmp_path / "run", config=CFG_C, state_path=state_path, _fetch=dying)
    st = json.loads(state_path.read_text())
    assert (st["cursor"], st["pages"], st["works"], st["done"]) == ("c10", 2, 10, False)
    assert len(st["parts"]) == 2

    resumed = FakeOpenAlex(works)
    stats = pull_works({"x": "1"}, tmp_path / "run", config=CFG_C, state_path=state_path, _fetch=resumed)
    assert [c for c, _ in resumed.calls] == ["c10", "c15", "c20"]
    assert stats.pages_resumed == 2
    assert (stats.works, stats.edges, stats.pages, stats.works_no_date) == (
        ref_stats.works, ref_stats.edges, ref_stats.pages, ref_stats.works_no_date)
    assert_same_output(ref, tmp_path / "run")
    st = json.loads(state_path.read_text())
    assert st["done"] is True and st["cursor"] is None
    assert not (tmp_path / "run" / "pull_state_parts").exists()     # transient parts removed

    again = FakeOpenAlex(works)
    stats2 = pull_works({"x": "1"}, tmp_path / "run", config=CFG_C, state_path=state_path, _fetch=again)
    assert again.calls == [] and stats2.works == 23                    # a finished pull is not repeated


def test_resume_with_several_pages_per_part(tmp_path):
    works = make_works(23)
    ref, _ = reference_pull(tmp_path, works)
    state_path = tmp_path / "run" / "s.json"
    dying = FakeOpenAlex(works, fail={4: RuntimeError("killed")})
    with pytest.raises(RuntimeError):
        pull_works({"x": "1"}, tmp_path / "run", config=CFG_C, state_path=state_path, pages_per_part=2,
                   _fetch=dying)
    st = json.loads(state_path.read_text())
    assert st["pages"] == 3 and st["cursor"] == "c15"   # buffered page 3 committed before re-raising
    stats = pull_works({"x": "1"}, tmp_path / "run", config=CFG_C, state_path=state_path, pages_per_part=2,
                       _fetch=FakeOpenAlex(works))
    assert stats.pages == 5
    assert_same_output(ref, tmp_path / "run")


def test_resume_rolls_back_to_first_missing_part(tmp_path):
    """A part listed in the state but missing on disk (Drive sync lag) is fetched again."""
    works = make_works(23)
    ref, _ = reference_pull(tmp_path, works)
    state_path = tmp_path / "run" / "pull_state.json"
    with pytest.raises(RuntimeError):
        pull_works({"x": "1"}, tmp_path / "run", config=CFG_C, state_path=state_path,
                   _fetch=FakeOpenAlex(works, fail={4: RuntimeError("x")}))
    (tmp_path / "run" / "pull_state_parts" / "edges_00001.parquet").unlink()
    resumed = FakeOpenAlex(works)
    with pytest.warns(UserWarning, match="missing"):
        stats = pull_works({"x": "1"}, tmp_path / "run", config=CFG_C, state_path=state_path, _fetch=resumed)
    assert [c for c, _ in resumed.calls] == ["c5", "c10", "c15", "c20"]
    assert stats.pages_resumed == 1
    assert_same_output(ref, tmp_path / "run")


def test_state_of_a_different_query_is_refused(tmp_path):
    works = make_works(12)
    state_path = tmp_path / "s.json"
    with pytest.raises(RuntimeError):
        pull_works({"x": "1"}, tmp_path, config=CFG_C, state_path=state_path,
                   _fetch=FakeOpenAlex(works, fail={2: RuntimeError("x")}))
    with pytest.raises(ValueError, match="different query"):
        pull_works({"x": "2"}, tmp_path, config=CFG_C, state_path=state_path, _fetch=FakeOpenAlex(works))


def test_max_works_with_state(tmp_path):
    stats = pull_works({}, tmp_path, max_works=7, config=CFG_C, state_path=tmp_path / "s.json",
                       _fetch=FakeOpenAlex(make_works(23)))
    assert stats.works == 7 and stats.pages == 2
    assert len(pd.read_parquet(tmp_path / "works.parquet")) == 7


def test_legacy_pager_rejects_state_path(tmp_path):
    with pytest.raises(ValueError, match="state_path"):
        pull_works({}, tmp_path, _pager=[[W6]], config=CFG, state_path=tmp_path / "s.json")


def test_missing_api_key_refuses_resumable_pull_and_check_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    with pytest.raises(ApiKeyMissing):
        pull_works({}, tmp_path, config=CFG, state_path=tmp_path / "s.json")
    with pytest.raises(ApiKeyMissing):
        check_key(CFG)
    assert not (tmp_path / "s.json").exists()


# --------------------------------------------------------------------------- HTTP client (fake session)

KEY = "sk-test-SECRET-1234"


class FakeResponse:
    def __init__(self, status, payload, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def json(self):
        return self._payload


class FakeSession:
    """Records (url, headers); answers from a list of responses or a callable(url)."""

    def __init__(self, responses):
        self.responses, self.requests = responses, []

    def get(self, url, headers=None, timeout=None):
        self.requests.append((url, dict(headers or {})))
        if callable(self.responses):
            return self.responses(url)
        return self.responses.pop(0)


PAGE = {"meta": {"next_cursor": None, "count": 1, "cost_usd": 0.0001}, "results": [{"id": "W1"}]}


def test_client_bearer_mode_sends_header_not_param():
    s = FakeSession([FakeResponse(200, PAGE)])
    c = OpenAlexClient(KEY, key_mode="bearer", session=s, max_requests_per_s=0)
    with pytest.warns(UserWarning, match="deprecated"):
        out = c.list_page("works", {"publication_year": "2020"}, cursor="*", per_page=200)
    url, headers = s.requests[0]
    assert out["results"] == [{"id": "W1"}]
    assert headers["Authorization"] == f"Bearer {KEY}" and "api_key" not in url
    assert url.startswith("https://api.openalex.org/works?filter=publication_year:2020")
    assert "per-page=100" in url and "cursor=%2A" in url and "mailto" not in url


def test_client_param_mode_sends_api_key_param_and_mailto_only_as_header():
    s = FakeSession([FakeResponse(200, PAGE)])
    c = OpenAlexClient(KEY, key_mode="param", session=s, max_requests_per_s=0, mailto="me@example.org")
    c.list_page("works", {}, cursor="Ab+c=", per_page=50)
    url, headers = s.requests[0]
    assert "Authorization" not in headers and f"api_key={KEY}" in url
    assert "cursor=Ab%2Bc%3D" in url and "per-page=50" in url and "mailto" not in url
    assert headers.get("From") == "me@example.org"


def test_key_mode_switch_from_env_and_config(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    monkeypatch.delenv("OPENALEX_KEY_MODE", raising=False)
    assert make_client({}).key_mode == "bearer"
    assert make_client({"api_key_mode": "param"}).key_mode == "param"
    monkeypatch.setenv("OPENALEX_KEY_MODE", "param")
    assert make_client({"api_key_mode": "bearer"}).key_mode == "param"     # env wins over config
    assert make_client({}, key_mode="bearer").key_mode == "bearer"          # argument wins over env
    with pytest.raises(ValueError):
        make_client({}, key_mode="cookie")


def test_http_errors_carry_status_and_never_the_key():
    s = FakeSession([FakeResponse(429, {"error": "Rate limit exceeded"}, {"X-RateLimit-Remaining": "0"})])
    c = OpenAlexClient(KEY, key_mode="param", session=s, max_requests_per_s=0)
    with pytest.raises(OpenAlexHTTPError) as info:
        c.list_page("works", {}, cursor="*", per_page=100)
    e = info.value
    assert e.status == 429 and e.response.status_code == 429
    assert e.response.headers["X-RateLimit-Remaining"] == "0"
    assert KEY not in str(e) and "api_key=***" in str(e) and "Rate limit exceeded" in str(e)


def test_transport_errors_are_transient_and_redacted():
    import requests

    class Boom:
        def get(self, url, headers=None, timeout=None):
            raise requests.ConnectionError(f"Max retries exceeded with url: {url}")

    c = OpenAlexClient(KEY, key_mode="param", session=Boom(), max_requests_per_s=0)
    with pytest.raises(OpenAlexHTTPError) as info:
        c.list_page("works", {}, cursor="*", per_page=100)
    assert info.value.transient and info.value.status is None
    assert KEY not in str(info.value) and info.value.__suppress_context__


def test_transient_errors_are_retried(tmp_path):
    fake = FakeOpenAlex(make_works(3), fail={1: OpenAlexHTTPError(None, transient=True)})
    stats = pull_works({}, tmp_path, config=CFG_C, _fetch=fake)
    assert stats.works == 3 and len(fake.calls) == 2


def test_client_throttles_to_max_requests_per_second():
    sleeps = []
    s = FakeSession(lambda url: FakeResponse(200, PAGE))
    c = OpenAlexClient(KEY, session=s, max_requests_per_s=10, clock=lambda: 100.0, sleep=sleeps.append)
    c.list_page("works", {}, cursor="*", per_page=1)
    c.list_page("works", {}, cursor="*", per_page=1)
    assert sleeps == [pytest.approx(0.1)]


def _rate_limit_session(used_sequence, probe_status=200):
    used = list(used_sequence)

    def answer(url):
        if "/rate-limit" in url:
            u = used.pop(0)
            return FakeResponse(200, {"api_key": "sk-...1234", "rate_limit": {
                "credits_limit": 10000, "credits_used": u, "credits_remaining": 10000 - u,
                "resets_in_seconds": 3600}})
        return FakeResponse(probe_status, PAGE, {"X-RateLimit-Limit": "10000", "X-RateLimit-Credits-Used": "1"})

    return FakeSession(answer)


def test_check_key_reports_calls_counted_against_the_key():
    s = _rate_limit_session([100, 100, 101])
    rep = check_key({"data": {}}, _client=OpenAlexClient(KEY, session=s, max_requests_per_s=0))
    assert rep["counted_against_key"] is True and rep["ok"] is True
    assert rep["probe_status"] == 200 and rep["probe_meta"]["cost_usd"] == 0.0001
    assert rep["probe_delta"] == 1 and rep["baseline_delta"] == 0
    assert rep["probe_rate_headers"]["X-RateLimit-Limit"] == "10000"
    rl_urls = [u for u, _ in s.requests if "/rate-limit" in u]
    assert len(rl_urls) == 3 and all("api_key=" in u for u in rl_urls)
    assert KEY not in json.dumps(rep, default=str)


def test_check_key_reports_calls_not_counted():
    s = _rate_limit_session([100, 100, 100])
    rep = check_key({"data": {}}, _client=OpenAlexClient(KEY, session=s, max_requests_per_s=0))
    assert rep["counted_against_key"] is False and rep["ok"] is False
    assert "not counted" in rep["verdict"]
