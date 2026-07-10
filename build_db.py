"""
build_db.py
-----------
Builds and maintains the local SQLite database of Reddit posts + comments
that replaces live Arctic Shift / PullPush API calls at request time.

Run with the project venv (system python lacks the dependencies, including
the dotenv package that loads .env): ./env/bin/python build_db.py ...

Workflow:
  1. Download subreddit dumps (posts AND comments) with the Arctic Shift
     download tool: https://arctic-shift.photon-reddit.com/download-tool
  2. ./env/bin/python build_db.py ingest <dump files...>
  3. ./env/bin/python build_db.py extract   (build the movie -> recs cache)
  4. Later, to pull everything newer than the last dump/update straight
     from the Arctic Shift API (no re-download needed), then extract the
     new threads: ./env/bin/python build_db.py update && ./env/bin/python
     build_db.py extract

Commands:
  ingest <files...>   Load dump files (.jsonl / .ndjson, optionally .gz or
                      .zst, or a single JSON array). Posts vs comments are
                      auto-detected per record; safe to rerun — records
                      upsert by id, so overlapping dumps just refresh rows.
  update              Fill the gap between the newest stored data and now,
                      per subreddit, by paging the Arctic Shift API.
                      Rerunnable; also refreshes comment scores near the
                      boundary via a small overlap window.
  stats               Show row counts and date coverage per subreddit.
  search <query>      Test a full-text search against stored post titles.

  extract [--limit N] Run Groq over stored threads in two stages: a cheap
                      small-model screen decides whether the post names a
                      specific movie; only then does the big model extract
                      the recommended titles (with upvotes) from comments.
                      Checkpoints per post — safe to stop and rerun; only
                      unprocessed posts are sent. Needs GROQ_API_KEY.
                      Re-aggregates the cache when done.
  aggregate           Rebuild the recs cache from existing extractions.
  recs <title>        Query the cache: recommendations for one movie,
                      sorted by frequency weighted by upvotes.
  movies              List all movies in the cache with their rec counts.
  export              Write the recs cache alone to a slim, committable
                      recs.db at the repo root — the deploy artifact the
                      app serves from when the full DB isn't present.
                      Rerun (and commit) after every extract to ship the
                      grown cache.

DB path defaults to data/movies.db (override with MOVIEREXER_DB env var).
"""

import gzip
import io
import itertools
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import requests

try:
    from dotenv import load_dotenv
    # load the project's .env regardless of the caller's working directory
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

DB_PATH     = os.environ.get("MOVIEREXER_DB", os.path.join("data", "movies.db"))
ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
SUBREDDITS  = ["MovieRecommendations", "MovieSuggestions"]
HEADERS     = {"User-Agent": "movierexer local dump builder"}

PAGE_LIMIT   = 100    # Arctic Shift max page size
PAGE_SLEEP   = 2.0    # seconds between API pages — be gentle, it rate-limits
                      # (422 "slow down" replies get exponential backoff too)
OVERLAP_SECS = 3600   # re-fetch this much before the newest stored row so
                      # boundary items and their late comments aren't missed
                      # (upserts make the overlap free)

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id          TEXT PRIMARY KEY,
    subreddit   TEXT NOT NULL,
    title       TEXT NOT NULL,
    selftext    TEXT NOT NULL DEFAULT '',
    created_utc INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_sub_time ON posts(subreddit, created_utc);

CREATE TABLE IF NOT EXISTS comments (
    id          TEXT PRIMARY KEY,
    post_id     TEXT NOT NULL,
    subreddit   TEXT NOT NULL,
    body        TEXT NOT NULL,
    score       INTEGER NOT NULL,
    created_utc INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_comments_sub_time ON comments(subreddit, created_utc);

CREATE VIRTUAL TABLE IF NOT EXISTS posts_fts USING fts5(
    title, selftext, content='posts', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS posts_ai AFTER INSERT ON posts BEGIN
    INSERT INTO posts_fts(rowid, title, selftext)
    VALUES (new.rowid, new.title, new.selftext);
END;
CREATE TRIGGER IF NOT EXISTS posts_ad AFTER DELETE ON posts BEGIN
    INSERT INTO posts_fts(posts_fts, rowid, title, selftext)
    VALUES ('delete', old.rowid, old.title, old.selftext);
END;

-- One row per LLM-processed post (the checkpoint for `extract`).
-- source_movies / recs are JSON blobs; a post with no usable content still
-- gets a row so it is never sent to the LLM twice.
CREATE TABLE IF NOT EXISTS post_extractions (
    post_id       TEXT PRIMARY KEY,
    source_movies TEXT NOT NULL,
    recs          TEXT NOT NULL,
    extracted_at  INTEGER NOT NULL
);

-- The queryable product: Movie -> Recommendations[], rebuilt by `aggregate`.
CREATE TABLE IF NOT EXISTS recs_cache (
    source_key    TEXT NOT NULL,   -- normalized source title (join key)
    source_title  TEXT NOT NULL,   -- display form
    rec_key       TEXT NOT NULL,
    rec_title     TEXT NOT NULL,
    n_threads     INTEGER NOT NULL,  -- how many threads suggested it
    total_upvotes INTEGER NOT NULL,  -- summed comment upvotes (the sort score)
    PRIMARY KEY (source_key, rec_key)
);
"""


def _connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


# ── RECORD CLEANING ───────────────────────────────────────────────────────────
# Mirrors the filters the live app applies (reddit_recs._get_comments): drop
# deleted/removed content, short bodies, and non-positive scores; truncate
# bodies to what the LLM extraction actually reads.

def _clean_post(raw):
    title = (raw.get("title") or "").strip()
    if not title or title in ("[deleted]", "[removed]"):
        return None
    selftext = raw.get("selftext") or ""
    if selftext in ("[deleted]", "[removed]"):
        selftext = ""
    try:
        created = int(float(raw.get("created_utc") or 0))
    except (TypeError, ValueError):
        created = 0
    return (raw.get("id"), raw.get("subreddit") or "", title,
            selftext[:1000], created)


def _clean_comment(raw):
    body = (raw.get("body") or "").strip()
    if body in ("[deleted]", "[removed]") or len(body) <= 10:
        return None
    try:
        score = int(raw.get("score") or 0)
    except (TypeError, ValueError):
        score = 0
    if score <= 0:
        return None
    link_id = raw.get("link_id") or ""
    post_id = link_id[3:] if link_id.startswith("t3_") else link_id
    if not post_id:
        return None
    try:
        created = int(float(raw.get("created_utc") or 0))
    except (TypeError, ValueError):
        created = 0
    return (raw.get("id"), post_id, raw.get("subreddit") or "",
            body[:300], score, created)


UPSERT_POST = """INSERT INTO posts (id, subreddit, title, selftext, created_utc)
                 VALUES (?, ?, ?, ?, ?)
                 ON CONFLICT(id) DO NOTHING"""
# Comments upsert refreshes the score: dumps and gap-fill pulls made at
# different times see different vote counts, and newer is better.
UPSERT_COMMENT = """INSERT INTO comments (id, post_id, subreddit, body, score, created_utc)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET score = excluded.score"""


def _store(con, raw):
    """Route one raw record to the right table. Returns 'post', 'comment' or None."""
    if "title" in raw:
        row = _clean_post(raw)
        if row and row[0]:
            con.execute(UPSERT_POST, row)
            return "post"
    elif "body" in raw:
        row = _clean_comment(raw)
        if row and row[0]:
            con.execute(UPSERT_COMMENT, row)
            return "comment"
    return None


# ── INGEST (local dump files) ─────────────────────────────────────────────────

def _open_stream(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    if path.endswith(".zst"):
        try:
            import zstandard
        except ImportError:
            sys.exit("Reading .zst dumps needs the zstandard package: pip install zstandard")
        dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
        return io.TextIOWrapper(dctx.stream_reader(open(path, "rb")),
                                encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _iter_records(path):
    """Yields dicts from a JSONL file, or from a single JSON array."""
    with _open_stream(path) as f:
        first = f.readline()
        while first and not first.strip():
            first = f.readline()
        if not first:
            return
        if first.lstrip().startswith("["):
            # Whole-file JSON array (the download tool's non-JSONL export)
            data = json.loads(first + f.read())
            for raw in data:
                yield raw
            return
        bad = 0
        for line in itertools.chain([first], f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                # e.g. a truncated final line from an interrupted download
                bad += 1
        if bad:
            print("  ⚠️  skipped " + str(bad) + " malformed line(s) — "
                  "if the download was cut short, re-run ingest on the complete file")


def cmd_ingest(paths):
    if not paths:
        sys.exit("Usage: python build_db.py ingest <dump files...>")
    con = _connect()
    totals = {"post": 0, "comment": 0, "skipped": 0}
    for path in paths:
        print("Ingesting " + path)
        n = 0
        for raw in _iter_records(path):
            kind = _store(con, raw)
            totals[kind or "skipped"] += 1
            n += 1
            if n % 20000 == 0:
                con.commit()
                print("  ..." + str(n) + " records")
        con.commit()
        print("  done (" + str(n) + " records)")
    con.close()
    print("Ingested: " + str(totals["post"]) + " posts, "
          + str(totals["comment"]) + " comments"
          + " (" + str(totals["skipped"]) + " filtered out)")


# ── UPDATE (fill the gap via the Arctic Shift API) ────────────────────────────

def _pull(con, subreddit, kind, after):
    """Pages the Arctic Shift API forward in time from `after`. Returns count."""
    endpoint = ARCTIC_BASE + ("/posts/search" if kind == "posts" else "/comments/search")
    total = 0
    while True:
        # Retry rate limits (422/429) and hiccups with exponential backoff
        # rather than bailing — bulk paging is expected to get throttled.
        resp = None
        for attempt in range(6):
            try:
                resp = requests.get(
                    endpoint,
                    params={"subreddit": subreddit, "after": int(after),
                            "sort": "asc", "limit": PAGE_LIMIT},
                    headers=HEADERS, timeout=30
                )
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            wait = 15 * (2 ** attempt)   # 15s → 480s
            code = str(resp.status_code) if resp is not None else "no response"
            print("    " + code + " from Arctic Shift — backing off " + str(wait) + "s")
            time.sleep(wait)
        else:
            sys.exit("Arctic Shift kept refusing after repeated backoff — "
                     "try again later. Progress so far is saved.")
        rows = resp.json().get("data") or []
        if not rows:
            break
        for raw in rows:
            if _store(con, raw):
                total += 1
        con.commit()
        newest = int(float(rows[-1].get("created_utc") or after))
        print("    r/" + subreddit + " " + kind + ": +" + str(len(rows))
              + " (through " + datetime.fromtimestamp(newest, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") + ")")
        if len(rows) < PAGE_LIMIT:
            break
        # guard against a page of identical timestamps looping forever
        after = newest if newest > after else after + 1
        time.sleep(PAGE_SLEEP)
    return total


def cmd_update():
    con = _connect()
    for subreddit in SUBREDDITS:
        print("Updating r/" + subreddit)
        for kind, table in (("posts", "posts"), ("comments", "comments")):
            last = con.execute(
                "SELECT MAX(created_utc) FROM " + table + " WHERE subreddit = ?",
                (subreddit,)
            ).fetchone()[0]
            if last is None:
                print("  no " + kind + " for r/" + subreddit
                      + " yet — ingest a dump first (update only fills gaps)")
                continue
            added = _pull(con, subreddit, kind, last - OVERLAP_SECS)
            print("  " + kind + ": " + str(added) + " new/updated")
    con.close()


# ── LLM EXTRACTION ────────────────────────────────────────────────────────────
# One Groq call per thread, offline, where rate limits don't hurt users.
# Unlike the live app's prompt, this also identifies the SOURCE movie(s) the
# poster is asking about, so results can be aggregated into movie -> recs.

# Model chains: first is preferred; later ones are fallbacks with separate
# Groq rate-limit quotas, used when the current model's daily cap is reached.
EXTRACT_MODELS = [                          # comment extraction (the real work)
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
]
SCREEN_MODELS = [                           # cheap gate: does the post name a movie?
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
]
EXTRACT_SLEEP = 2.0   # seconds between Groq calls (free tier is ~30 req/min)
COMMENTS_PER_POST = 8
SCREEN_BATCH  = 25    # posts classified per screening call — skipped posts
                      # cost 1/SCREEN_BATCH of a call instead of a whole one

SCREEN_PROMPT = """You detect movie titles in Reddit posts from movie \
recommendation subreddits.

You are given several numbered posts. For EACH post, list every movie title
mentioned anywhere in its text.

Return ONLY a JSON object:
{ "results": [ { "post": 1, "source_movies": ["Movie Title", ...] }, ... ] }

Rules:
- One entry per numbered post, in order.
- ONLY include titles that are literally written in that post's text. Never
  infer, guess, or add example movies that are not mentioned.
- Movies only, no TV shows or books.
- Empty array if the post names no specific movie.

Example:
POST 1:
Just binged Gravity, need more space movies. Apollo 13 is already on my list.

Answer: { "results": [ { "post": 1,
  "source_movies": ["Gravity", "Apollo 13"] } ] }"""

EXTRACT_PROMPT = """You are a movie title extractor for Reddit posts from movie \
recommendation subreddits.

Given a post (title + body) and its top comments, return ONLY a JSON object:
{
  "source_movies": ["Movie Title", ...],
  "recommendations": [ {"title": "Movie Title", "upvotes": 123}, ... ]
}

Rules:
- source_movies: EVERY movie title mentioned anywhere in the post TITLE or
  BODY, in any context ("just watched X", "Y is already on my list", "not
  Z though"). This is pure extraction: if a movie title appears in the post
  text, list it. Do not add movies that are not literally written there.
- recommendations: movies suggested in the comments, each with the upvote
  count of the comment it came from
- Exact canonical movie titles only
- Movies only, no TV shows or books
- Maximum 8 recommendations"""


def _groq_client():
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            import dotenv  # noqa: F401 — present in the venv, not system python
            sys.exit("GROQ_API_KEY not set — add it to .env")
        except ImportError:
            sys.exit("GROQ_API_KEY not found because .env was not loaded: "
                     "this isn't the project venv. Rerun with "
                     "./env/bin/python build_db.py ...")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


# Cache keys must be produced exactly the same way the app queries them.
from reddit_recs import _norm_title  # noqa: E402


def _screen_posts(client, batch, model=SCREEN_MODELS[0]):
    """Cheap small-model gate for a batch of (post_id, title, selftext):
    which movies does each post itself name? Returns {post_id: [sources]}.
    Posts the model fails to answer for are omitted (retried next run)."""
    # Title and body are appended into one plain text blob — the model treats
    # the whole post as one text, so nothing hinges on which part named a movie.
    numbered = "\n\n".join(
        "POST " + str(i + 1) + ":\n" + (title + " " + selftext[:300]).strip()
        for i, (_pid, title, selftext) in enumerate(batch)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SCREEN_PROMPT},
                  {"role": "user",   "content": numbered}],
        temperature=0.0,
        response_format={"type": "json_object"},
        # generous headroom: fallback models may spend tokens on reasoning
        max_tokens=96 * len(batch) + 256,
    )
    data = json.loads(response.choices[0].message.content)
    out = {}
    for entry in data.get("results", []):
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("post")) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < len(batch):
            continue
        post_id, title, selftext = batch[idx]
        sources = [s.strip() for s in entry.get("source_movies") or []
                   if isinstance(s, str) and s.strip()]
        # Hard hallucination guard: keep only titles that literally appear in
        # the post (normalized, whole words — "En" must not match "Need"), so
        # invented or truncated "examples" never enter the cache.
        haystack = " " + _norm_title(title + " " + selftext[:500]) + " "
        out[post_id] = [s for s in sources if _norm_title(s)
                        and " " + _norm_title(s) + " " in haystack]
    return out


def _extract_post(client, title, selftext, comments, model=EXTRACT_MODELS[0]):
    """Full-model extraction: source movies from the post (title + body) and
    recommended titles from the comments. Returns (sources, recs)."""
    comments_text = "\n".join(
        "[" + str(score) + " upvotes] " + body for body, score in comments
    )
    user_msg = ("TITLE: " + title + "\nBODY: " + (selftext[:500] or "(empty)")
                + "\n\nCOMMENTS:\n" + comments_text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": EXTRACT_PROMPT},
                  {"role": "user",   "content": user_msg}],
        temperature=0.1,
        response_format={"type": "json_object"},
        # generous headroom: fallback models may spend tokens on reasoning
        max_tokens=1024,
    )
    data = json.loads(response.choices[0].message.content)
    # Same literal-mention guard as screening: a claimed source must appear
    # (whole words, normalized) in the post text.
    haystack = " " + _norm_title(title + " " + selftext[:500]) + " "
    sources = [s.strip() for s in data.get("source_movies") or []
               if isinstance(s, str) and s.strip()
               and " " + _norm_title(s.strip()) + " " in haystack]
    recs = []
    for r in data.get("recommendations", []):
        if isinstance(r, dict) and isinstance(r.get("title"), str) and r["title"].strip():
            try:
                upvotes = int(r.get("upvotes") or 0)
            except (TypeError, ValueError):
                upvotes = 0
            recs.append({"title": r["title"].strip(), "upvotes": upvotes})
    return sources, recs


class ModelPool:
    """Walks a model chain forward as daily caps are hit (never backward —
    an exhausted quota stays exhausted for the rest of the run)."""
    def __init__(self, models):
        self.models = list(models)
        self.idx = 0

    @property
    def model(self):
        return self.models[self.idx]

    def advance(self):
        """Move to the next model. Returns False when the chain is exhausted."""
        self.idx += 1
        if self.idx < len(self.models):
            print("    → falling back to " + self.model)
            return True
        return False


def _call_with_fallback(pool, fn, attempts=6):
    """Runs fn(model) with the pool's current model. Transient errors retry
    with exponential backoff; daily-cap errors (or persistent failure)
    advance the pool to the next model. Returns None when every model in the
    chain is exhausted; [] if the model returned unparseable JSON."""
    while True:
        for attempt in range(attempts):
            try:
                return fn(pool.model)
            except json.JSONDecodeError:
                return []
            except Exception as e:
                msg = str(e).replace("\n", " ")
                if "per day" in msg.lower() or "daily" in msg.lower():
                    print("    " + pool.model + " daily limit reached")
                    break
                wait = min(15 * (2 ** attempt), 300)
                print("    Groq error on " + pool.model + ": " + msg[:150]
                      + " — retrying in " + str(wait) + "s")
                time.sleep(wait)
        else:
            print("    " + pool.model + " keeps failing — trying next model")
        if not pool.advance():
            return None


def cmd_extract(limit=None):
    con = _connect()
    # unprocessed posts that actually have comments, richest threads first
    rows = con.execute("""
        SELECT p.id, p.title, p.selftext,
               (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS nc
        FROM posts p
        WHERE p.id NOT IN (SELECT post_id FROM post_extractions) AND nc > 0
        ORDER BY nc DESC
    """).fetchall()
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        print("Nothing to extract — all commented posts are processed.")
        _aggregate(con)
        con.close()
        return

    print(str(len(rows)) + " posts to process. "
          + "Ctrl-C any time — progress is saved per post.")
    client = _groq_client()
    screen_pool  = ModelPool(SCREEN_MODELS)
    extract_pool = ModelPool(EXTRACT_MODELS)
    done = 0
    stalled = False
    try:
        for start in range(0, len(rows), SCREEN_BATCH):
            batch = [(pid, title, selftext)
                     for pid, title, selftext, _nc in rows[start:start + SCREEN_BATCH]]

            # Stage 1 — one cheap call screens the whole batch.
            screened = _call_with_fallback(
                screen_pool, lambda m: _screen_posts(client, batch, m))
            if screened is None:
                stalled = True
            elif not isinstance(screened, dict):
                screened = {}   # model returned junk for the batch; retry next run

            if stalled:
                print("All screening models are exhausted (processed " + str(done)
                      + " this run). Progress is saved — rerun `extract` later.")
                break

            extracted_in_batch = False
            for post_id, title, selftext in batch:
                if post_id not in screened:
                    continue    # model skipped it in the batch answer; retry next run

                # Stage 2 — full extraction for posts the screen says name a
                # movie. The big model re-derives the sources itself (from
                # title AND body) — the small model's list is only a gate,
                # since it unreliably combines title and body mentions.
                sources, recs = [], []
                if screened[post_id]:
                    comments = con.execute(
                        "SELECT body, score FROM comments WHERE post_id = ? "
                        "ORDER BY score DESC LIMIT ?", (post_id, COMMENTS_PER_POST)
                    ).fetchall()
                    result = _call_with_fallback(
                        extract_pool,
                        lambda m: _extract_post(client, title, selftext, comments, m))
                    if result is None:
                        stalled = True
                        print("All extraction models are exhausted (processed " + str(done)
                              + " this run). Progress is saved — rerun `extract` later.")
                        break
                    if result:
                        sources, recs = result
                    if not sources:
                        # big model found none the guard accepts; fall back to
                        # the screen's guarded titles so the gate's evidence
                        # isn't lost entirely
                        sources = screened[post_id]
                    extracted_in_batch = True
                    time.sleep(EXTRACT_SLEEP)   # pace the big model's bucket

                con.execute(
                    "INSERT OR REPLACE INTO post_extractions VALUES (?, ?, ?, ?)",
                    (post_id, json.dumps(sources), json.dumps(recs), int(time.time()))
                )
                con.commit()
                done += 1
                if sources:
                    print("  [" + str(done) + "/" + str(len(rows)) + "] \"" + title[:60]
                          + "\" → " + ", ".join(sources) + " (" + str(len(recs)) + " recs)")
                else:
                    print("  [" + str(done) + "/" + str(len(rows)) + "] \"" + title[:60]
                          + "\" → skipped (no specific movie named)")
            if stalled:
                break
            if not extracted_in_batch:
                # all-skip batch: stage 2 added no spacing, so pace the
                # screening bucket explicitly
                time.sleep(EXTRACT_SLEEP)
    except KeyboardInterrupt:
        print("\nInterrupted — progress is saved; rerun `extract` to resume.")

    print("Extracted " + str(done) + " posts this run.")
    _aggregate(con)
    con.close()


def _aggregate(con):
    """Rebuilds recs_cache from post_extractions. Every movie in a thread —
    the post's source movies and the comments' recommendations — is linked to
    every other, in both directions ("everything points to everything").
    Edge weight per thread: source-source 1 (co-mentioned in a post),
    source-rec the rec comment's upvotes, rec-rec the smaller of the two
    comments' upvotes. Weights sum across threads; every edge counts >= 1."""
    agg = {}
    for sources_json, recs_json in con.execute(
            "SELECT source_movies, recs FROM post_extractions"):
        # Collect the thread's movies: key -> [display title, upvotes|None].
        # upvotes None marks a source (post) movie; sources win a role clash.
        nodes = {}
        for source in json.loads(sources_json):
            key = _norm_title(source)
            if key:
                nodes[key] = [source, None]
        for rec in json.loads(recs_json):
            key = _norm_title(rec["title"])
            if key and key not in nodes:
                nodes[key] = [rec["title"], max(rec.get("upvotes") or 0, 1)]

        keys = list(nodes)
        for i, a_key in enumerate(keys):
            for b_key in keys[i + 1:]:
                a_title, a_up = nodes[a_key]
                b_title, b_up = nodes[b_key]
                if a_up is None and b_up is None:
                    weight = 1                    # source-source co-mention
                elif a_up is None or b_up is None:
                    weight = a_up or b_up         # source-rec endorsement
                else:
                    weight = min(a_up, b_up)      # rec-rec co-recommendation
                for x_key, x_title, y_key, y_title in (
                        (a_key, a_title, b_key, b_title),
                        (b_key, b_title, a_key, a_title)):
                    entry = agg.setdefault((x_key, y_key), [x_title, y_title, 0, 0])
                    entry[2] += 1
                    entry[3] += weight
    con.execute("DELETE FROM recs_cache")
    con.executemany(
        "INSERT INTO recs_cache VALUES (?, ?, ?, ?, ?, ?)",
        [(sk, e[0], rk, e[1], e[2], e[3]) for (sk, rk), e in agg.items()]
    )
    con.commit()
    n_sources = con.execute("SELECT COUNT(DISTINCT source_key) FROM recs_cache").fetchone()[0]
    print("Cache rebuilt: " + str(len(agg)) + " movie->rec pairs across "
          + str(n_sources) + " source movies.")


def cmd_export():
    """Copies recs_cache into a slim SQLite file the app can serve from in
    production. Small enough to commit — the multi-GB posts/comments archive
    stays local."""
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recs.db")
    con = _connect()
    n = con.execute("SELECT COUNT(*) FROM recs_cache").fetchone()[0]
    if not n:
        sys.exit("recs_cache is empty — run `extract` before exporting.")
    if os.path.exists(dest):
        os.remove(dest)
    con.execute("ATTACH DATABASE ? AS slim", (dest,))
    con.execute("""CREATE TABLE slim.recs_cache (
        source_key    TEXT NOT NULL,
        source_title  TEXT NOT NULL,
        rec_key       TEXT NOT NULL,
        rec_title     TEXT NOT NULL,
        n_threads     INTEGER NOT NULL,
        total_upvotes INTEGER NOT NULL,
        PRIMARY KEY (source_key, rec_key)
    )""")
    con.execute("INSERT INTO slim.recs_cache SELECT * FROM recs_cache")
    con.commit()
    con.execute("DETACH DATABASE slim")
    con.close()
    size_kb = os.path.getsize(dest) // 1024
    print("Exported " + str(n) + " edges to " + dest + " (" + str(size_kb) + " KB). "
          + "Commit it to deploy the updated cache.")


def cmd_movies():
    con = _connect()
    rows = con.execute(
        """SELECT MAX(source_title), COUNT(*), SUM(n_threads)
           FROM recs_cache GROUP BY source_key
           ORDER BY MAX(source_title) COLLATE NOCASE"""
    ).fetchall()
    for title, n_recs, n_threads in rows:
        print("  " + title + "  (" + str(n_recs) + " recs from "
              + str(n_threads) + " suggestions)")
    print(str(len(rows)) + " movies in the cache.")
    con.close()


def cmd_recs(title):
    con = _connect()
    rows = con.execute(
        """SELECT rec_title, n_threads, total_upvotes FROM recs_cache
           WHERE source_key = ?
           ORDER BY total_upvotes DESC, n_threads DESC LIMIT 20""",
        (_norm_title(title),)
    ).fetchall()
    for rec_title, n_threads, total_upvotes in rows:
        print("  " + rec_title + "  (" + str(n_threads) + " threads, "
              + str(total_upvotes) + " upvotes)")
    if not rows:
        print("No cached recommendations for '" + title + "'.")
    con.close()


# ── STATS / SEARCH ────────────────────────────────────────────────────────────

def _fmt_ts(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else "—"


def cmd_stats():
    con = _connect()
    for subreddit in SUBREDDITS:
        p = con.execute("SELECT COUNT(*), MIN(created_utc), MAX(created_utc) "
                        "FROM posts WHERE subreddit = ?", (subreddit,)).fetchone()
        c = con.execute("SELECT COUNT(*), MAX(created_utc) "
                        "FROM comments WHERE subreddit = ?", (subreddit,)).fetchone()
        print("r/" + subreddit + ":")
        print("  posts:    " + str(p[0]) + "  (" + _fmt_ts(p[1]) + " → " + _fmt_ts(p[2]) + ")")
        print("  comments: " + str(c[0]) + "  (newest " + _fmt_ts(c[1]) + ")")
    con.close()


def cmd_search(query):
    con = _connect()
    # quote each term so FTS operators in user input can't break the query
    fts = " ".join('"' + t.replace('"', '') + '"' for t in query.split())
    rows = con.execute(
        """SELECT p.id, p.subreddit, p.title,
                  (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id)
           FROM posts_fts f JOIN posts p ON p.rowid = f.rowid
           WHERE posts_fts MATCH ? ORDER BY bm25(posts_fts) LIMIT 10""",
        (fts,)
    ).fetchall()
    for pid, sub, title, ncomments in rows:
        print(pid + "  r/" + sub + "  [" + str(ncomments) + " comments]  " + title)
    if not rows:
        print("No matches.")
    con.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ingest":
        cmd_ingest(sys.argv[2:])
    elif cmd == "update":
        cmd_update()
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "search":
        cmd_search(" ".join(sys.argv[2:]))
    elif cmd == "extract":
        n = None
        if "--limit" in sys.argv:
            n = int(sys.argv[sys.argv.index("--limit") + 1])
        cmd_extract(limit=n)
    elif cmd == "aggregate":
        con = _connect()
        _aggregate(con)
        con.close()
    elif cmd == "recs":
        cmd_recs(" ".join(sys.argv[2:]))
    elif cmd == "movies":
        cmd_movies()
    elif cmd == "export":
        cmd_export()
    else:
        print(__doc__.strip())
