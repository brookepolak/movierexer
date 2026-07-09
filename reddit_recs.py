"""
reddit_recs.py
--------------
Standalone recommendation engine. Takes a list of movie titles + genres,
searches Arctic Shift for each movie individually, extracts recommendations
via Groq, filters by genre via TMDB, and returns posters sorted by upvotes.

Requires in .env:
    GROQ_API_KEY
    TMDB_API_KEY
"""

import os
import re
import json
import time
import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

if not TMDB_API_KEY:
    print("⚠️  TMDB_API_KEY not set — posters will not load. Get a free key at themoviedb.org/settings/api")
ARCTIC_BASE  = "https://arctic-shift.photon-reddit.com/api"
PULLPUSH_BASE = "https://api.pullpush.io"
TMDB_BASE    = "https://api.themoviedb.org/3"
TMDB_IMG     = "https://image.tmdb.org/t/p/w300"
SUBREDDITS   = ["MovieRecommendations", "MovieSuggestions"]
HEADERS      = {"User-Agent": "Mozilla/5.0"}

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# Map from genre names → TMDB genre IDs
# Covers FEATURE_SCHEMA genres + many common aliases/subgenres
GENRE_MAP = {
    # ── Core TMDB genres ──────────────────────────────────────────────────────
    "action":            28,
    "adventure":         12,
    "animation":         16,
    "comedy":            35,
    "crime":             80,
    "documentary":       99,
    "drama":             18,
    "family":            10751,
    "fantasy":           14,
    "history":           36,
    "horror":            27,
    "music":             10402,
    "mystery":           9648,
    "romance":           10749,
    "sci-fi":            878,
    "science fiction":   878,
    "science-fiction":   878,
    "tv movie":          10770,
    "thriller":          53,
    "war":               10752,
    "western":           37,

    # ── Common aliases and subgenres ─────────────────────────────────────────
    "action-adventure":  12,
    "superhero":         28,
    "spy":               28,
    "heist":             80,
    "gangster":          80,
    "noir":              80,
    "psychological":     53,
    "psychological thriller": 53,
    "suspense":          53,
    "slasher":           27,
    "supernatural":      27,
    "ghost":             27,
    "zombie":            27,
    "monster":           27,
    "creature feature":  27,
    "found footage":     27,
    "body horror":       27,
    "survival horror":   27,
    "dark comedy":       35,
    "romantic comedy":   35,
    "rom-com":           35,
    "satire":            35,
    "buddy comedy":      35,
    "slapstick":         35,
    "stand-up":          35,
    "coming of age":     18,
    "coming-of-age":     18,
    "slice of life":     18,
    "indie":             18,
    "biographical":      36,
    "biopic":            36,
    "period piece":      36,
    "historical":        36,
    "epic":              12,
    "road movie":        12,
    "survival":          12,
    "disaster":          28,
    "space":             878,
    "alien":             878,
    "cyberpunk":         878,
    "dystopian":         878,
    "post-apocalyptic":  878,
    "time travel":       878,
    "mind-bending":      878,
    "cerebral":          878,
    "experimental":      878,
    "martial arts":      28,
    "sports":            18,
    "war film":          10752,
    "anti-war":          10752,
    "whodunit":          9648,
    "detective":         9648,
    "mystery thriller":  9648,
    "fairy tale":        14,
    "mythology":         14,
    "sword and sorcery": 14,
    "musical":           10402,
    "concert film":      10402,
    "animated":          16,
    "anime":             16,
    "stop motion":       16,
    "silent":            36,
    "foreign":           18,
    "arthouse":          18,
    "erotic thriller":   53,
    "legal thriller":    53,
    "political thriller": 53,
    "tech thriller":     53,
    "medical":           18,
    "courtroom":         18,
}

SYSTEM_PROMPT = """You are a movie title extractor for Reddit posts.

Given a post title and its top comments from a movie recommendation subreddit, extract
the movies being recommended in the comments.

Return ONLY a JSON object:
{ "recommendations": [ {"title": "Movie Title", "upvotes": 123}, ... ] }

Rules:
- Exact canonical movie titles only
- Movies only, no TV shows or books
- Maximum 8 recommendations per post"""


# ── TMDB ──────────────────────────────────────────────────────────────────────

def _tmdb_data(title):
    """
    Search TMDB for a movie. Returns dict with poster_url, genre_ids, and
    movie_url (TMDB page), or None if not found / no API key.
    """
    if not TMDB_API_KEY:
        return None
    try:
        resp = requests.get(
            TMDB_BASE + "/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title, "page": 1},
            timeout=6
        )
        results = resp.json().get("results", [])
        if not results:
            return None
        top      = results[0]
        poster   = (TMDB_IMG + top["poster_path"]) if top.get("poster_path") else None
        movie_id = top.get("id")
        return {
            "poster_url": poster,
            "genre_ids":  top.get("genre_ids", []),
            "movie_url":  f"https://www.themoviedb.org/movie/{movie_id}" if movie_id else None,
        }
    except Exception:
        return None


def _matches_genres(genre_ids, requested_genres):
    """
    Returns True if the movie's TMDB genre_ids overlap with the requested genres.
    If no genres were requested, always returns True.
    """
    if not requested_genres:
        return True
    wanted_ids = set(GENRE_MAP[g] for g in requested_genres if g in GENRE_MAP)
    if not wanted_ids:
        return True
    return bool(wanted_ids & set(genre_ids))


# ── RT URL ────────────────────────────────────────────────────────────────────

def _rt_url(title):
    """Best-effort Rotten Tomatoes URL. RT drops leading articles and uses underscores."""
    slug = title.lower().strip()
    # drop leading articles RT commonly omits
    for article in ("the ", "a ", "an "):
        if slug.startswith(article):
            slug = slug[len(article):]
            break
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug.strip())
    return "https://www.rottentomatoes.com/m/" + slug


def _movie_url(title, tmdb):
    """Return the best available movie link: TMDB page > RT slug."""
    if tmdb and tmdb.get("movie_url"):
        return tmdb["movie_url"]
    return _rt_url(title)


# ── ARCTIC SHIFT ──────────────────────────────────────────────────────────────

class ArcticUnavailable(Exception):
    """Raised when a Reddit data source is unreachable or down for maintenance."""


def _search_arctic(subreddit, title, limit=100):
    # NOTE: Arctic Shift's `title` param does a full-text title scan across all
    # history and times out (HTTP 422 "Timeout. Maybe slow down a bit").
    # `query` searches title+body via the fast index and returns quickly.
    try:
        resp = requests.get(
            ARCTIC_BASE + "/posts/search",
            params={"subreddit": subreddit, "query": title, "limit": limit},
            headers=HEADERS, timeout=12
        )
    except Exception:
        raise ArcticUnavailable("Could not reach the Reddit search service.")

    if resp.status_code == 200:
        return resp.json().get("data", [])
    # 503 = maintenance, 429/422 = rate limited — the backend is temporarily
    # unavailable, which is distinct from a genuinely empty result set.
    if resp.status_code in (429, 500, 502, 503, 504):
        raise ArcticUnavailable("Reddit search is temporarily unavailable (HTTP " + str(resp.status_code) + ").")
    return []


def _filter_comments(raw_comments):
    comments = []
    for c in raw_comments:
        body  = c.get("body", "").strip()
        score = c.get("score", 0)
        if body and len(body) > 10 and score > 0:
            comments.append({"body": body[:300], "upvotes": score})
    return sorted(comments, key=lambda x: -x["upvotes"])[:5]


def _get_comments(post_id, limit=40):
    # Arctic Shift rate-limits bursts ("Timeout. Maybe slow down a bit"), so
    # retry once with a short backoff before giving up on a post.
    for attempt in range(2):
        try:
            resp = requests.get(
                ARCTIC_BASE + "/comments/search",
                params={"link_id": post_id, "limit": limit, "sort": "desc"},
                headers=HEADERS, timeout=8
            )
            if resp.status_code != 200:
                time.sleep(0.5)
                continue
            return _filter_comments(resp.json().get("data", []))
        except Exception:
            time.sleep(0.5)
    return []


# ── PULLPUSH (fallback when Arctic Shift is down) ─────────────────────────────
# PullPush (api.pullpush.io) is the other volunteer-run Pushshift successor.
# Same data, Pushshift schema: posts have id/title, comments have body/score.
# Caveat: it archives comments soon after posting, so scores are often frozen
# at 1-2 rather than final vote counts — upvote ordering is weaker here.

def _search_pullpush(subreddit, title, limit=100):
    try:
        resp = requests.get(
            PULLPUSH_BASE + "/reddit/search/submission/",
            params={"subreddit": subreddit, "q": title, "size": limit},
            headers=HEADERS, timeout=12
        )
    except Exception:
        raise ArcticUnavailable("Could not reach the Reddit search service.")

    if resp.status_code == 200:
        try:
            return resp.json().get("data", [])
        except ValueError:
            # 200 but not JSON — almost certainly a Cloudflare challenge page.
            print("    PullPush returned non-JSON 200: " + resp.text[:200].replace("\n", " "))
            raise ArcticUnavailable("Reddit search is temporarily unavailable (upstream returned an invalid response).")

    # Surface the real response in the server logs — a silent empty result and
    # an upstream block (e.g. Cloudflare 403 on datacenter IPs) look identical
    # to users otherwise.
    print("    PullPush HTTP " + str(resp.status_code) + ": " + resp.text[:200].replace("\n", " "))
    if resp.status_code in (403, 429, 500, 502, 503, 504):
        raise ArcticUnavailable("Reddit search is temporarily unavailable (HTTP " + str(resp.status_code) + ").")
    return []


def _get_comments_pullpush(post_id, limit=40):
    for attempt in range(2):
        try:
            resp = requests.get(
                PULLPUSH_BASE + "/reddit/search/comment/",
                params={"link_id": post_id, "size": limit},
                headers=HEADERS, timeout=8
            )
            if resp.status_code != 200:
                print("    PullPush comments HTTP " + str(resp.status_code) + " for post " + str(post_id))
                time.sleep(0.5)
                continue
            return _filter_comments(resp.json().get("data", []))
        except Exception:
            time.sleep(0.5)
    return []


# ── GROQ EXTRACTION ───────────────────────────────────────────────────────────

def _extract_recs_llm(post_title, comments):
    comments_text = "\n".join(
        "[" + str(c["upvotes"]) + " upvotes] " + c["body"] for c in comments
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "POST: " + post_title + "\n\nCOMMENTS:\n" + comments_text}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        return json.loads(response.choices[0].message.content).get("recommendations", [])
    except Exception:
        return []


# ── PER-MOVIE SEARCH ──────────────────────────────────────────────────────────

MIN_RECS             = 10  # keep searching until we have at least this many recs
MAX_POSTS_PER_SUB    = 30  # posts fetched per subreddit per movie
MAX_POSTS_TO_PROCESS = 30  # hard safety cap on LLM-processed posts (bounds runtime)


def _search_one_movie(source_title, genres, initial_seen, min_recs=MIN_RECS):
    """
    Searches both subreddits for one source movie and keeps going until it has
    at least `min_recs` recommendations (or it genuinely runs out of posts /
    hits the MAX_POSTS_TO_PROCESS safety cap).

    Uses Arctic Shift first; if it's down, falls back to PullPush for the rest
    of the search. Raises ArcticUnavailable only if both sources are down.

    initial_seen  — set of lowercase titles to skip (the source movie itself).

    Arctic Shift and Groq rate-limit bursty access, so this runs SEQUENTIALLY
    with a small sleep between comment fetches — reliable over fast. A single
    movie search comfortably fits the server's request timeout.
    Returns a list of rec dicts: {title, poster_url, rt_url, upvotes}.
    """
    seen      = set(initial_seen)  # local copy
    recs      = {}
    processed = 0
    use_pullpush = False  # flips permanently once Arctic Shift fails

    print("  Searching Reddit for: " + source_title + " (target " + str(min_recs) + " recs)")

    for subreddit in SUBREDDITS:
        if len(recs) >= min_recs or processed >= MAX_POSTS_TO_PROCESS:
            break

        if not use_pullpush:
            try:
                posts = _search_arctic(subreddit, source_title, limit=MAX_POSTS_PER_SUB)
            except ArcticUnavailable:
                print("    Arctic Shift unavailable — falling back to PullPush")
                use_pullpush = True
        if use_pullpush:
            # If PullPush is also down, this raises ArcticUnavailable, which
            # propagates to the route and becomes a 503 for the user.
            posts = _search_pullpush(subreddit, source_title, limit=MAX_POSTS_PER_SUB)

        print("    [" + subreddit + "] " + str(len(posts)) + " posts for " + source_title
              + (" (PullPush)" if use_pullpush else ""))

        for post in posts:
            if len(recs) >= min_recs or processed >= MAX_POSTS_TO_PROCESS:
                break

            if use_pullpush:
                comments = _get_comments_pullpush(post.get("id", ""))
            else:
                comments = _get_comments(post.get("id", ""))
            time.sleep(0.2)  # be gentle with the archive's rate limiter
            if not comments:
                continue

            processed += 1
            for rec in _extract_recs_llm(post.get("title", ""), comments):
                rec_title = rec.get("title", "").strip()
                if not rec_title or rec_title.lower() in seen:
                    continue

                tmdb = _tmdb_data(rec_title)  # TMDB tolerates high rate; no sleep needed

                if tmdb and not _matches_genres(tmdb["genre_ids"], genres):
                    continue

                seen.add(rec_title.lower())
                recs[rec_title] = {
                    "title":      rec_title,
                    "poster_url": tmdb["poster_url"] if tmdb else None,
                    "rt_url":     _movie_url(rec_title, tmdb),
                    "upvotes":    rec.get("upvotes", 0),
                }

    print("  Done [" + source_title + "]: " + str(len(recs)) + " recs from " + str(processed) + " posts")
    return sorted(recs.values(), key=lambda x: -x["upvotes"])