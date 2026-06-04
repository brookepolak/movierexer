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
import queue
import threading
import concurrent.futures
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

def _search_arctic(subreddit, title, limit=100):
    try:
        resp = requests.get(
            ARCTIC_BASE + "/posts/search",
            params={"subreddit": subreddit, "title": title, "limit": limit, "sort": "desc"},
            headers=HEADERS, timeout=8
        )
        return resp.json().get("data", []) if resp.status_code == 200 else []
    except Exception:
        return []


def _get_comments(post_id, limit=100):
    try:
        resp = requests.get(
            ARCTIC_BASE + "/comments/search",
            params={"link_id": post_id, "limit": limit, "sort": "desc"},
            headers=HEADERS, timeout=8
        )
        if resp.status_code != 200:
            return []
        comments = []
        for c in resp.json().get("data", []):
            body  = c.get("body", "").strip()
            score = c.get("score", 0)
            if body and len(body) > 10 and score > 0:
                comments.append({"body": body[:300], "upvotes": score})
        return sorted(comments, key=lambda x: -x["upvotes"])[:5]
    except Exception:
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

MAX_POSTS_PER_SUB = 15   # posts fetched per subreddit per movie
MAX_RECS_PER_MOVIE = 25  # stop early once we have this many unique recs


def _search_one_movie(source_title, genres, initial_seen):
    """
    Searches both subreddits for one source movie.
    initial_seen  — set of lowercase titles to skip (source movies).
    Returns a list of rec dicts: {title, poster_url, rt_url, upvotes}.
    Caps at MAX_POSTS_PER_SUB posts per subreddit and MAX_RECS_PER_MOVIE
    unique results so each request completes within a reasonable time.
    """
    seen = set(initial_seen)  # local copy — no cross-thread mutation
    recs = {}

    print("  Searching Reddit for: " + source_title)

    for subreddit in SUBREDDITS:
        posts = _search_arctic(subreddit, source_title, limit=MAX_POSTS_PER_SUB)
        print("    [" + subreddit + "] " + str(len(posts)) + " posts for " + source_title)
        time.sleep(0.2)

        for post in posts:
            if len(recs) >= MAX_RECS_PER_MOVIE:
                break

            comments = _get_comments(post.get("id", ""))
            time.sleep(0.2)
            if not comments:
                continue

            extracted = _extract_recs_llm(post.get("title", ""), comments)

            for rec in extracted:
                if len(recs) >= MAX_RECS_PER_MOVIE:
                    break

                rec_title = rec.get("title", "").strip()
                if not rec_title or rec_title.lower() in seen:
                    continue

                tmdb = _tmdb_data(rec_title)
                time.sleep(0.15)

                if tmdb and not _matches_genres(tmdb["genre_ids"], genres):
                    continue

                seen.add(rec_title.lower())
                upvotes = rec.get("upvotes", 0)

                if rec_title in recs:
                    recs[rec_title]["upvotes"] += upvotes
                else:
                    recs[rec_title] = {
                        "title":      rec_title,
                        "poster_url": tmdb["poster_url"] if tmdb else None,
                        "rt_url":     _movie_url(rec_title, tmdb),
                        "upvotes":    upvotes,
                    }

        if len(recs) >= MAX_RECS_PER_MOVIE:
            break

    print("  Done [" + source_title + "]: " + str(len(recs)) + " recs found")
    return list(recs.values())


# ── MAIN ENTRY POINTS ─────────────────────────────────────────────────────────

def get_recommendations(movie_titles, genres=None, max_recs=120):
    """
    Searches Reddit for every movie in parallel and returns a merged, sorted list.
    """
    genres       = genres or []
    initial_seen = set(t.lower() for t in movie_titles)
    merged       = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(movie_titles), 4)) as ex:
        futures = {ex.submit(_search_one_movie, t, genres, initial_seen): t
                   for t in movie_titles}
        for future in concurrent.futures.as_completed(futures):
            for rec in future.result():
                key = rec["title"].lower()
                if key in merged:
                    merged[key]["upvotes"] += rec["upvotes"]
                else:
                    merged[key] = rec

    sorted_recs = sorted(merged.values(), key=lambda x: -x["upvotes"])
    return sorted_recs[:max_recs]


def get_recommendations_streaming(movie_titles, genres=None):
    """
    Generator. Searches every movie in parallel; yields (movie_title, recs_list)
    as each search finishes — callers see results arrive one movie at a time.
    """
    genres       = genres or []
    initial_seen = set(t.lower() for t in movie_titles)
    result_queue = queue.Queue()

    def _worker(title):
        recs = _search_one_movie(title, genres, initial_seen)
        result_queue.put((title, recs))

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(movie_titles), 4)) as ex:
        for title in movie_titles:
            ex.submit(_worker, title)

        for _ in movie_titles:
            try:
                title, recs = result_queue.get(timeout=300)
                yield title, recs
            except queue.Empty:
                break