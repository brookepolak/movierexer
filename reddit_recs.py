"""
reddit_recs.py
--------------
Serves movie recommendations from the local cache built by build_db.py
(recs_cache: a graph of movie-to-movie edges mined from Reddit threads,
weighted by upvotes). No live Reddit APIs at request time — TMDB is called
only to decorate results with posters, genres, and links.

Requires in .env:
    TMDB_API_KEY   (optional — posters/genre filtering degrade without it)
"""

import os
import re
import sqlite3

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# First hit wins: explicit override → full local archive → the slim,
# committed export (what production serves from).
DB_CANDIDATES = [
    os.environ.get("MOVIEREXER_DB"),
    os.path.join(BASE_DIR, "data", "movies.db"),
    os.path.join(BASE_DIR, "recs.db"),
]

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE    = "https://api.themoviedb.org/3"
TMDB_IMG     = "https://image.tmdb.org/t/p/w300"

if not TMDB_API_KEY:
    print("⚠️  TMDB_API_KEY not set — posters will not load. Get a free key at themoviedb.org/settings/api")

MAX_RECS       = 24   # results returned per query
MAX_CANDIDATES = 80   # ranked candidates considered before giving up (genre
                      # filtering can reject many, but TMDB calls aren't free)

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


def _norm_title(title):
    """Normalization key so 'The Matrix', 'the matrix' and 'Matrix!' merge.
    Must stay identical to the keying used when building the cache
    (build_db.py imports this function)."""
    key = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in title.lower())
    key = " ".join(key.split())
    if key.startswith("the "):
        key = key[4:]
    return key


# ── CACHE ─────────────────────────────────────────────────────────────────────

class CacheUnavailable(Exception):
    """Raised when the local recommendations DB is missing or empty."""


def _connect():
    for path in DB_CANDIDATES:
        if not path or not os.path.exists(path):
            continue
        con = sqlite3.connect(path)
        try:
            n = con.execute("SELECT COUNT(*) FROM recs_cache").fetchone()[0]
        except sqlite3.OperationalError:
            n = 0
        if n:
            return con
        con.close()
    raise CacheUnavailable("The recommendations cache is missing or empty — "
                           "build it with build_db.py (ingest + extract + export).")


def get_recs(movies, genres):
    """
    Queries the cache for any number of input movies and merges the results.
    A candidate recommended by several of the user's movies ranks above one
    recommended by a single movie; ties break on summed edge weight (upvotes).
    Input movies themselves are excluded from the results.

    Returns {"recs": [...], "found": [movies with cache hits],
             "missing": [movies without]}.
    """
    con = _connect()
    input_keys = set()
    cleaned    = []
    for movie in movies:
        key = _norm_title(movie)
        if key and key not in input_keys:
            input_keys.add(key)
            cleaned.append((key, movie))

    found, missing = [], []
    agg = {}
    for key, movie in cleaned:
        rows = con.execute(
            """SELECT rec_key, rec_title, total_upvotes FROM recs_cache
               WHERE source_key = ?""", (key,)
        ).fetchall()
        (found if rows else missing).append(movie)
        for rec_key, rec_title, total_upvotes in rows:
            if rec_key in input_keys:
                continue
            entry = agg.setdefault(rec_key, {"title": rec_title,
                                             "weight": 0, "matches": 0})
            entry["weight"]  += total_upvotes
            entry["matches"] += 1
    con.close()

    ranked = sorted(agg.values(), key=lambda e: (-e["matches"], -e["weight"]))

    recs = []
    for entry in ranked[:MAX_CANDIDATES]:
        if len(recs) >= MAX_RECS:
            break
        tmdb = _tmdb_data(entry["title"])
        if tmdb and not _matches_genres(tmdb["genre_ids"], genres):
            continue
        recs.append({
            "title":      entry["title"],
            "poster_url": tmdb["poster_url"] if tmdb else None,
            "rt_url":     _movie_url(entry["title"], tmdb),
            "upvotes":    entry["weight"],
            "matches":    entry["matches"],
        })
    return {"recs": recs, "found": found, "missing": missing}


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


# ── MOVIE LINKS ───────────────────────────────────────────────────────────────

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
