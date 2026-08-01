"""
Player-name cleaning and matching helpers.

Data sources disagree on names: nflverse/ESPN carry suffixes ("Michael
Pittman Jr.", "Odell Beckham Jr.", "Marvin Harrison Jr."), PrizePicks/Underdog
sometimes drop them. All matching goes through normalize_name; last-name
fallback refuses ambiguous matches rather than blending two players' stats
(critical for NFL — many rosters carry Jr./Sr./II/III variants of the same
surname on the same team, e.g. "Odell Beckham Jr." vs. unrelated "Beckham").
"""
import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_ESCAPED_BYTES = re.compile(r"\\x[0-9a-fA-F]{2}")


def clean_name(raw) -> str:
    """Repair mojibake from scraped sources and strip whitespace."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if _ESCAPED_BYTES.search(s):
        # Literal backslash-x sequences in the text (scrape artifact)
        try:
            s = (s.encode("latin-1", "backslashreplace")
                  .decode("unicode_escape")
                  .encode("latin-1")
                  .decode("utf-8"))
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
    else:
        # UTF-8 bytes decoded as latin-1
        try:
            fixed = s.encode("latin-1").decode("utf-8")
            if fixed != s:
                s = fixed
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
    return s.strip()


def normalize_name(raw) -> str:
    """Accent-stripped, lowercase, suffix-free key for matching."""
    s = clean_name(raw)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace(".", " ").replace(",", " ")
    tokens = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def make_lookup(df, name_col: str = "player_name"):
    """
    Build a name → row (pandas Series) lookup over a DataFrame.
    Exact normalized match first; last-name fallback only when it is
    unambiguous (exactly one player with that last name). Returns a
    function target_name → row | None.
    """
    if df is None or len(df) == 0 or name_col not in df.columns:
        return lambda target: None

    rows: dict = {}
    by_last: dict = {}
    for _, row in df.iterrows():
        key = normalize_name(row[name_col])
        if not key:
            continue
        rows.setdefault(key, row)
        by_last.setdefault(key.split()[-1], set()).add(key)

    def lookup(target):
        k = normalize_name(target)
        if not k:
            return None
        if k in rows:
            return rows[k]
        cands = by_last.get(k.split()[-1], set())
        if len(cands) == 1:
            return rows[next(iter(cands))]
        return None

    return lookup
