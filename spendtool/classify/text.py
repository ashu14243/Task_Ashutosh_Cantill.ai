"""Tokenisation shared by all classifiers."""
from __future__ import annotations

import re
from functools import lru_cache

from ..normalize import fix_ocr_token
from .lexicon import ABBR, BROADER, STOPWORDS, SYNONYMS, stem, strip_accents

_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_DIGIT_ALPHA = re.compile(r"(?<=\d)(?=[A-Za-z]{2,})|(?<=[A-Za-z]{2})(?=\d)")


KNOWN: set[str] = set(SYNONYMS) | {b for v in BROADER.values() for b in v} | {t for v in ABBR.values() for t in v}
_SUFFIXES = ("ation", "ition", "ing", "ers", "er", "ed", "d", "al", "ly", "ment", "ive")


def register_known(words) -> None:
    """Add words (e.g. the taxonomy's own vocabulary) that inflected forms should be reduced to."""
    new = set(words) - KNOWN
    if new:
        KNOWN.update(new)
        tokens.cache_clear()


def _lemma(w: str) -> str:
    s = stem(w)
    if s in KNOWN or len(s) <= 4:
        return s
    for suf in _SUFFIXES:
        if s.endswith(suf) and len(s) - len(suf) >= 3:
            base = s[: -len(suf)]
            for cand in (base, base + "e", base + "y"):
                if cand in KNOWN:
                    return cand
    return s


@lru_cache(maxsize=500000)
def tokens(text: str, extra_stop: frozenset = frozenset()) -> tuple[str, ...]:
    """'VehicleMaintRepairSV; P0WDER 20MM' -> ('vehicle','maintenance','repair','service','powder')"""
    if not text:
        return ()
    s = strip_accents(str(text))
    s = _CAMEL.sub(" ", s)
    out = []
    for raw in _SPLIT.split(s):
        if not raw:
            continue
        t = fix_ocr_token(raw.upper()).lower()
        for part in _DIGIT_ALPHA.sub(" ", t).split():
            if part.isdigit() or len(part) < 2:
                continue
            if part in ABBR:
                out.extend(ABBR[part])
                continue
            if part in STOPWORDS or part in extra_stop:
                continue
            lem = _lemma(part)
            if lem not in KNOWN and len(part) >= 6 and part.isalpha():
                split = _split_compound(part)
                if split:
                    out.extend(split)
                    continue
            out.append(lem)
    return tuple(out)


def _resolve(w: str):
    if w in ABBR:
        return ABBR[w]
    lem = _lemma(w)
    return [lem] if lem in KNOWN else None


def _split_compound(w: str):
    """'rofbltr' -> ['roof', 'bolter'], 'stonedust' -> ['stone', 'dust'] when both halves are known."""
    for i in range(len(w) - 3, 2, -1):
        a, b = _resolve(w[:i]), _resolve(w[i:])
        if a and b:
            return a + b
    return None


def expand(tok: str) -> dict[str, float]:
    """A token plus its synonyms (0.7) and broader terms (0.3)."""
    d = {tok: 1.0}
    for s in sorted(SYNONYMS.get(tok, ())):
        if s != tok:
            d[s] = max(d.get(s, 0), 0.7)
    for b in sorted(BROADER.get(tok, ())):
        d[b] = max(d.get(b, 0), 0.3)
    return d
