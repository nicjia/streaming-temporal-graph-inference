"""
Resolve extracted entity strings onto graph vertices, in two tiers.

Anchored nodes are listed equities, resolved by ticker; a mis-resolution here
attaches a shock to the wrong instrument, so the resolution budget goes here.
Unanchored nodes are everything else the extractor names (private companies,
people, regulators, products) -- intermediaries that are never traded, resolved
by normalised string identity, where a split is cheap.

The filer itself is resolved by construction: an 8-K arrives with a CIK and the
SEC publishes CIK -> ticker, so only entities mentioned inside the text need
matching.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Corporate form suffixes, stripped before matching. "Acme Corp", "Acme Corp.",
# "Acme Corporation" and "Acme, Inc." are one company and four strings.
_SUFFIXES = [
    "incorporated", "corporation", "company", "holdings", "holding",
    "group", "limited", "partners", "partnership", "trust",
    "inc", "corp", "co", "llc", "llp", "lp", "ltd", "plc", "sa", "nv",
    "ag", "se", "spa", "ab", "as", "oyj", "pte", "bhd", "kk",
]
_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(_SUFFIXES) + r")\b\.?\s*$", re.IGNORECASE)

_PUNCT = re.compile(r"[^\w\s&]")
_SPACE = re.compile(r"\s+")

# Self-reference forms an 8-K uses for its own filer. These must never become
# vertices: "the Company" appears in most filings in the corpus, and left
# unresolved it collapses every registrant into one enormous spurious hub.
_SELF_REFERENCE = frozenset({
    "the company", "company", "the registrant", "registrant", "the issuer",
    "issuer", "the corporation", "we", "us", "our company", "the firm",
    "the parent", "the borrower", "the filer",
})


def normalize(name: str) -> str:
    """
    Canonical form for matching: lowercase, punctuation dropped, corporate
    suffixes removed, whitespace collapsed.

    Suffix stripping repeats because names stack them -- "Acme Holdings Inc"
    needs two passes, and "Acme Group Holdings Ltd" three.
    """
    text = _PUNCT.sub(" ", str(name).lower())
    text = _SPACE.sub(" ", text).strip()
    for _ in range(3):
        stripped = _SUFFIX_RE.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    return text


# Runs of capitalised tokens, which is what a company name looks like in prose.
# Allows internal lowercase connectors ("Bank of America") and punctuation
# ("1-800-FLOWERS.COM, Inc.").
_CAPITALISED_RUN = re.compile(
    r"\b[A-Z][\w&.'-]*(?:\s+(?:of|and|for|the|de|van|von)\s+|\s+)?"
    r"(?:[A-Z][\w&.'-]*(?:\s+(?:of|and|for|the|de|van|von)\s+|\s+)?){0,4}")


def name_candidates(text: str, max_words: int = 5, min_length: int = 5) -> set[str]:
    """
    Normalised capitalised phrases from `text` that might name a company.

    Progressively shorter prefixes of each run are emitted, because "Siemens AG
    announced" should yield both "siemens ag" and "siemens" -- the index may
    hold either form.
    """
    found = set()
    for match in _CAPITALISED_RUN.finditer(text):
        phrase = match.group(0).strip(" .,;:")
        if len(phrase) < 4:
            continue
        words = phrase.split()
        for size in range(min(len(words), max_words), 0, -1):
            found.add(normalize(" ".join(words[:size])))
    return {f for f in found if len(f) >= min_length}


def build_name_index(tickers, min_length: int = 5) -> dict[str, str]:
    """
    Normalised company name -> ticker, with ambiguous names dropped.

    Separate from EntityResolver because this one gets serialised and shipped to
    machines that cannot import pandas.
    """
    index: dict[str, str] = {}
    collisions = set()
    for row in tickers.itertuples(index=False):
        key = normalize(row.name)
        if len(key) < min_length:
            continue
        if key in index and index[key] != row.ticker:
            collisions.add(key)
        index[key] = row.ticker
    for key in collisions:
        index.pop(key, None)
    return index


def find_listed(text: str, index: dict[str, str], exclude: str | None = None,
                hubs: frozenset | set = frozenset()) -> list[str]:
    """
    Listed companies named in `text`, as canonical company names.

    `hubs` drops names that appear in a large share of every filing --
    exchanges, transfer agents, index providers. Unfiltered, every registrant
    writes "listed on the Nasdaq Global Market", which linked the whole universe
    through NDAQ and made a one-artifact graph look dense.
    """
    hits = {}
    for phrase in name_candidates(text):
        ticker = index.get(phrase)
        if ticker and ticker != exclude and ticker not in hubs:
            hits[ticker] = phrase
    return sorted(hits)


class EntityResolver:
    """
    Maps entity strings to (kind, key) where kind is "anchored" or "unanchored".

    Ambiguous normalised names -- two different tickers reducing to the same
    string -- are dropped from the anchored index rather than resolved
    arbitrarily. A name that could be either of two listed companies is not
    evidence about either, and guessing would put real shocks on wrong tickers.
    """

    def __init__(self, tickers, min_length: int = 4):
        self.min_length = min_length
        index: dict[str, set] = {}
        for row in tickers.itertuples(index=False):
            key = normalize(row.name)
            if len(key) < min_length:
                continue
            index.setdefault(key, set()).add(row.ticker)

        self.ambiguous = {k for k, v in index.items() if len(v) > 1}
        self.by_name = {k: next(iter(v)) for k, v in index.items()
                        if len(v) == 1}
        self.unanchored: dict[str, int] = {}
        self.stats = {"anchored": 0, "unanchored": 0, "self_reference": 0,
                      "ambiguous": 0, "too_short": 0}

    def resolve(self, name: str, filer_ticker: str | None = None):
        """
        Returns (kind, key) or None when the string carries no entity.

        `filer_ticker` resolves self-reference. An 8-K written from inside the
        company says "the Company" throughout, and that phrase means the filer
        -- so with the filer known it becomes a real anchored edge, and without
        it the mention is dropped rather than becoming a shared vertex.
        """
        raw = str(name).strip().lower()
        if raw in _SELF_REFERENCE:
            self.stats["self_reference"] += 1
            return ("anchored", filer_ticker) if filer_ticker else None

        key = normalize(name)
        if len(key) < self.min_length:
            self.stats["too_short"] += 1
            return None
        if key in self.ambiguous:
            self.stats["ambiguous"] += 1
            return ("unanchored", key)
        if key in self.by_name:
            self.stats["anchored"] += 1
            return ("anchored", self.by_name[key])

        self.stats["unanchored"] += 1
        return ("unanchored", key)


class VertexTable:
    """
    Dense uint32 ids for the C++ engine, assigned in first-seen order.

    Anchored vertices are allocated first and kept in a separate namespace, so
    a graph can be restricted to tradeable nodes by an id comparison rather than
    a lookup -- which matters when the restriction happens inside a sampling
    loop run millions of times.
    """

    def __init__(self):
        self.anchored: dict[str, int] = {}
        self.unanchored: dict[str, int] = {}
        self._next = 0

    def id_for(self, kind: str, key: str) -> int:
        table = self.anchored if kind == "anchored" else self.unanchored
        if key not in table:
            table[key] = self._next
            self._next += 1
        return table[key]

    def __len__(self) -> int:
        return self._next

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "anchored": self.anchored,
            "unanchored": self.unanchored,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "VertexTable":
        payload = json.loads(Path(path).read_text())
        table = cls()
        table.anchored = payload["anchored"]
        table.unanchored = payload["unanchored"]
        table._next = max([*table.anchored.values(),
                           *table.unanchored.values(), -1]) + 1
        return table
