"""
8-K ingestion from SEC EDGAR.

8-K filings arrive with a CIK attached, and the SEC publishes CIK -> ticker, so
the filer is resolved by construction and only entities mentioned inside the
text need matching. The SEC requires a User-Agent identifying the requester and
rate-limits to 10 requests/second; both are handled below, and the contact
string is supplied by the caller rather than baked in.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

SEC_ARCHIVES = "https://www.sec.gov/Archives"
SEC_FILES = "https://www.sec.gov/files"

# The SEC's published request ceiling. Exceeding it earns a block on the source
# IP rather than a 429, so the limiter is not optional politeness.
MAX_REQUESTS_PER_SECOND = 10.0

# ITEM_CODES lives in extraction_schema so the cluster can import it
# without pulling in pandas through this module.
from .extraction_schema import ITEM_CODES  # noqa: F401,E402

# Items whose content is procedural rather than economic. Excluded by default
# from extraction, not from the index: 9.01 is an exhibit list and 5.03 a bylaw
# amendment, and spending model inference on either is spending it on nothing.
BOILERPLATE_ITEMS = frozenset({"5.03", "5.04", "5.05", "5.08", "9.01"})


def _user_agent() -> str:
    """
    SEC requires a User-Agent naming the requester with a contact address.

    Read from SEC_USER_AGENT, via .env like the WRDS credentials, rather than
    hardcoded. The header is transmitted to a third party on every request, so
    which address goes in it is a decision recorded in an ignored file rather
    than committed to the repository.
    """
    agent = os.getenv("SEC_USER_AGENT")
    if not agent:
        try:
            from dotenv import load_dotenv
            load_dotenv(".env")
            agent = os.getenv("SEC_USER_AGENT")
        except ImportError:
            pass
    if not agent:
        raise RuntimeError(
            "SEC requires a User-Agent identifying the requester. Set "
            "SEC_USER_AGENT, e.g.\n"
            "    export SEC_USER_AGENT='Your Name your@email.edu'\n"
            "See https://www.sec.gov/os/webmaster-faq#developers")
    return agent


class _RateLimiter:
    """Spaces requests to stay under the published ceiling."""

    def __init__(self, per_second: float = MAX_REQUESTS_PER_SECOND):
        self.interval = 1.0 / per_second
        self.last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self.last
        if gap < self.interval:
            time.sleep(self.interval - gap)
        self.last = time.monotonic()


_LIMITER = _RateLimiter()


def _get(url: str, cache: Path | None = None, binary: bool = False):
    """
    Fetch with rate limiting and an on-disk cache.

    The cache is not a convenience. Extraction is re-run every time the relation
    taxonomy changes, and re-downloading a corpus of hundreds of thousands of
    filings on each iteration would dominate both wall-clock time and the SEC's
    patience.
    """
    if cache is not None and cache.exists():
        return cache.read_bytes() if binary else cache.read_text(
            encoding="utf-8", errors="replace")

    _LIMITER.wait()
    response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=60)
    response.raise_for_status()
    payload = response.content if binary else response.text

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        if binary:
            cache.write_bytes(payload)
        else:
            cache.write_text(payload, encoding="utf-8")
    return payload


def company_tickers(data_dir: str = "data/edgar") -> pd.DataFrame:
    """
    The anchored node universe: every CIK the SEC maps to a listed ticker.

    This is the file that makes 8-Ks worth starting with. Roughly 10,000 rows of
    (cik, ticker, name), maintained by the SEC, free, and the authoritative
    resolution of filer to tradeable instrument.

    Note what it does *not* solve: a company that changed ticker or was acquired
    appears under its current mapping only, so a point-in-time study still needs
    a historical link table (CRSP's is the usual answer). For a first pass over
    recent filings the current mapping is adequate, and the limitation is worth
    knowing before it silently biases a backtest toward survivors.
    """
    cache = Path(data_dir) / "company_tickers.json"
    payload = _get(f"{SEC_FILES}/company_tickers.json", cache=cache)
    frame = pd.read_json(io.StringIO(payload), orient="index")
    frame = frame.rename(columns={"cik_str": "cik", "title": "name"})
    frame["cik"] = frame["cik"].astype(int)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return frame[["cik", "ticker", "name"]].drop_duplicates("cik")


def form_index(year: int, quarter: int, form_type: str = "8-K",
               data_dir: str = "data/edgar") -> pd.DataFrame:
    """
    Every filing of one form type in one quarter, from EDGAR's full index.

    Returns (cik, company, form_type, date_filed, path). `path` is relative to
    SEC_ARCHIVES and is what fetch_filing takes.
    """
    cache = Path(data_dir) / "index" / f"{year}Q{quarter}.idx"
    raw = _get(f"{SEC_ARCHIVES}/edgar/full-index/{year}/QTR{quarter}/form.idx",
               cache=cache)

    rows = []
    # Parsed by splitting on runs of two or more spaces, not by byte offset.
    # form.idx *looks* fixed-width, but its header row and its data rows sit at
    # different column positions -- taking the documented header offsets shifts
    # every field and silently reads dates one day early with filename
    # fragments attached. Single-space company names ("1 800 FLOWERS COM INC")
    # survive a 2+ space split, which is why it is the right delimiter.
    #
    # Fields are then anchored from both ends rather than by index, so a name
    # that does contain a double space costs at most a mangled company string
    # and never a misparsed CIK, date or path.
    body = raw.split("-" * 20, 1)[-1]
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 5 or parts[0] != form_type:
            continue
        rows.append({
            "form_type": parts[0],
            "company": " ".join(parts[1:-3]).strip(),
            "cik": parts[-3],
            "date_filed": parts[-2],
            "path": parts[-1],
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64")
    frame["date_filed"] = pd.to_datetime(frame["date_filed"], errors="coerce")
    return frame.dropna(subset=["cik", "date_filed"]).reset_index(drop=True)


def fetch_filing(path: str, data_dir: str = "data/edgar") -> str:
    """Raw filing text (SGML wrapper included) for a path from form_index."""
    cache = Path(data_dir) / "filings" / path.replace("/", "_")
    return _get(f"{SEC_ARCHIVES}/{path}", cache=cache)


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------

_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>",
                           re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")

# Everything after this heading is exhibit boilerplate: press release
# attachments, certifications, XBRL. The body of an 8-K is what precedes it.
#
# A word boundary rather than trailing punctuation. Signature blocks are
# routinely a bare "SIGNATURES" on its own line, so requiring a following
# period or colon silently failed to cut the most common case of all.
_EXHIBIT_CUT = re.compile(
    r"\n\s*(?:item\s*9\.01|exhibit\s+index|signatures?)\b",
    re.IGNORECASE)

# Minimum body length before the cut is honoured. Some filings place a
# signature block near the top, and cutting there would leave nothing to
# extract from -- but the threshold has to stay low, because a single-item 8-K
# reporting one fact is legitimately short.
_MIN_BODY_CHARS = 200

_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'",
             "nbsp": " ", "ldquo": '"', "rdquo": '"', "lsquo": "'",
             "rsquo": "'", "mdash": "-", "ndash": "-", "hellip": "..."}


def _unescape(match: re.Match) -> str:
    body = match.group(1)
    if body.startswith("#x") or body.startswith("#X"):
        try:
            return chr(int(body[2:], 16))
        except ValueError:
            return " "
    if body.startswith("#"):
        try:
            return chr(int(body[1:]))
        except ValueError:
            return " "
    return _ENTITIES.get(body.lower(), " ")


def clean_text(raw: str, drop_exhibits: bool = True) -> str:
    """
    Filing markup down to readable prose.

    Written by hand rather than pulled from an HTML library because EDGAR
    documents are not reliably well-formed -- they range from 1990s SGML to
    Word-exported HTML with unclosed tags -- and a strict parser rejects a
    meaningful fraction of the corpus. Regex stripping degrades gracefully on
    exactly the malformed documents a parser refuses.

    `drop_exhibits` cuts at the exhibit index. Worth doing before the text
    reaches a model: exhibits are frequently longer than the filing body and are
    mostly certifications and XBRL, so keeping them would spend most of the
    inference budget on boilerplate.
    """
    text = _SCRIPT_STYLE.sub(" ", raw)
    text = _TAG.sub(" ", text)
    text = _ENTITY.sub(_unescape, text)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    text = text.strip()

    if drop_exhibits:
        cut = _EXHIBIT_CUT.search(text)
        if cut and cut.start() >= _MIN_BODY_CHARS:
            text = text[:cut.start()].rstrip()
    return text


_ITEM_PATTERN = re.compile(r"item\s+(\d\.\d{2})", re.IGNORECASE)
_ITEM_INFORMATION = re.compile(r"^\s*ITEM INFORMATION:\s*(.+?)\s*$",
                               re.IGNORECASE | re.MULTILINE)
_DESCRIPTION_TO_CODE = {v.lower(): k for k, v in ITEM_CODES.items()}

_DOCUMENT = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
_DOC_TYPE = re.compile(r"<TYPE>\s*([^\s<]+)", re.IGNORECASE)
_HEADER_END = re.compile(r"</SEC-HEADER>", re.IGNORECASE)


def header_items(raw: str) -> list[str]:
    """
    Item codes from the submission's own SGML header.

    The header carries one `ITEM INFORMATION:` line per reported item, written
    by EDGAR rather than by the filer's word processor. That makes it strictly
    better than pattern-matching numbered headings in the body, which vary in
    formatting and appear again inside exhibits and cross-references.

    Descriptions are mapped back through ITEM_CODES, so a wording EDGAR uses
    that is not in that table yields nothing here and the body scan covers it.
    """
    codes = []
    for match in _ITEM_INFORMATION.finditer(raw):
        code = _DESCRIPTION_TO_CODE.get(match.group(1).strip().lower())
        if code and code not in codes:
            codes.append(code)
    return codes


_POINTER = re.compile(
    r"(?:incorporated herein by reference|furnished (?:herewith|as exhibit)|"
    r"attached (?:hereto )?as exhibit|a copy of the press release)",
    re.IGNORECASE)
_EX99 = re.compile(r"^EX-99", re.IGNORECASE)


def split_documents(raw: str) -> list[tuple[str, str]]:
    """[(type, body)] for every <DOCUMENT> in the submission, in order."""
    out = []
    for block in _DOCUMENT.finditer(raw):
        body = block.group(1)
        kind = _DOC_TYPE.search(body)
        out.append((kind.group(1).upper() if kind else "", body))
    return out


def is_pointer(text: str, max_chars: int = 2500) -> bool:
    """
    Does this body just point at an exhibit instead of containing the event?

    The single most important property of this corpus, and the one that decides
    whether extraction has anything to work with. A typical item 2.02 body reads
    in full: "On May 2, 2024, the Company issued a press release announcing its
    financial results. A copy is included as Exhibit 99.1 and incorporated
    herein by reference." Measured over 2024Q2, **82.5% of item 2.02 filings and
    51.6% of the whole corpus are pointers like that** -- so treating the 8-K
    document as the content silently throws away half the events and most of the
    economically interesting ones.

    Both conditions are required. Length alone catches genuinely terse filings
    that do contain their event, and the reference phrase alone appears in long
    filings that also say something.
    """
    return len(text) < max_chars and bool(_POINTER.search(text))


def filing_body(raw: str, form_type: str = "8-K",
                max_exhibit_chars: int = 12000) -> str:
    """
    The text an extractor should actually read.

    Normally the primary document. When that is only a pointer, the EX-99
    exhibits are appended, because that is where the earnings numbers, the
    guidance and the customer commentary live.

    Exhibits are capped rather than taken whole: a press release opens with the
    narrative that carries the relationships and then runs into pages of
    tabulated financials, which cost tokens and say nothing a relation extractor
    can use. Twelve thousand characters keeps the narrative and cuts the tables.
    """
    # item_body first, then the pointer test. Testing the uncut document gets
    # the answer wrong: the cover page adds ~3 kB of registrant boilerplate to
    # every filing, which pushes a 325-character pointer body over any sane
    # length threshold and silently disables exhibit inclusion entirely.
    primary = item_body(clean_text(primary_document(raw, form_type)))
    if not is_pointer(primary):
        return primary

    parts = [primary]
    budget = max_exhibit_chars
    for kind, body in split_documents(raw):
        if budget <= 0:
            break
        if not _EX99.match(kind):
            continue
        text = clean_text(body, drop_exhibits=False)
        if len(text) < 200:
            continue  # a stub or an image wrapper, not a release
        parts.append(text[:budget])
        budget -= len(text[:budget])
    return "\n\n".join(parts)


def primary_document(raw: str, form_type: str = "8-K") -> str:
    """
    The filing itself, without the accession header or the exhibits.

    A path from form_index points at the *full submission* text file: an SGML
    header, then every attached document in sequence -- press releases,
    certifications, XBRL instance data, occasionally images encoded as text.
    Handing that whole thing to an extractor spends most of the inference budget
    on an accession header and an exhibit list, and invites the model to
    describe boilerplate as if it were an event.

    Falls back to everything after </SEC-HEADER>, and then to the raw text, so a
    submission with unusual document markup degrades to something usable rather
    than to nothing.
    """
    for block in _DOCUMENT.finditer(raw):
        body = block.group(1)
        kind = _DOC_TYPE.search(body)
        if kind and kind.group(1).upper() == form_type.upper():
            return body

    end = _HEADER_END.search(raw)
    if end:
        return raw[end.end():]
    return raw


_FIRST_ITEM = re.compile(r"item\s+\d\.\d{2}", re.IGNORECASE)


def item_body(text: str) -> str:
    """
    The filing from its first numbered item onwards.

    Everything before it is the cover page -- state of incorporation, commission
    file number, IRS employer number, registered address, the title-of-security
    table. That is roughly 1,500 characters of every 8-K, it is identical across
    filings from the same registrant, and it says nothing about the event.

    Cutting it matters more than it sounds on CPU inference, where prompt
    prefill dominates wall-clock time: it is a quarter of the tokens in a
    typical filing, removed at no information cost.

    Returns the text unchanged if no item heading is found, so a filing with
    unusual headings still reaches the extractor whole.
    """
    match = _FIRST_ITEM.search(text)
    return text[match.start():] if match else text


def extract_items(text: str) -> list[str]:
    """
    The 8-K item codes a filing reports, in order of first appearance.

    Body scan, used as the fallback when header_items() finds nothing.
    """
    seen = []
    for match in _ITEM_PATTERN.finditer(text):
        code = match.group(1)
        if code in ITEM_CODES and code not in seen:
            seen.append(code)
    return seen


def is_worth_extracting(items: list[str]) -> bool:
    """
    Should this filing be sent to a model?

    A cheap gate in front of the expensive stage. Filings that report only
    procedural items carry no economic content, and at corpus scale the
    difference between filtering here and not is most of the inference bill.
    Anything with no recognised item is kept, because an unparsed heading is
    more likely a formatting quirk than a genuinely empty filing.
    """
    if not items:
        return True
    return any(item not in BOILERPLATE_ITEMS for item in items)


def load_filings(year: int, quarter: int, limit: int | None = None,
                 data_dir: str = "data/edgar",
                 tickers: pd.DataFrame | None = None,
                 keep_raw: bool = False) -> pd.DataFrame:
    """
    One quarter of 8-Ks, cleaned, item-tagged and resolved to tickers.

    Returns (cik, ticker, company, date_filed, items, worth_extracting, text).
    Filings whose CIK has no ticker mapping are dropped: an unanchored filer
    cannot be traded and cannot anchor a shock, so it has no role in the graph
    as a *filer* -- though it may still appear as a mentioned entity in someone
    else's filing, which is resolved separately.

    **The cache is the cleaned record, not the raw download.** A full submission
    text file averages 1.35 MB -- exhibits, XBRL instance data, base64 images --
    against 3.6 kB for the primary document once cleaned, a factor of 375.
    Caching raw would cost ~20 GB for one quarter and ~300 GB for five years,
    which does not fit on a laptop, and none of it is ever read again after the
    first clean. So each filing is fetched, cleaned, appended to a per-quarter
    JSONL, and its raw copy deleted unless keep_raw is set.

    That JSONL is also the resume point: a re-run skips anything already in it,
    so an interrupted multi-hour fetch continues rather than restarting.
    """
    index = form_index(year, quarter, "8-K", data_dir=data_dir)
    if index.empty:
        return index

    if tickers is None:
        tickers = company_tickers(data_dir=data_dir)
    index = index.merge(tickers[["cik", "ticker"]], on="cik", how="inner")

    if limit is not None:
        index = index.head(limit)

    cleaned_path = Path(data_dir) / "cleaned" / f"{year}Q{quarter}.jsonl"
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    cached: dict[str, dict] = {}
    if cleaned_path.exists():
        with cleaned_path.open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    cached[record["path"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue  # torn final line from an interrupted run

    wanted = set(index["path"])
    todo = [r for r in index.itertuples(index=False) if r.path not in cached]
    if cached:
        print(f"  {len(wanted & set(cached)):,} already cleaned, "
              f"{len(todo):,} to fetch", flush=True)

    with cleaned_path.open("a") as sink:
        for done, record in enumerate(todo, 1):
            try:
                raw = fetch_filing(record.path, data_dir=data_dir)
            except Exception as error:  # one unreachable filing is not fatal
                print(f"  skipped {record.path}: {type(error).__name__}: {error}")
                continue

            # Items come from EDGAR's own header where possible, falling back to
            # a scan of the uncut body. The body is the primary document only,
            # with the accession header and every exhibit removed.
            items = header_items(raw)
            if not items:
                items = extract_items(
                    clean_text(primary_document(raw), drop_exhibits=False))

            # filing_body pulls in the EX-99 press release when the 8-K itself
            # is only a pointer to it, which is the majority of item 2.02.
            body = filing_body(raw)
            row = {
                "path": record.path,
                "cik": int(record.cik),
                "ticker": record.ticker,
                "company": record.company,
                "date_filed": f"{record.date_filed:%Y-%m-%d}",
                "items": items,
                "worth_extracting": is_worth_extracting(items),
                "text": item_body(body),
            }
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            cached[record.path] = row

            if not keep_raw:
                raw_cache = Path(data_dir) / "filings" / record.path.replace("/", "_")
                raw_cache.unlink(missing_ok=True)

            if done % 250 == 0:
                print(f"  fetched {done:,}/{len(todo):,}", flush=True)

    rows = [cached[p] for p in index["path"] if p in cached]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["date_filed"] = pd.to_datetime(frame["date_filed"])
    return frame
