"""
Closed-vocabulary schema for turning filing and news text into typed graph edges.

Every field the model emits is a choice from a fixed list, enforced by
constrained decoding, so an off-vocabulary answer is structurally impossible.
The numeric edge weight is derived from those labels deterministically, so the
label-to-magnitude mapping lives here and edges can be re-derived without
re-running inference. Two integers per edge reach the engine:

  relation (uint16)  packs the relation type and the model's confidence
  weight   (float)   sign * a geometric magnitude ladder

The ladder is geometric because shocks compose multiplicatively along a path.
"""

from __future__ import annotations

import json
import re
from enum import IntEnum


# 8-K item codes. The item is a coarse, free, human-assigned event label that
# arrives with every filing, which makes it two useful things at once: a filter
# for deciding which filings are worth spending inference on, and an independent
# check on what the extractor produced. An 8-K filed under 2.02 whose extracted
# edges contain no EARNINGS_EVENT is a signal that extraction failed.
ITEM_CODES = {
    "1.01": "entry into material definitive agreement",
    "1.02": "termination of material definitive agreement",
    "1.03": "bankruptcy or receivership",
    "1.04": "mine safety",
    "2.01": "completion of acquisition or disposition",
    "2.02": "results of operations and financial condition",
    "2.03": "creation of direct financial obligation",
    "2.04": "triggering event accelerating an obligation",
    "2.05": "costs associated with exit or disposal",
    "2.06": "material impairments",
    "3.01": "notice of delisting or listing-rule failure",
    "3.02": "unregistered sales of equity securities",
    "3.03": "material modification to security holder rights",
    "4.01": "change in certifying accountant",
    "4.02": "non-reliance on previously issued financials",
    "5.01": "change in control of registrant",
    "5.02": "departure or election of directors or officers",
    "5.03": "amendment to articles or bylaws",
    "5.04": "suspension of trading under employee benefit plans",
    "5.05": "amendment to code of ethics",
    "5.06": "change in shell company status",
    "5.07": "submission of matters to a shareholder vote",
    "5.08": "shareholder director nominations",
    "7.01": "regulation FD disclosure",
    "8.01": "other events",
    "9.01": "financial statements and exhibits",
}


class Relation(IntEnum):
    """
    Closed relation vocabulary. Eight entries, and the number is the point.

    An earlier version had 33, on the theory that a richer taxonomy carries more
    information per edge. Measured, it carries less. Two models 2.3x apart in
    size -- Qwen2.5 3B and 7B, same 150 filings, same prompt -- both scored 8%
    on the one externally checkable statistic available: whether a filing EDGAR
    labelled item 2.02 ("results of operations") produced an earnings-related
    edge. The 3B collapsed onto ACCOUNTING_EVENT for 39% of everything; the 7B
    spread across 24 types and typed them close to arbitrarily, calling an
    earnings release an APPROVAL_DECISION 67 times. Identical failure at both
    sizes is the signature of an unlearnable task specification rather than an
    undersized model.

    The diagnosis is that many of those labels were not separable from the text.
    APPROVAL_DECISION, RESTRUCTURING, ACCOUNTING_EVENT and EARNINGS_RESULT all
    describe things a quarterly announcement mentions, and asking which one it
    "is" has no determinate answer. So they are merged into families whose
    boundaries a reader could defend.

    Sign still carries direction: EARNINGS_EVENT with sign -1 is a miss or a
    guidance cut, with +1 a beat or a raise. That keeps the vocabulary small
    without losing the distinction that matters for propagation.

    UNKNOWN is 0 to match RELATION_UNKNOWN in the C++ core, whose zero-fill of
    the relation array assumes that sentinel.
    """

    UNKNOWN = 0

    # -- structural: a standing arrangement between two entities -------------
    SUPPLIES_TO = 1        # supplier -> customer, either direction stated
    OWNS = 2               # parent/subsidiary, equity stake, acquisition target
    COMPETES_WITH = 3
    PARTNERS_WITH = 4      # joint venture, collaboration, licensing, alliance
    LENDS_TO = 5           # creditor, lender, underwriter

    # -- events: something that happened, usually to the filer ---------------
    EARNINGS_EVENT = 10    # results, guidance, impairment, restatement
    LEADERSHIP_CHANGE = 11 # officers and directors arriving or leaving
    DEAL_EVENT = 12        # M&A, divestiture, financing, buyback, bankruptcy
    LEGAL_EVENT = 13       # litigation, regulator, investigation, approval


class Sign(IntEnum):
    """Direction of the effect on the destination node's value."""

    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1


class Magnitude(IntEnum):
    """
    Ordinal effect size. Four buckets, because a small model can rank into four
    and cannot calibrate a continuous value.
    """

    NEGLIGIBLE = 0
    SMALL = 1
    MODERATE = 2
    LARGE = 3


class Confidence(IntEnum):
    """How sure the extractor is that this edge is stated by the text at all."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2


# Geometric ladder. MODERATE is the unit so that a typical edge carries weight
# 1.0 and a path of typical hops neither explodes nor vanishes.
MAGNITUDE_SCALE = {
    Magnitude.NEGLIGIBLE: 0.1,
    Magnitude.SMALL: 0.3,
    Magnitude.MODERATE: 1.0,
    Magnitude.LARGE: 3.0,
}

# Structural relations describe a standing arrangement rather than a shock, so
# they carry no directional effect of their own. Recorded separately because an
# extractor that assigns them a sign is making something up, and the parser
# should not quietly accept it.
STRUCTURAL = frozenset({
    Relation.SUPPLIES_TO, Relation.OWNS, Relation.COMPETES_WITH,
    Relation.PARTNERS_WITH, Relation.LENDS_TO,
})

# Bit layout for the uint16 relation channel.
#   bits 0-7   relation id      (0-255; the enum uses 0-13)
#   bits 8-9   confidence       (0-2)
#   bits 10-15 reserved
_RELATION_MASK = 0x00FF
_CONFIDENCE_SHIFT = 8
_CONFIDENCE_MASK = 0x0300


def pack_relation(relation: Relation, confidence: Confidence) -> int:
    """Relation type and confidence into the engine's uint16 channel."""
    return (int(relation) & _RELATION_MASK) | (
        (int(confidence) << _CONFIDENCE_SHIFT) & _CONFIDENCE_MASK)


def unpack_relation(code: int) -> tuple[Relation, Confidence]:
    """Inverse of pack_relation."""
    return (Relation(code & _RELATION_MASK),
            Confidence((code & _CONFIDENCE_MASK) >> _CONFIDENCE_SHIFT))


def edge_weight(sign: Sign, magnitude: Magnitude) -> float:
    """
    Signed effect size for the engine's float weight array.

    Structural edges should be passed sign NEUTRAL and come back 0.0, which is
    correct: "A supplies B" is a conduit, not a shock. The propagation model
    learns what flows along it.
    """
    return float(int(sign)) * MAGNITUDE_SCALE[Magnitude(magnitude)]


# --------------------------------------------------------------------------
# Constrained decoding
# --------------------------------------------------------------------------

_RELATION_NAMES = [r.name for r in Relation if r is not Relation.UNKNOWN]
_SIGN_NAMES = ["negative", "neutral", "positive"]
_MAGNITUDE_NAMES = ["negligible", "small", "moderate", "large"]
_CONFIDENCE_NAMES = ["low", "medium", "high"]

_SIGN_BY_NAME = {"negative": Sign.NEGATIVE, "neutral": Sign.NEUTRAL,
                 "positive": Sign.POSITIVE}
_MAGNITUDE_BY_NAME = {name: Magnitude(i) for i, name in enumerate(_MAGNITUDE_NAMES)}
_CONFIDENCE_BY_NAME = {name: Confidence(i) for i, name in enumerate(_CONFIDENCE_NAMES)}


def json_schema(max_edges: int = 12) -> dict:
    """
    JSON Schema for guided generation.

    Accepted directly by vLLM (`guided_json`), Outlines, llama-cpp-python's
    response_format, and the OpenAI-compatible structured-output APIs. Every
    field is an enum or a bounded string, so a conforming generation cannot be
    off-vocabulary and the parser below never has to reject one.

    `subject` and `object` are free strings only because entity resolution is a
    separate stage -- the model names what it saw, and resolution against the
    anchored universe happens afterwards where it can be audited and corrected
    without re-running inference.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["edges"],
        "properties": {
            "edges": {
                "type": "array",
                "maxItems": max_edges,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["subject", "object", "relation", "sign",
                                 "magnitude", "confidence"],
                    "properties": {
                        "subject": {"type": "string", "maxLength": 120},
                        "object": {"type": "string", "maxLength": 120},
                        "relation": {"type": "string", "enum": _RELATION_NAMES},
                        "sign": {"type": "string", "enum": _SIGN_NAMES},
                        "magnitude": {"type": "string", "enum": _MAGNITUDE_NAMES},
                        "confidence": {"type": "string", "enum": _CONFIDENCE_NAMES},
                    },
                },
            }
        },
    }


def gbnf_grammar(max_edges: int = 12) -> str:
    """
    The same constraint as a llama.cpp GBNF grammar, for running a local GGUF
    model through llama.cpp or Ollama without a JSON-Schema layer in between.

    Kept alongside json_schema() rather than generated from it because the two
    runtimes disagree about whitespace handling, and a grammar that is subtly
    wrong fails by producing plausible-looking garbage rather than an error.
    """
    def alternatives(names):
        return " | ".join(f'"\\"{n}\\""' for n in names)

    return f'''
root        ::= "{{" ws "\\"edges\\"" ws ":" ws edges ws "}}"
edges       ::= "[" ws "]" | "[" ws edge (ws "," ws edge){{0,{max_edges - 1}}} ws "]"
edge        ::= "{{" ws
                "\\"subject\\"" ws ":" ws string ws "," ws
                "\\"object\\"" ws ":" ws string ws "," ws
                "\\"relation\\"" ws ":" ws relation ws "," ws
                "\\"sign\\"" ws ":" ws sign ws "," ws
                "\\"magnitude\\"" ws ":" ws magnitude ws "," ws
                "\\"confidence\\"" ws ":" ws confidence ws
                "}}"
relation    ::= {alternatives(_RELATION_NAMES)}
sign        ::= {alternatives(_SIGN_NAMES)}
magnitude   ::= {alternatives(_MAGNITUDE_NAMES)}
confidence  ::= {alternatives(_CONFIDENCE_NAMES)}
string      ::= "\\"" char{{1,120}} "\\""
char        ::= [^"\\\\\\x00-\\x1F]
ws          ::= [ \\t\\n]*
'''.strip()


SYSTEM_PROMPT = """\
You extract causal relationships from financial text as structured edges.

An edge means: this text asserts that SUBJECT affects, or stands in a stated \
relationship to, OBJECT.

SUBJECT and OBJECT must both be NAMED ENTITIES:
  - a company or organisation ("Beta Industries", "the Federal Reserve")
  - a named person ("Jane Okafor")

NEVER use as a subject or object:
  - documents: "press release", "Exhibit 99.1", "financial results"
  - dates: "May 3, 2024", "Q3 2024", "fiscal 2024"
  - categories: "the semiconductor industry", "the market", "shareholders"
If one side is not a named entity, do not emit that edge.

The eight relations, and what belongs in each:
  SUPPLIES_TO        one supplies, sells to or is a customer of the other
  OWNS               parent, subsidiary, equity stake, acquisition target
  COMPETES_WITH      named as a competitor
  PARTNERS_WITH      joint venture, collaboration, licensing, alliance
  LENDS_TO           lender, creditor, underwriter, credit facility provider
  EARNINGS_EVENT     results, revenue, guidance, impairment, restatement
  LEADERSHIP_CHANGE  an officer or director appointed, resigning or departing
  DEAL_EVENT         merger, acquisition, divestiture, financing, buyback
  LEGAL_EVENT        lawsuit, regulator, investigation, approval, delisting

Rules:
- Only emit an edge the text actually states. Do not use outside knowledge.
- An event affecting only the filer is an edge from the filer TO ITSELF. \
Reporting quarterly results is EARNINGS_EVENT from the filer to the filer.
- An officer appointment is LEADERSHIP_CHANGE from the company to that person.
- sign is the effect on the OBJECT's value: positive, negative, or neutral.
- magnitude is how large that effect is: negligible, small, moderate, large.
- confidence is how clearly the text states the edge, not how important it is.
- The first five relations describe standing arrangements rather than shocks: \
always give them sign neutral.
- A company is never related to itself by OWNS, SUPPLIES_TO, COMPETES_WITH, \
PARTNERS_WITH or LENDS_TO. Those need two different entities.
- Do not repeat the same edge. Emit each distinct relationship once.

Returning {"edges": []} is a good answer when the text names no relationship. \
Prefer an empty list over a speculative edge.
"""


def build_prompt(text: str, filer: str | None = None,
                 max_chars: int = 6000, items: list[str] | None = None,
                 item_descriptions: dict[str, str] | None = None,
                 candidates: list[str] | None = None) -> str:
    """
    The user half of the extraction prompt.

    Naming the filer matters more than it looks. 8-K prose is written from
    inside the company and refers to itself as "the Company" throughout, so an
    extractor without the name emits edges on a node called "the Company" and
    every filer in the corpus collapses into one vertex. Supplying the name up
    front is a one-line fix for what would otherwise be the single largest
    entity-resolution failure in the dataset.
    """
    lines = []
    if filer:
        lines.append(f"Filing entity: {filer}")
    if items and item_descriptions:
        # EDGAR's own item labels, supplied as context. They are free, assigned
        # by a human rather than inferred, and they tell the model what kind of
        # event the document reports -- which is precisely what it was getting
        # wrong when it read an earnings release and called it an approval
        # decision. This is the one external signal available without labelling.
        described = [f"{code} ({item_descriptions[code]})"
                     for code in items if code in item_descriptions]
        if described:
            lines.append("This filing reports: " + "; ".join(described))
    if candidates:
        # Publicly listed companies a dictionary scan already found in this
        # text. Naming them is the difference between a graph and a pile of
        # stars: on a full quarter the model produced 20,439 edges but only 325
        # listed companies ended up connected to another listed company,
        # because most extracted relationships point at private subsidiaries,
        # products and individual executives. Those are legitimate vertices and
        # useless for a tradeable graph. The scan knows which filings contain a
        # second listed name; telling the model is far cheaper than hoping it
        # notices.
        lines.append(
            "Other publicly listed companies named in this text: "
            + ", ".join(candidates[:20])
            + ". If the text states a relationship between the filer and any of "
              "them, emit it.")
    header = "\n".join(lines)
    if header:
        header += "\n\n"
    return f"{header}Text:\n{text[:max_chars]}\n\nExtract the edges."


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _same_entity(left: str, right: str) -> bool:
    """Same company under different corporate suffixes or spacing."""
    strip = lambda t: re.sub(
        r"\b(?:inc|corp|corporation|company|co|llc|llp|lp|ltd|plc|holdings?|"
        r"group|incorporated)\b\.?", " ", re.sub(r"[^\w\s]", " ", t.lower()))
    return re.sub(r"\s+", " ", strip(left)).strip() == \
           re.sub(r"\s+", " ", strip(right)).strip()


class ExtractionError(ValueError):
    """Raised when generated output does not conform to the closed vocabulary."""


# Strings the extractor reaches for when a filing names no counterparty. Every
# one of these was produced by a 3B model on a real 8-K, typed as a relation to
# the filer: "APi Group Corp -> press release", "10x Genomics -> financial
# results", "American Assets Trust -> supplemental information". They are the
# filing talking about itself, and as vertices they would be enormous spurious
# hubs joining every registrant that ever attached an exhibit.
#
# Enforced here rather than left to the prompt, because a small model asked
# nicely still does it, and a rule that matters should not depend on compliance.
_NON_ENTITY = frozenset({
    "press release", "the press release", "exhibit", "exhibit 99", "exhibit 99.1",
    "the filing", "this filing", "the report", "the presentation", "presentation",
    "financial results", "results", "the results", "supplemental information",
    "the company's financial results", "earnings release", "the announcement",
    "announcement", "common stock", "the common stock", "shares", "the shares",
    "revenue", "revenues", "guidance", "the guidance", "earnings", "the market",
    "markets", "shareholders", "stockholders", "the quarter", "the fiscal year",
    "operations", "the transaction", "the agreement", "the board", "the offering",
    "investors", "the sec", "management", "the conference call", "conference call",
    "the webcast", "webcast", "the document", "document", "the exhibit",
})


# A date is not a vertex. Small models reach for them constantly -- "Cognex
# Corporation -> May 3, 2024, EARNINGS_RESULT" was 25.8% of a pilot's objects --
# because a date is the most salient other noun phrase in the sentence.
_DATE_LIKE = re.compile(
    r"^\s*(?:"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{4}"
    r"|(?:q[1-4]|fy)\s*\d{0,4}"
    r"|(?:first|second|third|fourth)\s+quarter.*"
    r"|fiscal\s+(?:year\s+)?\d{0,4}"
    r")\s*$", re.IGNORECASE)

# "the <something>" is almost always a description rather than a name: "the U.S.
# cannabis industry", "the DEA's commitment to". Proper nouns that legitimately
# start with an article ("The Coca-Cola Company") keep an internal capital, so
# the capital-letter test below readmits them.
_ARTICLE_LEAD = re.compile(r"^\s*the\s+", re.IGNORECASE)

# Head nouns that make a phrase a description of a category rather than the name
# of a thing. "The U.S. cannabis industry" is not a vertex; "Canopy Growth" is.
_DESCRIPTIVE_TAIL = re.compile(
    r"\b(?:industry|industries|sector|sectors|market|markets|economy|"
    r"environment|conditions|landscape|commitment|commitments|ability|"
    r"outlook|demand|supply|growth|performance|results|position|strategy|"
    r"initiative|initiatives|programme|program|programs|portfolio|business|"
    r"businesses|operations|segment|segments|customers|suppliers|competitors|"
    r"shareholders|stockholders|employees|team|community|public)\s*$",
    re.IGNORECASE)


# Legal-instrument roles. A rights plan says "Acquiring Person", a warrant says
# "Exercising Holder", a credit agreement says "the Lender" -- defined terms that
# stand in for whoever fills the role, not names of anyone. They read like
# entities (capitalised, singular, grammatically a party) and a model treats them
# as such, but as vertices they merge every unrelated counterparty in the corpus
# into one node.
_LEGAL_ROLE = re.compile(
    r"^(?:the\s+)?(?:acquiring\s+person|exercising\s+holder|holder|purchaser|"
    r"lender|borrower|agent|trustee|underwriter|investor|participant|subscriber|"
    r"grantee|optionee|seller|buyer|counterparty|guarantor|issuer|registrant|"
    r"pledgor|assignee|indemnitee|obligor)s?\s*$", re.IGNORECASE)


def _is_entity(name: str) -> bool:
    """
    Reject anything that is not plausibly a named entity.

    Four tests, each one earned by a specific failure observed in a pilot:
    a blocklist for document artifacts, a date test, an article-lead test for
    descriptive phrases, and a requirement that the string contain a capital
    letter somewhere -- which is what separates "Beta Industries" from "the
    quarterly dividend" without needing a list of every common noun.
    """
    raw = str(name).strip()
    if not raw:
        return False
    if _DATE_LIKE.match(raw):
        return False

    if _LEGAL_ROLE.match(raw):
        return False

    cleaned = re.sub(r"[^\w\s&.-]", " ", raw.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned or cleaned in _NON_ENTITY:
        return False

    # A name contains a capital somewhere past any leading article, so
    # "The Coca-Cola Company" passes and "the quarterly dividend" does not.
    body = _ARTICLE_LEAD.sub("", raw)
    if not re.search(r"[A-Z]", body):
        return False

    # The capital test alone is not enough: "the U.S. cannabis industry" borrows
    # one from "U.S." while being a description, not an entity. The head noun is
    # what separates them -- but only in combination with the leading article,
    # because "Beta Industries" and "Illinois Tool Works" are real names ending
    # in exactly those words. Gating on the article keeps both calls right.
    if _ARTICLE_LEAD.match(raw) and _DESCRIPTIVE_TAIL.search(cleaned):
        return False
    return True


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_edges(raw: str | dict, strict: bool = True) -> list[dict]:
    """
    Validate generated output against the closed vocabulary and derive the
    engine's two numeric fields.

    Returns one dict per edge with `subject`, `object`, `relation_code`
    (uint16), `weight` (float), plus the original labels for auditing and for
    re-deriving weights if MAGNITUDE_SCALE changes.

    With `strict=False`, malformed edges are dropped rather than raising, which
    is what a bulk run wants: one bad generation in a million filings should not
    end the job. Strict is the default so that a misconfigured grammar surfaces
    immediately instead of silently producing an empty graph.
    """
    if isinstance(raw, str):
        match = _JSON_BLOCK.search(raw)
        if match is None:
            if strict:
                # Quote what actually came back. Under constrained decoding this
                # should be unreachable, so when it fires the useful question is
                # whether the constraint was applied at all -- and that is only
                # answerable if the output is in the message rather than
                # discarded.
                snippet = raw[:200].replace("\n", " ") if raw else "<empty>"
                raise ExtractionError(
                    f"no JSON object in generated output ({len(raw)} chars): "
                    f"{snippet!r}")
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            if strict:
                raise ExtractionError(f"unparseable JSON: {error}") from error
            return []
    else:
        payload = raw

    out = []
    # Small models loop. One pilot filing emitted the same
    # "-> the U.S. cannabis industry" edge ten times identically, and exact
    # repeats were 28.6% of all edges -- enough to make one verbose filing look
    # like a relation type's whole population. Dedup on the triple that defines
    # an edge; sign and magnitude are consequences of it, not part of its
    # identity, so a repeat with a different magnitude is still a repeat.
    seen: set[tuple[str, str, str]] = set()

    for index, item in enumerate(payload.get("edges", []) or []):
        try:
            relation = Relation[str(item["relation"]).strip().upper()]
            sign = _SIGN_BY_NAME[str(item["sign"]).strip().lower()]
            magnitude = _MAGNITUDE_BY_NAME[str(item["magnitude"]).strip().lower()]
            confidence = _CONFIDENCE_BY_NAME[str(item["confidence"]).strip().lower()]
            subject = str(item["subject"]).strip()
            obj = str(item["object"]).strip()
            if not subject or not obj:
                raise KeyError("empty endpoint")
            if not _is_entity(subject) or not _is_entity(obj):
                continue  # a document artifact, not a relationship
        except (KeyError, ValueError) as error:
            if strict:
                raise ExtractionError(
                    f"edge {index} is off-vocabulary: {error}") from error
            continue

        # A structural relation with a sign is the model inventing a shock it
        # was told not to invent. Neutralise rather than drop: the relationship
        # itself is still usable as a conduit.
        if relation in STRUCTURAL:
            sign = Sign.NEUTRAL
            # A standing arrangement needs two parties. Compared on the
            # normalised form, not the raw string: "Ellington Credit Co OWNS
            # Ellington Credit Company" and "Cogent Communications Holdings
            # LENDS_TO Cogent Communications Group" are the same company under
            # two corporate suffixes, and a raw comparison lets both through.
            if _same_entity(subject, obj):
                continue

        signature = (subject.lower(), obj.lower(), relation.name)
        if signature in seen:
            continue
        seen.add(signature)

        out.append({
            "subject": subject,
            "object": obj,
            "relation": relation.name,
            "sign": sign.name.lower(),
            "magnitude": magnitude.name.lower(),
            "confidence": confidence.name.lower(),
            "relation_code": pack_relation(relation, confidence),
            "weight": edge_weight(sign, magnitude),
        })
    return out
