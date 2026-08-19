"""
Stable string -> integer id assignment for graph vertices.

The C++ engine addresses vertices by dense integer ids in [0, max_vertices).
This maps GDELT's free-text actor names (or country codes) onto that range and
keeps the assignment stable across runs by persisting it.
"""

import json
import os
import re
from collections import defaultdict

import numpy as np

# Nationality/demonym forms that GDELT emits interchangeably with the country
# name. Handled by an explicit table rather than fuzzy matching: these are the
# merges that are actually correct, and listing them is cheap and auditable.
DEFAULT_ALIASES = {
    "AMERICAN": "UNITED STATES",
    "ARMENIAN": "ARMENIA",
    "AUSTRALIAN": "AUSTRALIA",
    "AZERBAIJANI": "AZERBAIJAN",
    "BRITISH": "UNITED KINGDOM",
    "BRITAIN": "UNITED KINGDOM",
    "GREAT BRITAIN": "UNITED KINGDOM",
    "UK": "UNITED KINGDOM",
    "USA": "UNITED STATES",
    "US": "UNITED STATES",
    "CHINESE": "CHINA",
    "EGYPTIAN": "EGYPT",
    "FRENCH": "FRANCE",
    "GERMAN": "GERMANY",
    "INDIAN": "INDIA",
    "IRANIAN": "IRAN",
    "IRAQI": "IRAQ",
    "ISRAELI": "ISRAEL",
    "ITALIAN": "ITALY",
    "JAPANESE": "JAPAN",
    "KOREAN": "SOUTH KOREA",
    "SOUTH KOREAN": "SOUTH KOREA",
    "NORTH KOREAN": "NORTH KOREA",
    "MEXICAN": "MEXICO",
    "PAKISTANI": "PAKISTAN",
    "PALESTINIAN": "PALESTINE",
    "RUSSIAN": "RUSSIA",
    "SAUDI": "SAUDI ARABIA",
    "SPANISH": "SPAIN",
    "SYRIAN": "SYRIA",
    "TURKISH": "TURKEY",
    "TURKIYE": "TURKEY",
    "UKRAINIAN": "UKRAINE",
}

_WHITESPACE = re.compile(r"\s+")


def normalize(name):
    """Canonical surface form: trimmed, upper-cased, internal runs collapsed."""
    return _WHITESPACE.sub(" ", str(name).strip().upper())


class EntityMapper:
    """
    Assigns each distinct entity string a dense integer id.

    Three things the previous version got wrong, all of which mattered:

    1. `get_id` was defined twice; the second definition silently won. Only one
       exists now.

    2. That surviving version ran `difflib.get_close_matches` against *every*
       known entity on each miss -- O(unique) per lookup, so O(unique^2) over a
       run. At 276 distinct actors that is 23 microseconds a call; at the 50,000
       the engine is provisioned for it is milliseconds each, and the mapper
       becomes slower than the graph engine it feeds by four orders of
       magnitude. Fuzzy matching is now off by default, and when enabled it
       compares only against a small blocked candidate set (same first letters,
       similar length) instead of the whole vocabulary.

    3. Nothing bounded the ids by the graph's vertex count, so a long run would
       eventually hand the engine an id >= max_vertices and take an
       out_of_range exception mid-replay. `max_entities` now caps assignment;
       past the cap `get_id` returns -1 and the event is skipped, which the
       replayer already handles.

    On fuzzy matching generally: at cutoff 0.85 it merged PRESIDENT into
    RESIDENTS on a single 15-minute slice. The legitimate merges it was catching
    (RUSSIAN -> RUSSIA) are demonyms, which the alias table handles exactly and
    without false positives. Prefer aliases; reach for fuzzy only when you have
    inspected what it does to your vocabulary.

    Args:
        filepath: JSON file the mapping persists to.
        max_entities: Hard cap on distinct ids. Set this to the graph's
            max_vertices. None means unbounded.
        fuzzy: Enable blocked approximate matching for unseen strings.
        fuzzy_cutoff: difflib ratio threshold when fuzzy is on.
        fuzzy_max_candidates: Hard ceiling on comparisons per miss. Blocking
            alone is not enough -- a vocabulary whose strings share a prefix
            ("ENTITY 1", "ENTITY 2", ...) lands entirely in one block -- so the
            candidate list is also truncated. This trades recall for a bounded
            worst case; see the note above about preferring aliases.
        aliases: Overrides DEFAULT_ALIASES. Pass {} to disable.
    """

    def __init__(self, filepath="data/entity_map.json", max_entities=None,
                 fuzzy=False, fuzzy_cutoff=0.88, fuzzy_max_candidates=32,
                 aliases=None):
        self.filepath = filepath
        self.max_entities = max_entities
        self.fuzzy = fuzzy
        self.fuzzy_cutoff = fuzzy_cutoff
        self.fuzzy_max_candidates = fuzzy_max_candidates
        self.aliases = {normalize(k): normalize(v)
                        for k, v in (DEFAULT_ALIASES if aliases is None else aliases).items()}

        self.entity_to_id = {}
        self.id_to_entity = {}
        self.current_id = 0
        self.overflow_count = 0

        # Blocking index for fuzzy candidates, only populated when fuzzy is on.
        self._blocks = defaultdict(list)

        self.load_map()

    # -- blocking ---------------------------------------------------------

    @staticmethod
    def _block_key(name):
        """
        Candidates share a prefix and a coarse length bucket.

        Two strings that differ by more than a couple of characters of length,
        or in their first three characters, will not score above a sane cutoff
        anyway, so restricting comparisons to the same bucket costs no recall
        worth having and turns a full-vocabulary scan into a handful of
        comparisons.
        """
        return (name[:3], len(name) // 4)

    def _index(self, name):
        if self.fuzzy:
            self._blocks[self._block_key(name)].append(name)

    def _fuzzy_lookup(self, name):
        import difflib

        candidates = self._blocks.get(self._block_key(name), ())
        if not candidates:
            return None
        if len(candidates) > self.fuzzy_max_candidates:
            candidates = candidates[:self.fuzzy_max_candidates]
        matches = difflib.get_close_matches(name, candidates, n=1,
                                            cutoff=self.fuzzy_cutoff)
        return matches[0] if matches else None

    # -- assignment -------------------------------------------------------

    def _assign(self, name):
        if self.max_entities is not None and self.current_id >= self.max_entities:
            self.overflow_count += 1
            return -1

        entity_id = self.current_id
        self.entity_to_id[name] = entity_id
        self.id_to_entity[entity_id] = name
        self.current_id += 1
        self._index(name)
        return entity_id

    def get_id(self, entity_name):
        """
        Integer id for an entity string, assigning one if it is new.

        Returns -1 for missing/blank names and for anything arriving after the
        max_entities cap is reached. Callers must skip -1 rather than pass it
        to the engine.
        """
        if entity_name is None:
            return -1
        try:
            if isinstance(entity_name, float) and np.isnan(entity_name):
                return -1
        except TypeError:
            pass

        name = normalize(entity_name)
        if not name or name in ("NAN", "NONE"):
            return -1

        name = self.aliases.get(name, name)

        existing = self.entity_to_id.get(name)
        if existing is not None:
            return existing

        if self.fuzzy:
            match = self._fuzzy_lookup(name)
            if match is not None:
                # Cache the surface form so the next occurrence is a dict hit
                # instead of another fuzzy comparison.
                resolved = self.entity_to_id[match]
                self.entity_to_id[name] = resolved
                return resolved

        return self._assign(name)

    def get_ids(self, names):
        """
        Bulk lookup returning an int64 array.

        Distinct values are resolved once and the result broadcast back over
        the input, so a column with 1,300 rows and 292 distinct actors does 292
        lookups rather than 1,300. Feeding the engine's bulk insert path from
        Python is pointless if the id mapping stays row-at-a-time.
        """
        values = np.asarray(names, dtype=object).astype(str)
        uniques, first_index, inverse = np.unique(
            values, return_index=True, return_inverse=True)

        # np.unique returns sorted order, but ids must be handed out in order
        # of first appearance so that bulk and row-at-a-time mapping produce
        # identical ids for identical input.
        order = np.argsort(first_index)
        resolved = np.empty(len(uniques), dtype=np.int64)
        for position in order:
            resolved[position] = self.get_id(uniques[position])
        return resolved[inverse]

    def get_name(self, entity_id):
        """Reverse lookup: id back to the canonical text name."""
        return self.id_to_entity.get(int(entity_id), "UNKNOWN")

    # -- persistence ------------------------------------------------------

    def save_map(self):
        directory = os.path.dirname(self.filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.filepath, "w") as handle:
            json.dump(self.entity_to_id, handle, indent=4, sort_keys=True)
        print(f"Saved {len(self.entity_to_id)} entity strings "
              f"({self.current_id} distinct ids) to {self.filepath}")

    def load_map(self):
        if not os.path.exists(self.filepath):
            return

        with open(self.filepath) as handle:
            self.entity_to_id = json.load(handle)

        # Several surface forms can share an id (aliases, fuzzy hits), so the
        # reverse index keeps the first name seen for each id and the next free
        # id is one past the maximum -- not len(), which would collide.
        self.id_to_entity = {}
        for name, entity_id in self.entity_to_id.items():
            self.id_to_entity.setdefault(int(entity_id), name)

        self.current_id = (max(self.id_to_entity) + 1) if self.id_to_entity else 0

        if self.fuzzy:
            for name in self.entity_to_id:
                self._index(name)

    def __len__(self):
        return self.current_id

    def __repr__(self):
        cap = "unbounded" if self.max_entities is None else str(self.max_entities)
        return (f"<EntityMapper ids={self.current_id}/{cap} "
                f"surface_forms={len(self.entity_to_id)} fuzzy={self.fuzzy}>")
