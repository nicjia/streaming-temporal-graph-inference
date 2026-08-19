"""
Mapping between GDELT actors and tradable instruments.

GDELT talks about countries; a backtest needs tickers. This module owns that
correspondence and nothing else, so the assumption is in one editable place
rather than scattered through the strategy code.
"""

import json
import os

# Single-country equity ETFs, keyed by ISO 3166-1 alpha-3, which is what GDELT
# puts in Actor1CountryCode / Actor2CountryCode.
#
# Restricted to funds that are liquid and currently listed. Two deliberate
# omissions worth knowing about: Russia has had no investable US-listed ETF
# since 2022 (RSX was liquidated), and several frontier funds (EGPT, NGE) were
# delisted, so they are left out rather than silently producing empty series.
COUNTRY_TO_ETF = {
    "USA": "SPY",
    "CHN": "FXI",
    "JPN": "EWJ",
    "DEU": "EWG",
    "GBR": "EWU",
    "FRA": "EWQ",
    "IND": "INDA",
    "BRA": "EWZ",
    "CAN": "EWC",
    "AUS": "EWA",
    "KOR": "EWY",
    "MEX": "EWW",
    "ISR": "EIS",
    "TUR": "TUR",
    "ZAF": "EZA",
    "CHE": "EWL",
    "ESP": "EWP",
    "ITA": "EWI",
    "NLD": "EWN",
    "SWE": "EWD",
    "SGP": "EWS",
    "HKG": "EWH",
    "TWN": "EWT",
    "IDN": "EIDO",
    "THA": "THD",
    "MYS": "EWM",
    "POL": "EPOL",
    "SAU": "KSA",
    "ARE": "UAE",
    "QAT": "QAT",
    "CHL": "ECH",
    "PER": "EPU",
    "COL": "GXG",
    "ARG": "ARGT",
    "VNM": "VNM",
    "PHL": "EPHE",
    "NZL": "ENZL",
    "GRC": "GREK",
    "NOR": "NORW",
    "IRL": "EIRL",
}

# Instruments that respond to global conflict rather than to one country.
# Excluded from the cross-sectional book by default: they are not comparable to
# a country ETF, and mixing them in silently changes what the ranking means.
THEMATIC = {
    "GLD": "gold",
    "USO": "crude oil",
    "XLE": "energy equities",
    "ITA_DEFENSE": "US aerospace and defense (ticker ITA)",
    "VIXY": "short-term VIX futures",
}

# GDELT frequently names a country without filling in the country code column,
# and slices downloaded before the schema was widened have no code column at
# all. This is the fallback path from actor name to ISO3.
COUNTRY_NAMES = {
    "UNITED STATES": "USA", "AMERICA": "USA", "WASHINGTON": "USA",
    "CHINA": "CHN", "BEIJING": "CHN",
    "JAPAN": "JPN", "TOKYO": "JPN",
    "GERMANY": "DEU", "BERLIN": "DEU",
    "UNITED KINGDOM": "GBR", "LONDON": "GBR", "ENGLAND": "GBR", "SCOTLAND": "GBR",
    "FRANCE": "FRA", "PARIS": "FRA",
    "INDIA": "IND", "NEW DELHI": "IND",
    "BRAZIL": "BRA", "BRASILIA": "BRA",
    "CANADA": "CAN", "OTTAWA": "CAN",
    "AUSTRALIA": "AUS", "CANBERRA": "AUS",
    "SOUTH KOREA": "KOR", "SEOUL": "KOR",
    "MEXICO": "MEX",
    "ISRAEL": "ISR", "JERUSALEM": "ISR", "TEL AVIV": "ISR",
    "TURKEY": "TUR", "ANKARA": "TUR",
    "SOUTH AFRICA": "ZAF",
    "SWITZERLAND": "CHE", "GENEVA": "CHE", "DAVOS": "CHE",
    "SPAIN": "ESP", "MADRID": "ESP",
    "ITALY": "ITA", "ROME": "ITA",
    "NETHERLANDS": "NLD", "AMSTERDAM": "NLD", "THE HAGUE": "NLD",
    "SWEDEN": "SWE", "STOCKHOLM": "SWE",
    "SINGAPORE": "SGP",
    "HONG KONG": "HKG",
    "TAIWAN": "TWN", "TAIPEI": "TWN",
    "INDONESIA": "IDN", "JAKARTA": "IDN",
    "THAILAND": "THA", "BANGKOK": "THA",
    "MALAYSIA": "MYS",
    "POLAND": "POL", "WARSAW": "POL",
    "SAUDI ARABIA": "SAU", "RIYADH": "SAU",
    "UNITED ARAB EMIRATES": "ARE", "DUBAI": "ARE", "ABU DHABI": "ARE",
    "QATAR": "QAT", "DOHA": "QAT",
    "CHILE": "CHL", "PERU": "PER", "COLOMBIA": "COL",
    "ARGENTINA": "ARG", "BUENOS AIRES": "ARG",
    "VIETNAM": "VNM", "PHILIPPINES": "PHL", "NEW ZEALAND": "NZL",
    "GREECE": "GRC", "ATHENS": "GRC",
    "NORWAY": "NOR", "OSLO": "NOR",
    "IRELAND": "IRL", "DUBLIN": "IRL",
    # Present in the event stream and useful as graph vertices even though they
    # have no investable ETF here.
    "RUSSIA": "RUS", "MOSCOW": "RUS", "KREMLIN": "RUS",
    "UKRAINE": "UKR", "KYIV": "UKR", "KIEV": "UKR",
    "IRAN": "IRN", "TEHRAN": "IRN",
    "IRAQ": "IRQ", "BAGHDAD": "IRQ",
    "SYRIA": "SYR", "DAMASCUS": "SYR",
    "PALESTINE": "PSE", "GAZA": "PSE", "WEST BANK": "PSE",
    "LEBANON": "LBN", "BEIRUT": "LBN",
    "NORTH KOREA": "PRK", "PYONGYANG": "PRK",
    "PAKISTAN": "PAK", "AFGHANISTAN": "AFG", "YEMEN": "YEM",
    "EGYPT": "EGY", "CAIRO": "EGY", "NIGERIA": "NGA", "KENYA": "KEN",
    "ETHIOPIA": "ETH", "SUDAN": "SDN", "LIBYA": "LBY", "ALGERIA": "DZA",
    "MOROCCO": "MAR", "VENEZUELA": "VEN", "CUBA": "CUB",
    "BELARUS": "BLR", "AZERBAIJAN": "AZE", "ARMENIA": "ARM", "GEORGIA": "GEO",
    "KAZAKHSTAN": "KAZ", "UZBEKISTAN": "UZB", "SERBIA": "SRB", "CROATIA": "HRV",
    "HUNGARY": "HUN", "ROMANIA": "ROU", "BULGARIA": "BGR", "CZECH REPUBLIC": "CZE",
    "AUSTRIA": "AUT", "BELGIUM": "BEL", "DENMARK": "DNK", "FINLAND": "FIN",
    "PORTUGAL": "PRT", "BANGLADESH": "BGD", "SRI LANKA": "LKA", "MYANMAR": "MMR",
}


class Universe:
    """
    The set of countries the strategy can express a view on, and the
    instrument each maps to.

    Args:
        mapping: ISO3 -> ticker. Defaults to COUNTRY_TO_ETF.
        thematic: Extra non-country instruments to carry alongside. Off by
            default; see the note on THEMATIC.
    """

    def __init__(self, mapping=None, thematic=None):
        self.mapping = dict(COUNTRY_TO_ETF if mapping is None else mapping)
        self.thematic = dict(thematic or {})

    # -- resolution -------------------------------------------------------

    @staticmethod
    def country_from_name(name):
        """ISO3 for a GDELT actor name, or None if it is not a country we know."""
        if name is None:
            return None
        return COUNTRY_NAMES.get(str(name).strip().upper())

    def is_tradable(self, iso3):
        return iso3 in self.mapping

    def ticker(self, iso3):
        return self.mapping.get(iso3)

    def countries(self):
        return sorted(self.mapping)

    def tickers(self):
        return sorted(set(self.mapping.values()) | set(self.thematic))

    # -- persistence ------------------------------------------------------

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            json.dump({"mapping": self.mapping, "thematic": self.thematic},
                      handle, indent=2, sort_keys=True)
        print(f"Saved universe ({len(self.mapping)} countries) to {path}")

    @classmethod
    def load(cls, path):
        with open(path) as handle:
            payload = json.load(handle)
        return cls(mapping=payload.get("mapping"), thematic=payload.get("thematic"))

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f"<Universe countries={len(self.mapping)} thematic={len(self.thematic)}>"
