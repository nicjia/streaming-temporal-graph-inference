"""
Mapping between GDELT actors and tradable instruments.

GDELT talks about countries; a backtest needs tickers. This module owns that
correspondence and nothing else, so the assumption is in one editable place
rather than scattered through the strategy code.
"""

import json
import os

import numpy as np
import pandas as pd

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

    def __init__(self, mapping=None, thematic=None, invert=None):
        self.mapping = dict(COUNTRY_TO_ETF if mapping is None else mapping)
        self.thematic = dict(thematic or {})
        # Instruments whose quoted series moves opposite to the exposure we
        # want; the engine uses 1/price for these. See FX_INVERTED.
        self.invert = set(invert or ())

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

    def liquidity_screen(self, prices, min_dollar_volume=2_000_000, quiet=False):
        """
        Drop instruments too thin to trade at the modelled cost.

        A cross-sectional book weights every name by signal strength, not by
        how tradable it is, so a fund turning over $82k a day (QAT, measured)
        gets the same treatment as SPY at $28bn. Charging both 5bps is fiction:
        for the thin tail the real round trip is one to two orders of magnitude
        worse, and any backtest that includes them is quietly reporting returns
        nobody could have captured.

        Requires a `volume` column; without one, prices alone cannot say what
        is tradable and the universe is returned unchanged.
        """
        if "volume" not in prices.columns:
            if not quiet:
                print("liquidity_screen: no volume column, universe unchanged")
            return self

        frame = prices.copy()
        frame["dollar_volume"] = frame["close"] * frame["volume"]
        median = frame.groupby("ticker")["dollar_volume"].median()

        kept, dropped, unknown = {}, {}, set()
        for iso3, ticker in self.mapping.items():
            value = median.get(ticker, np.nan)
            # Spot FX reports no volume anywhere -- the interbank market has no
            # consolidated tape. Absent volume means "cannot be measured", not
            # "illiquid", and screening it out would delete the most liquid
            # instruments in the world. Only a positive-but-small figure is
            # evidence of thinness.
            if not np.isfinite(value) or value <= 0:
                kept[iso3] = ticker
                unknown.add(ticker)
            elif value >= min_dollar_volume:
                kept[iso3] = ticker
            else:
                dropped[ticker] = value

        if unknown and not quiet:
            print(f"Liquidity screen: {len(unknown)} instrument(s) report no volume "
                  f"(spot FX has no consolidated tape); kept without screening")
        if dropped and not quiet:
            listing = ", ".join(f"{t} (${v/1e6:.2f}M)"
                                for t, v in sorted(dropped.items(), key=lambda x: x[1]))
            print(f"Liquidity screen at ${min_dollar_volume/1e6:.1f}M/day dropped "
                  f"{len(dropped)} of {len(self.mapping)}: {listing}")

        return Universe(mapping=kept, thematic=self.thematic, invert=self.invert)

    def tickers(self):
        return sorted(set(self.mapping.values()) | set(self.thematic))

    def tradability_screen(self, prices, min_ann_vol=0.025, max_ann_vol=0.20,
                           max_abs_daily=0.10, quiet=False):
        """
        Drop instruments that are not tradable as spot, from both ends.

        Volume screening is the wrong tool for currencies, and the FX universe
        as first written contained two kinds of instrument that quietly wreck a
        cross-sectional book:

        * **Pegs.** AED at 0.2% annualised volatility, HKD at 0.7%. A z-scored
          signal will size into them exactly as hard as into anything else, and
          the position cannot pay off in either direction because the exchange
          rate is administered.

        * **Managed and crisis currencies.** USDRUB measured 608% annualised
          volatility with a single +1414% day across the 2022 dislocation; ARS
          fell 54% in one session on devaluation, EGP 38%. These are step
          repricings behind capital controls, not returns a spot book could
          have earned, and one of them dominates the entire book's variance.

        The middle band -- roughly 2.5% to 20% annualised, with no single day
        beyond 10% -- is what actually trades continuously at a spread worth
        modelling.
        """
        frame = prices.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        wide = frame.pivot_table(index="date", columns="ticker", values="close",
                                 aggfunc="last").sort_index()

        for ticker in self.invert & set(wide.columns):
            wide[ticker] = 1.0 / wide[ticker]
        returns = wide.pct_change(fill_method=None)

        volatility = returns.std() * np.sqrt(252)
        extreme = returns.abs().max()

        kept, dropped = {}, {}
        for iso3, ticker in self.mapping.items():
            vol = volatility.get(ticker, np.nan)
            worst = extreme.get(ticker, np.nan)
            if not np.isfinite(vol):
                kept[iso3] = ticker
            elif vol < min_ann_vol:
                dropped[ticker] = f"peg ({vol:.1%} vol)"
            elif vol > max_ann_vol:
                dropped[ticker] = f"unstable ({vol:.0%} vol)"
            elif np.isfinite(worst) and worst > max_abs_daily:
                dropped[ticker] = f"step move ({worst:.0%} in a day)"
            else:
                kept[iso3] = ticker

        if dropped and not quiet:
            listing = ", ".join(f"{t}: {why}" for t, why in sorted(dropped.items()))
            print(f"Tradability screen dropped {len(dropped)} instrument(s): {listing}")

        return Universe(mapping=kept, thematic=self.thematic, invert=self.invert)

    # -- persistence ------------------------------------------------------

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            json.dump({"mapping": self.mapping, "thematic": self.thematic,
                       "invert": sorted(self.invert)}, handle, indent=2, sort_keys=True)
        print(f"Saved universe ({len(self.mapping)} countries) to {path}")

    @classmethod
    def load(cls, path):
        with open(path) as handle:
            payload = json.load(handle)
        return cls(mapping=payload.get("mapping"), thematic=payload.get("thematic"),
                   invert=payload.get("invert"))

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return (f"<Universe countries={len(self.mapping)} "
                f"instruments={len(set(self.mapping.values()))} "
                f"thematic={len(self.thematic)}>")

# ---------------------------------------------------------------------------
# FX universe
# ---------------------------------------------------------------------------
#
# Country ETFs are the wrong instrument for event-driven work and the measured
# reasons are specific: most of them track markets that are *closed* during US
# trading hours, so a GDELT event about Japan at 14:00 ET cannot reprice EWJ
# against Tokyo -- what you see is US market makers estimating NAV until the
# next Tokyo open. Ten of the forty also trade under $2M a day.
#
# Currencies have neither problem. FX trades continuously, spreads on the
# majors are the tightest in any market, and a country's currency is the most
# direct liquid expression of a view on that country.
#
# One structural difference: currencies are shared. Eleven eurozone countries
# map to one EUR, so several countries can point at the same instrument and
# their signals have to be aggregated before trading. The backtest engine
# averages duplicate instruments rather than opening the same position twice.

FX_INSTRUMENTS = {
    "JPN": "USDJPY=X", "CAN": "USDCAD=X", "CHE": "USDCHF=X", "CHN": "USDCNY=X",
    "IND": "USDINR=X", "KOR": "USDKRW=X", "BRA": "USDBRL=X", "MEX": "USDMXN=X",
    "ZAF": "USDZAR=X", "TUR": "USDTRY=X", "ISR": "USDILS=X", "RUS": "USDRUB=X",
    "POL": "USDPLN=X", "SWE": "USDSEK=X", "NOR": "USDNOK=X", "THA": "USDTHB=X",
    "IDN": "USDIDR=X", "PHL": "USDPHP=X", "MYS": "USDMYR=X", "SGP": "USDSGD=X",
    "HKG": "USDHKD=X", "TWN": "USDTWD=X", "CHL": "USDCLP=X", "COL": "USDCOP=X",
    "ARG": "USDARS=X", "VNM": "USDVND=X", "SAU": "USDSAR=X", "ARE": "USDAED=X",
    "EGY": "USDEGP=X", "NGA": "USDNGN=X", "PAK": "USDPKR=X", "UKR": "USDUAH=X",
    "GBR": "GBPUSD=X", "AUS": "AUDUSD=X", "NZL": "NZDUSD=X",
    # The euro bloc: one instrument, many countries.
    "DEU": "EURUSD=X", "FRA": "EURUSD=X", "ITA": "EURUSD=X", "ESP": "EURUSD=X",
    "NLD": "EURUSD=X", "IRL": "EURUSD=X", "GRC": "EURUSD=X", "PRT": "EURUSD=X",
    "AUT": "EURUSD=X", "BEL": "EURUSD=X", "FIN": "EURUSD=X",
}

# Pairs quoted as USD-per-foreign-unit already rise when the local currency
# strengthens. Pairs quoted the other way round (USDJPY = yen per dollar) rise
# when it *weakens*, so their series is inverted to 1/quote. Without this the
# book would be long half its currencies and short the other half purely as an
# artefact of quoting convention -- a sign error invisible in every diagnostic
# except the P&L.
FX_INVERTED = {t for t in FX_INSTRUMENTS.values() if t.startswith("USD")}

# Continuously-traded instruments that respond to global risk rather than to one
# country. Not comparable to a single-country position, so they stay out of the
# cross-sectional book by default.
FUTURES = {
    "ES=F": "S&P 500 futures",
    "NQ=F": "Nasdaq 100 futures",
    "CL=F": "WTI crude",
    "GC=F": "gold",
    "SI=F": "silver",
    "NG=F": "natural gas",
    "ZN=F": "10-year Treasury note",
    "HG=F": "copper",
    "DX-Y.NYB": "US dollar index",
    "^VIX": "volatility index",
}


def fx_universe():
    """Country -> currency instrument, with quote inversion handled."""
    return Universe(mapping=FX_INSTRUMENTS, invert=FX_INVERTED)


def etf_universe():
    """Country -> single-country equity ETF (the original universe)."""
    return Universe(mapping=COUNTRY_TO_ETF)
