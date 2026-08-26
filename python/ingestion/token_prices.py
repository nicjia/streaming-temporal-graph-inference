"""
Map Ethereum contract addresses to priced tokens, and pull daily price history.

This is what turns the on-chain graph result into a financial one. The model
predicts which contract a wallet will interact with for the first time; if those
contracts are priced tokens, then aggregated first-time adoption becomes a
capital-flow signal that can be tested against returns.

Most heavily-called contracts are routers, aggregators and pools rather than
tokens, and those have no price. The lookup filters them out for free: DefiLlama
returns nothing for a contract that is not a priced asset.
"""

import time

import numpy as np
import pandas as pd
import requests

BASE = "https://coins.llama.fi"
CHAIN = "ethereum"


def resolve_tokens(addresses, batch=40, pause=0.25, quiet=False):
    """
    Keep the addresses DefiLlama prices, with their symbols.

    Args:
        addresses: iterable of lowercase hex strings without the 0x prefix.

    Returns:
        DataFrame [address, symbol, price, decimals].
    """
    addresses = [a.lower().removeprefix("0x") for a in addresses]
    rows = []
    for start in range(0, len(addresses), batch):
        chunk = addresses[start:start + batch]
        key = ",".join(f"{CHAIN}:0x{a}" for a in chunk)
        try:
            payload = requests.get(f"{BASE}/prices/current/{key}", timeout=30).json()
        except Exception:
            time.sleep(2.0)
            continue
        for address in chunk:
            entry = payload.get("coins", {}).get(f"{CHAIN}:0x{address}")
            if entry and entry.get("price"):
                rows.append((address, entry.get("symbol", "?"),
                             float(entry["price"]), entry.get("decimals")))
        time.sleep(pause)

    frame = pd.DataFrame(rows, columns=["address", "symbol", "price", "decimals"])
    if not quiet:
        print(f"Priced tokens: {len(frame)} of {len(addresses)} contracts "
              f"({len(frame)/max(len(addresses),1):.0%}); the rest are routers, "
              f"pools and other unpriced contracts")
    return frame


MAX_SPAN = 300  # the chart endpoint rejects longer spans with HTTP 400


def price_history(addresses, start_ts, days, pause=0.25, quiet=False):
    """
    Daily price series per address.

    Requested per address and chunked in windows of MAX_SPAN days. The chart
    endpoint hard-fails above roughly 400 days rather than truncating, so a
    single long request for a multi-year window returns nothing at all -- which
    is easy to mistake for "this token has no price history".

    Returns:
        DataFrame [date, ticker, close, address].
    """
    rows = []
    for index, address in enumerate(addresses):
        symbol = None
        cursor = int(start_ts)
        remaining = int(days)
        while remaining > 0:
            span = min(MAX_SPAN, remaining)
            url = (f"{BASE}/chart/{CHAIN}:0x{address}"
                   f"?start={cursor}&span={span}&period=1d")
            try:
                payload = requests.get(url, timeout=30).json()
            except Exception:
                time.sleep(2.0)
                remaining -= span
                cursor += span * 86400
                continue
            entry = payload.get("coins", {}).get(f"{CHAIN}:0x{address}")
            if entry:
                symbol = symbol or entry.get("symbol", address[:8])
                for point in entry.get("prices", []):
                    rows.append((pd.Timestamp(point["timestamp"], unit="s").date().isoformat(),
                                 symbol, float(point["price"]), address))
            remaining -= span
            cursor += span * 86400
            time.sleep(pause)
        if not quiet and index % 25 == 0:
            print(f"  {index}/{len(addresses)} price series", flush=True)

    frame = pd.DataFrame(rows, columns=["date", "ticker", "close", "address"])
    frame = frame.drop_duplicates(subset=["date", "ticker"], keep="last")
    if not quiet and len(frame):
        print(f"Price history: {frame['ticker'].nunique()} tokens, "
              f"{frame['date'].nunique()} days, {len(frame):,} observations")
    return frame


def adoption_panel(events, contracts, freq="W", first_time_only=True):
    """
    Count wallets touching each contract per period.

    `first_time_only` restricts to a wallet's *first* interaction with that
    contract, which is the adoption event the model predicts. Repeat traffic is
    dominated by bots and routers and swamps the signal.

    Returns:
        DataFrame [date, contract, adopters].
    """
    frame = events[events["dst"].isin(contracts)].copy()
    if first_time_only:
        frame = frame.sort_values("ts").drop_duplicates(["src", "dst"], keep="first")
    frame["date"] = pd.to_datetime(frame["ts"], unit="s").dt.to_period(freq).dt.start_time
    panel = frame.groupby(["date", "dst"]).size().reset_index(name="adopters")
    return panel.rename(columns={"dst": "contract"})
