"""Download the public datasets used by the non-GDELT benchmarks.

Examples::

    python benchmarks/download_research_data.py aave
    python benchmarks/download_research_data.py dex elliptic

Downloads are resumable through the Hugging Face cache.  Large raw files live
under ``data/`` and are intentionally excluded from git.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent.parent

AAVE_ASSETS = (
    "AAVE-*", "DAI-*", "GHO-*", "LINK-*", "USDC-*", "USDT-*",
    "WBTC-*", "WETH-*", "wstETH-*",
)


def download_aave(data_root: Path) -> None:
    destination = data_root / "amm_events" / "AaveEventData"
    snapshot_download(
        repo_id="Jackson668/AMM-Events",
        repo_type="dataset",
        local_dir=destination,
        allow_patterns=[f"AaveEventData/{pattern}.json" for pattern in AAVE_ASSETS],
    )
    # snapshot_download preserves the repository's top-level folder.  Flatten
    # it to the path consumed by the benchmark and remove only the empty copy.
    nested = destination / "AaveEventData"
    if nested.exists():
        for path in nested.glob("*.json"):
            target = destination / path.name
            if not target.exists():
                shutil.move(str(path), str(target))
        try:
            nested.rmdir()
        except OSError:
            pass
    found = list(destination.glob("*.json"))
    if len(found) != len(AAVE_ASSETS):
        raise RuntimeError(f"expected {len(AAVE_ASSETS)} Aave assets, found {len(found)}")
    print(f"Aave: {len(found)} asset histories -> {destination}")


def download_dex(data_root: Path) -> None:
    destination = data_root / "chainticks"
    snapshot_download(
        repo_id="Chainticks/dex-swaps",
        repo_type="dataset",
        local_dir=destination,
        allow_patterns=["swaps/**/*.parquet"],
    )
    count = len(list((destination / "swaps").glob("**/*.parquet")))
    if count == 0:
        raise RuntimeError("DEX download contains no swap partitions")
    print(f"DEX: {count} swap partitions -> {destination}")


def download_elliptic(data_root: Path) -> None:
    destination = data_root / "elliptic"
    snapshot_download(
        repo_id="yhoma/elliptic-bitcoin-dataset",
        repo_type="dataset",
        local_dir=destination,
        allow_patterns=["*.csv", "README.md"],
    )
    expected = (
        "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv",
        "elliptic_txs_features.csv",
    )
    missing = [name for name in expected if not (destination / name).exists()]
    if missing:
        raise RuntimeError(f"Elliptic download is missing {missing}")
    print(f"Elliptic: {len(expected)} tables -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="+", choices=("aave", "dex", "elliptic", "all"))
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("GRAPH_DATA_ROOT", ROOT / "data")))
    args = parser.parse_args()
    selected = {"aave", "dex", "elliptic"} if "all" in args.dataset else set(args.dataset)
    args.data_root.mkdir(parents=True, exist_ok=True)
    for name, function in (("aave", download_aave), ("dex", download_dex),
                           ("elliptic", download_elliptic)):
        if name in selected:
            function(args.data_root)


if __name__ == "__main__":
    main()
