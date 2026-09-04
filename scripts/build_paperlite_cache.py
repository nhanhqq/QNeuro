#!/usr/bin/env python3
"""Build PaperLite16 caches sequentially to avoid RAM contention."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qneuro_paperlite.features import build_paperlite_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    contracts = json.loads(Path("configs/dataset_signal_contracts.json").read_text())
    names = args.datasets or list(contracts)
    unknown = sorted(set(names) - set(contracts))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    for name in names:
        config = contracts[name]
        start = time.time()
        path = build_paperlite_cache(
            Path(name) / "X.npy",
            Path(name) / "cache",
            config["sampling_rate"],
            config["frame_seconds"],
            config["hop_seconds"],
            chunk_size=args.chunk_size,
        )
        print(f"PAPERLITE_CACHE dataset={name} path={path} seconds={time.time()-start:.2f}", flush=True)


if __name__ == "__main__":
    main()
