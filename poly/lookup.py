#!/usr/bin/env python3
"""Look up geoid height from a tiled polynomial model.

Usage:
    python -m poly.lookup <lat> <lon> [--model PATH]
    python -m poly.lookup --model poly/europe_geoid.npz 51.8 -1.2
    echo "51.8 -1.2\n48.85 2.35" | python -m poly.lookup --stdin
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from poly.fit import evaluate, _load

DEFAULT_MODEL = str(Path(__file__).parent / "europe_geoid.npz")


def _in_bounds(model, lat, lon):
    lon_min, lat_min, lon_max, lat_max = tuple(model["bounds"].tolist())
    return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)


def lookup(model, lat, lon):
    if not _in_bounds(model, lat, lon):
        lon_min, lat_min, lon_max, lat_max = tuple(model["bounds"].tolist())
        raise ValueError(
            f"({lat}, {lon}) outside model bounds "
            f"lat[{lat_min}, {lat_max}] lon[{lon_min}, {lon_max}]"
        )
    return float(evaluate(model, lat, lon))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lat", type=float, nargs="?")
    p.add_argument("lon", type=float, nargs="?")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"tiled polynomial .npz (default: {DEFAULT_MODEL})")
    p.add_argument("--stdin", action="store_true",
                   help="read 'lat lon' pairs from stdin, one per line")
    p.add_argument("--precision", type=int, default=4)
    args = p.parse_args(argv)

    model = _load(args.model)

    if args.stdin:
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            lat, lon = float(parts[0]), float(parts[1])
            h = lookup(model, lat, lon)
            print(f"{lat} {lon} {h:.{args.precision}f}")
        return

    if args.lat is None or args.lon is None:
        p.error("provide lat and lon (or use --stdin)")

    h = lookup(model, args.lat, args.lon)
    print(f"{h:.{args.precision}f}")


if __name__ == "__main__":
    main()
