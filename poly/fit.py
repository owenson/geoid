#!/usr/bin/env python3
"""Fit and evaluate a bivariate polynomial approximation of the geoid GeoTIFF.

Supports a single global polynomial or a uniform grid of per-tile polynomials.

Usage:
    python -m poly.fit fit   <geotiff> <out.npz> [--degree N] [--stride S]
                                                 [--tiles NLON NLAT]
    python -m poly.fit eval  <out.npz> <lat> <lon>
    python -m poly.fit error <geotiff> <out.npz> [--stride S]
"""
import argparse
import sys
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy


def _grid(src, stride=1):
    """Return (lon, lat, z) flat arrays for every `stride`-th pixel (nodata removed)."""
    band = src.read(1)
    if stride > 1:
        band = band[::stride, ::stride]
    rows, cols = np.indices(band.shape)
    rows = rows * stride
    cols = cols * stride
    xs, ys = xy(src.transform, rows.ravel(), cols.ravel())
    lon = np.asarray(xs)
    lat = np.asarray(ys)
    z = band.ravel().astype(np.float64)
    if src.nodata is not None:
        mask = z != src.nodata
        lon, lat, z = lon[mask], lat[mask], z[mask]
    return lon, lat, z


def _normalize(lon, lat, bounds):
    """Map lon/lat into [-1, 1] using bounds so polynomial basis is well-conditioned."""
    lon_min, lat_min, lon_max, lat_max = bounds
    u = 2 * (lon - lon_min) / (lon_max - lon_min) - 1
    v = 2 * (lat - lat_min) / (lat_max - lat_min) - 1
    return u, v


def _fit_one(u, v, z, degree):
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    A = np.column_stack([u ** i * v ** j for i, j in terms])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs, terms, A @ coeffs


def fit(geotiff_path, out_path, degree=8, stride=4, tiles=None):
    with rasterio.open(geotiff_path) as src:
        lon, lat, z = _grid(src, stride=stride)
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

    lon_min, lat_min, lon_max, lat_max = bounds

    if tiles is None:
        u, v = _normalize(lon, lat, bounds)
        coeffs, terms, pred = _fit_one(u, v, z, degree)
        resid = z - pred
        np.savez(
            out_path,
            tiled=np.bool_(False),
            coeffs=coeffs,
            terms=np.array(terms, dtype=np.int16),
            bounds=np.array(bounds, dtype=np.float64),
            degree=np.int16(degree),
        )
        print(f"fit: {len(z)} samples, degree={degree}, terms={len(terms)}")
        print(f"rmse={np.sqrt(np.mean(resid**2)):.4f} m   "
              f"max|err|={np.max(np.abs(resid)):.4f} m")
        print(f"saved {out_path}")
        return

    n_lon, n_lat = tiles
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    n_terms = len(terms)
    tile_coeffs = np.zeros((n_lat, n_lon, n_terms), dtype=np.float64)

    lon_edges = np.linspace(lon_min, lon_max, n_lon + 1)
    lat_edges = np.linspace(lat_min, lat_max, n_lat + 1)

    # Assign every sample to a tile (clip to valid range).
    ix = np.clip(np.searchsorted(lon_edges, lon, side="right") - 1, 0, n_lon - 1)
    iy = np.clip(np.searchsorted(lat_edges, lat, side="right") - 1, 0, n_lat - 1)

    total_sq = 0.0
    total_n = 0
    max_abs = 0.0
    empty = 0
    for j in range(n_lat):
        for i in range(n_lon):
            m = (ix == i) & (iy == j)
            if m.sum() < n_terms:
                empty += 1
                continue
            tb = (lon_edges[i], lat_edges[j], lon_edges[i + 1], lat_edges[j + 1])
            u, v = _normalize(lon[m], lat[m], tb)
            c, _, pred = _fit_one(u, v, z[m], degree)
            tile_coeffs[j, i] = c
            resid = z[m] - pred
            total_sq += float(np.sum(resid ** 2))
            total_n += int(m.sum())
            max_abs = max(max_abs, float(np.max(np.abs(resid))))

    np.savez(
        out_path,
        tiled=np.bool_(True),
        tile_coeffs=tile_coeffs.astype(np.float32),
        terms=np.array(terms, dtype=np.int16),
        bounds=np.array(bounds, dtype=np.float64),
        tiles=np.array([n_lon, n_lat], dtype=np.int32),
        degree=np.int16(degree),
    )
    rmse = np.sqrt(total_sq / max(total_n, 1))
    print(f"fit: {total_n} samples, degree={degree}, tiles={n_lon}x{n_lat}, "
          f"terms/tile={n_terms} (empty tiles: {empty})")
    print(f"rmse={rmse:.4f} m   max|err|={max_abs:.4f} m")
    print(f"saved {out_path}")


def _load(path):
    return np.load(path)


def _poly_eval(coeffs, terms, u, v):
    total = np.zeros_like(np.asarray(u, dtype=np.float64))
    for c, (i, j) in zip(coeffs, terms):
        total = total + c * (u ** i) * (v ** j)
    return total


def evaluate(model, lat, lon):
    """Evaluate the model (global or tiled) at scalar or array lat/lon."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    terms = model["terms"]
    bounds = tuple(model["bounds"].tolist())

    if not bool(model["tiled"]):
        u, v = _normalize(lon, lat, bounds)
        return _poly_eval(model["coeffs"], terms, u, v)

    n_lon, n_lat = (int(x) for x in model["tiles"])
    tile_coeffs = model["tile_coeffs"]
    lon_min, lat_min, lon_max, lat_max = bounds
    lon_edges = np.linspace(lon_min, lon_max, n_lon + 1)
    lat_edges = np.linspace(lat_min, lat_max, n_lat + 1)

    lon_a = np.atleast_1d(lon)
    lat_a = np.atleast_1d(lat)
    ix = np.clip(np.searchsorted(lon_edges, lon_a, side="right") - 1, 0, n_lon - 1)
    iy = np.clip(np.searchsorted(lat_edges, lat_a, side="right") - 1, 0, n_lat - 1)

    out = np.empty_like(lon_a, dtype=np.float64)
    # Evaluate grouped by (ix, iy) so we normalize with the tile's bounds.
    flat_idx = iy * n_lon + ix
    for key in np.unique(flat_idx):
        sel = flat_idx == key
        j, i = divmod(int(key), n_lon)
        tb = (lon_edges[i], lat_edges[j], lon_edges[i + 1], lat_edges[j + 1])
        u, v = _normalize(lon_a[sel], lat_a[sel], tb)
        out[sel] = _poly_eval(tile_coeffs[j, i], terms, u, v)
    return out if np.ndim(lon) else float(out[0])


def eval_cmd(npz_path, lat, lon):
    model = _load(npz_path)
    h = evaluate(model, lat, lon)
    print(f"polynomial geoid height at ({lat}, {lon}): {float(h):.4f} m")


def error_cmd(geotiff_path, npz_path, stride=1):
    model = _load(npz_path)
    with rasterio.open(geotiff_path) as src:
        lon, lat, z = _grid(src, stride=stride)
    pred = evaluate(model, lat, lon)
    resid = z - pred
    print(f"samples={len(z)}  rmse={np.sqrt(np.mean(resid**2)):.4f} m  "
          f"max|err|={np.max(np.abs(resid)):.4f} m  "
          f"mean={np.mean(resid):+.4f} m")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit")
    pf.add_argument("geotiff")
    pf.add_argument("out")
    pf.add_argument("--degree", type=int, default=8)
    pf.add_argument("--stride", type=int, default=4)
    pf.add_argument("--tiles", type=int, nargs=2, metavar=("NLON", "NLAT"),
                    default=None, help="fit a uniform grid of per-tile polynomials")

    pe = sub.add_parser("eval")
    pe.add_argument("npz")
    pe.add_argument("lat", type=float)
    pe.add_argument("lon", type=float)

    perr = sub.add_parser("error")
    perr.add_argument("geotiff")
    perr.add_argument("npz")
    perr.add_argument("--stride", type=int, default=1)

    args = p.parse_args(argv)
    if args.cmd == "fit":
        fit(args.geotiff, args.out, degree=args.degree, stride=args.stride,
            tiles=tuple(args.tiles) if args.tiles else None)
    elif args.cmd == "eval":
        eval_cmd(args.npz, args.lat, args.lon)
    elif args.cmd == "error":
        error_cmd(args.geotiff, args.npz, stride=args.stride)


if __name__ == "__main__":
    main()
