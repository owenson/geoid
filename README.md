# geoid

Tools for looking up EGM2008 geoid heights over Europe, either directly from
the source GeoTIFF or from a compact polynomial approximation.

Author: Gareth Owenson

## Layout

```
.
├── us_egm2008_europe.tif    # source raster (EGM2008, cropped to Europe)
├── raw/                     # direct GeoTIFF lookup (requires rasterio + the .tif)
│   └── geoid.py
└── poly/                    # polynomial approximation (small .npz, no raster needed at runtime)
    ├── fit.py               # fit a model from the GeoTIFF
    ├── lookup.py            # CLI for evaluating a fitted model
    ├── europe_geoid.npz     # default tiled model
    ├── geoid_tiled.npz
    └── geoid_poly_d20.npz
```

## Source raster

The GeoTIFF is produced from the NGA EGM2008 1′ grid, cropped to a European
bounding box:

```
gdalwarp -te -10 35 45 71 -te_srs EPSG:4326 -r max \
    us_nga_egm2008_1.tif us_egm2008_europe.tif
```

## Raw GeoTIFF lookup

`raw/geoid.py` reads the GeoTIFF directly and returns either the
nearest-pixel value or a bilinearly interpolated height. Needs `rasterio` and
the `.tif` at runtime.

```python
from raw.geoid import get_geoid_height, interpolated_geoid_height

get_geoid_height("us_egm2008_europe.tif", 51.8, -1.2)
interpolated_geoid_height("us_egm2008_europe.tif", 51.8, -1.2)
```

## Polynomial approximation

`poly/` contains a bivariate polynomial fit of the raster. Models can be a
single global polynomial or a uniform grid of per-tile polynomials. At runtime
only the `.npz` and `numpy` are needed — the GeoTIFF is not required.

### Fit a model

```bash
# global polynomial
python -m poly.fit fit us_egm2008_europe.tif poly/geoid_poly_d20.npz --degree 20

# tiled polynomials (recommended)
python -m poly.fit fit us_egm2008_europe.tif poly/europe_geoid.npz \
    --degree 8 --stride 4 --tiles 8 6
```

### Evaluate

```bash
# via the dedicated CLI (defaults to poly/europe_geoid.npz)
python -m poly.lookup 51.8 -1.2
python -m poly.lookup --model poly/geoid_tiled.npz 48.85 2.35

# batch from stdin
printf "51.8 -1.2\n48.85 2.35\n" | python -m poly.lookup --stdin

# via poly.fit
python -m poly.fit eval poly/europe_geoid.npz 51.8 -1.2
```

### Report fit error against the raster

```bash
python -m poly.fit error us_egm2008_europe.tif poly/europe_geoid.npz --stride 1
```
