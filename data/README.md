# Data

## `samples/`

`regions.geojson` contains real study areas with known cloud-free Sentinel-2
coverage, along with suggested date windows. Use them to exercise the analysis
workflow without hunting for a usable region and period.

These are geographic definitions only. They contain no environmental
measurements — every value the product reports is computed from imagery
retrieved at analysis time.

## `fixtures/`

Reserved for deterministic test inputs. The current test suite generates its
synthetic rasters in code (`backend/tests/conftest.py`), where the numerical
edge cases being exercised are visible next to the assertions that depend on
them.

Nothing in this directory is reachable from a production code path.
