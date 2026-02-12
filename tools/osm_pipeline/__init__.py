"""OSM offline preprocessing pipeline.

Converts OpenStreetMap data into numpy tile files (.npz) for UAVBench
realistic scenarios. Requires heavy geospatial dependencies (osmnx,
geopandas, rasterio, shapely) -- install via:

    pip install -e ".[pipeline]"

This package is NOT part of the uavbench runtime. It runs offline to
generate data consumed at runtime by uavbench environments.
"""
