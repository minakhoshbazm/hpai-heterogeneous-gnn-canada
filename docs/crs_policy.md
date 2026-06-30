# Spatial Coordinate Reference System (CRS) Policy

To ensure geometric accuracy, correct area-based feature calculations (like farm and road density), and reliable distance computations, this project enforces a strict, explicit Coordinate Reference System (CRS) policy.

---

## 1. CRS Standards

We define two core reference systems for our data pipeline:

| Use Case | CRS ID | Name / Description | Properties |
| :--- | :--- | :--- | :--- |
| **Analysis & Modeling** | `EPSG:3347` | NAD83 / Statistics Canada Lambert | **Projected (Conformal Conic)**. Preserves shapes and relative angles across Canada. Crucial for distance-based edge weights, road density, buffer creation, and area calculations. |
| **Ingestion & Sharing** | `EPSG:4326` | WGS84 (Latitude / Longitude) | **Geographic (Spherical)**. Standard for GPS telemetry data (eBird, CFIA points), web mapping platforms (Leaflet, GEE), and sharing GeoJSON boundaries. |

---

## 2. Ingestion & Transformation Rules

1. **Preserve Raw CRS on Ingestion:** Keep raw spatial datasets (e.g., raw OSM road networks, boundary shapefiles) in their original format and projection until loaded into the pipeline.
2. **Explicit Reprojection for Analysis:** Prior to any spatial calculations (joins, distance matrices, area metrics, density quotients), reproject all vector layers to `EPSG:3347`:
   ```python
   gdf = gdf.to_crs("EPSG:3347")
   ```
3. **Point Vector Standard:** GPS coordinates (like CFIA outbreak events) must first be instantiated as point geometries in `EPSG:4326`, and then reprojected to `EPSG:3347` before performing spatial joins with census polygons.
4. **Distance and Area Calculations:** All metric calculations (such as calculating polygon land areas or finding nearest major highways) must be executed in a projected CRS (`EPSG:3347`). Running these on geographic WGS84 degrees is strictly prohibited.
5. **Standardized Web Export:** Output vectors generated for web visualization or sharing (e.g., GeoJSON) must be reprojected back to `EPSG:4326` for compatibility.

---

## 3. Spatial Validation Checkpoints
To prevent silent CRS mixing (which results in empty spatial joins or corrupt distance weights), our data pipelines must implement validation checks:
* Verify that the CRS is non-null.
* Ensure both datasets match in projection before performing spatial joins (`assert gdf1.crs == gdf2.crs`).
* Fail loudly with clear log messages if mismatched projections are detected.

---

## 4. Boundary CRS Audit & Geometry Validation

A strict geometric audit was conducted on the boundary datasets:
1. **Raw CSD Boundary Source (`lcsd000b21a_e.shp`):**
   - Verified coordinate reference system is natively **`EPSG:3347`**.
   - Contains 5,161 geometries. Validated shapefiles using `is_valid` check (all 5,161 shapes are topologically closed and valid).
2. **Census Division Geopackage (`census_divisions_epsg3347.gpkg`):**
   - Created by dissolving CSD polygons by their 4-digit prefix.
   - Target projection explicitly set to conformal conic **`EPSG:3347`**.
   - Audit confirmed all 293 dissolved Census Division boundaries are topologically valid.
3. **Interactive/Web GeoJSON (`census_divisions_wgs84.geojson`):**
   - Web export maps are simplified with a tolerance threshold of 100 meters (to reduce footprint from ~900MB to ~59MB) and reprojected to geographic coordinate system **`EPSG:4326`**.
4. **Audit Checkpoint Rule:** To ensure distance decay calculations (inverse distances) and spatial densities are accurate, any coordinate arrays parsed from external feeds (such as GPS markers in CFIA spreadsheets or OSM road shapefiles) must match `EPSG:3347` before geometric joins are processed.
