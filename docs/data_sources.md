# Data Sources: HPAI Canada Project

The data required for this project is staged into three versions (V1 Core, V2 Secondary, V3 Advanced) based on complexity, licensing, and processing requirements.

## Staged Data Directory

| Dataset Name | Project Role | Provider / Source | Spatial Resolution | Temporal Resolution | Format | Priority | Limitations / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CFIA Poultry Outbreaks** | Target Labels | Canadian Food Inspection Agency | Point / Municipality level | Daily (aggregated to Monthly) | CSV | **V1 (Core)** | Locations may be aggregated to counties or provinces for privacy. |
| **StatsCan CD / CAR Boundaries** | Graph Node Definition | Statistics Canada | Vector Boundaries (Polygons) | Decennial (2021) | Shapefile / GeoJSON | **V1 (Core)** | Boundaries can change slightly between census cycles. |
| **Census of Agriculture** | Agricultural Covariates | Statistics Canada | CD / CAR level | 5-yearly (2021) | CSV / Table | **V1 (Core)** | Static between census years; some values suppressed for privacy. |
| **eBird Status & Trends** | eBird Species Abundance | Cornell Lab of Ornithology | 2.96 km grid | Weekly (aggregated to Monthly) | GeoTIFF / Parquet | **V1 (Core)** | Subject to citizen-science reporting bias. Requires access approvals. |
| **GBIF Occurrence Fallback** | Species Occurrences | GBIF | Point coordinates | Continuous | CSV / API | **V1 (Core)** | Highly subject to spatial sampling bias (opportunistic sightings). |
| **Dynamic World Land Cover** | Ecological Features | Google/EE / WRI | 10 m | Continuous (processed monthly) | GEE Image Collection | **V1 (Core)** | Heavy processing required to aggregate to region boundaries. |
| **Sentinel-2 NDVI / NDWI** | Vegetation & Water Anomalies | ESA / Copernicus | 10 m | 5-day revisit | GEE Image Collection | **V1 (Core)** | Subject to cloud cover interference; require median-compositing. |
| **ERA5-Land Climate** | Temp/Precip anomalies | ECMWF / Copernicus | 9 km | Monthly | NetCDF / GEE | **V1 (Core)** | Reanalysis data; might smooth out highly localized microclimatic spikes. |
| **JRC Global Surface Water** | Wetland & surface water | Joint Research Centre | 30 m | Annual / Monthly | GEE Image Collection | **V1 (Core)** | Focuses on open water; may miss small under-canopy wetlands. |
| **Roads & Transport Networks** | Node / Edge Covariates | OpenStreetMap / StatsCan | Vector Lines | Continuous | GeoPackage / Shapefile | **V1 (Core)** | Requires spatial network processing (density, distance calculations). |
| **CWHC Wild Bird Detections** | Secondary Target / Feature | Canadian Wildlife Health Cooperative | Point / CD level | Weekly / Monthly | CSV | **V2 (Secondary)** | Passive surveillance; bias toward areas with human activity. |
| **Sentinel-1 SAR Flooding** | Radar soil moisture / flood | ESA | 20 m | 6–12 day revisit | GEE Image Collection | **V2 (Secondary)** | SAR processing requires specialized calibration (terrain flattening). |
| **Movebank / Motus Tracking** | Migratory priors / corridors | Movebank / Motus networks | GPS / Radio telemetry points | Continuous | Parquet / CSV | **V2 (Secondary)** | Sparse species coverage; biased toward tagged individuals. |
| **Important Bird Areas (IBA)** | Wetland / Habitat priority | BirdLife International | Vector Polygons | Static | Shapefile | **V2 (Secondary)** | Binary habitat inclusion metric; does not reflect year-to-year water level. |
| **Poultry Movement / Trade** | Edge weight connectivity | CFIA / Provincial Agriculture | Farm-to-farm | Daily / Weekly | Database export | **V3 (Advanced)** | Highly confidential; requires strict data sharing agreements (NDA). |
| **WAHIS Outbreak Data** | International context | WOAH / WAHIS | National / State level | Continuous | CSV / API | **V3 (Advanced)** | Delay in reporting; variable surveillance quality across borders. |
| **Viral Clade / Genomics** | Clade similarity graph edges | GISAID / NCIB / CFIA | Genomic sequences | Continuous | FASTA / Metadata | **V3 (Advanced)** | Sequence coverage is uneven; linking sequences to specific farms is hard. |
