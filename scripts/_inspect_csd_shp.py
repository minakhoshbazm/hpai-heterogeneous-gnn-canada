import geopandas as gpd
shp = r'd:/gdrive/models/hpai-heterogeneous-gnn-canada/data/raw/boundaries/lcsd000b21a_e.shp'
gdf = gpd.read_file(shp)
print('Columns:', gdf.columns.tolist())
print(gdf.head())
