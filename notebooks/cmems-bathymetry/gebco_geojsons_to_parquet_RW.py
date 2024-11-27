# %%
import geopandas as gpd
import glob
import pandas as pd
from tqdm import tqdm
from time import time

# %%
# Select year
year = '2024_latminus2'

# File paths in
if year == '2023':
    dir_path_geojsons = r'p:\11209821-cmems-global-sdb\00_miscellaneous\Feasibility_maps\2023\Buffered\AOI_results\*\gebco_2023_*_result.geojson'
elif year == '2024':
    dir_path_geojsons = r'p:\11209821-cmems-global-sdb\00_miscellaneous\Feasibility_maps\2024\AOI_results_2024\gebco_2024_*_result.geojson'
elif year == '2024_latminus1':
    dir_path_geojsons = r'p:\11209821-cmems-global-sdb\00_miscellaneous\Feasibility_maps\2024\AOI_results_2024_latminus1\gebco_2024_*_result.geojson'
elif year == '2024_latminus2':
    dir_path_geojsons = r'p:\11209821-cmems-global-sdb\00_miscellaneous\Feasibility_maps\2024\AOI_results_2024_latminus2\gebco_2024_*_result.geojson'
file_path_geojsons = glob.glob(dir_path_geojsons)

# File paths out
if year == '2023':
    file_path_merge_parquet = r'p:\11209821-cmems-global-sdb\00_miscellaneous\Feasibility_maps\Buffered\AOI_results\gebco_2023_merge_result.parquet'
elif year == '2024':
    file_path_merge_parquet = r'p:\11209821-cmems-global-sdb\00_miscellaneous\Feasibility_maps\2024\gebco_2024_merge_result.parquet'
elif year == '2024_latminus1':
    file_path_merge_parquet = r'p:\11209821-cmems-global-sdb\00_miscellaneous\Feasibility_maps\2024\gebco_2024_latminus1_merge_result.parquet'
elif year == '2024_latminus2':
    file_path_merge_parquet = r'p:\11209821-cmems-global-sdb\00_miscellaneous\Feasibility_maps\2024\gebco_2024_latminus2_merge_result.parquet'

# %%
# Load geodataframes
gdfs = []
for file_path_geojson in tqdm(file_path_geojsons):
    gdfs.append(gpd.read_file(file_path_geojson))

# Merge geodataframes
gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

# %%
# Save merged geodataframe to parquet
gdf.to_parquet(file_path_merge_parquet)

# %%
# Time to load geodataframe from parquet
t0 = time()
gdf = gpd.read_parquet(file_path_merge_parquet)
print(f'Time to load parquet: {time() - t0:.2f} s')

# %%