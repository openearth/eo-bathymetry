# %%
# Notes
'''
Google Earth Engine accepts the following types: tif, shp and csv
- Tif files are used for raster data, and cannot be created row by row, structured
- Shp files have a maximum size of 2GB, and cannot be created row by row
- Csv files can be created row by row, and can be used for vector data <--
'''

# %%
# Packages
import glob
import numpy as np
import os
import time
import xarray as xr
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

import ee
#ee.Authenticate()
ee.Initialize()

# %%
def export_data(his, file_path, variables=['waterlevel'], row_limit=np.inf):
    '''
    Function to write GTSM data to a csv file.
    '''
    # Get Data
    station_idxs = his.stations.values
    lons = his.station_x_coordinate.values
    lats = his.station_y_coordinate.values
    times = his.time.values

    # Remove times 

    # Convert times to milliseconds
    str_times = np.array([str(time) for time in times])
    int_times = np.array([time.astype('datetime64[ms]').astype('int64') for time in times])

    # Write data to file
    with open(file_path, 'w') as f:
        # Write header
        header_str = 'geometry,system:time_start,system:time_end,station,times'
        header_str += ''.join([',{}'.format(var) for var in variables]) + '\n'
        f.write(header_str)

        # Get data per station
        row_idx = 0
        for station_idx in tqdm(station_idxs, desc='Stations', leave=True):
            # Determine non-nan time indices based on first variable
            time_idxs = np.where(~np.isnan(his.isel(stations=station_idx)[variables[0]].values))[0]

            # Get data for station
            data = his.isel(stations=station_idx, time=time_idxs)[variables]
            
            # Get values
            data = {var: data[var].values for var in variables}

            # Get data per time step
            for time_idx in time_idxs:
                # Write data per station and time step
                row_str = 'POINT ({lon} {lat}),{sys_stime},{sys_etime},{station},{time}'.format(
                    lon=lons[station_idx], lat=lats[station_idx], sys_stime=int_times[time_idx], sys_etime=int_times[time_idx],
                    station=station_idx, time=str_times[time_idx])
                row_str += ''.join([','+str(data[var][time_idx].item()) for var in variables]) + '\n'
                f.write(row_str)
            
                # Break if row limit is reached
                row_idx += 1
                if row_idx >= row_limit:
                    break
            
            # Break if row limit is reached
            if row_idx >= row_limit:
                break

# %%
# Settings
# File paths
file_path_his_ncs = r'p:\1230882-emodnet_hrsm\GTSMv3.0EMODnet\CMEMS_intertidal_SDB\*\output\gtsm_model_0000_his.nc'
file_path_his_ncs = glob.glob(file_path_his_ncs)
dir_path_out = r'p:\11209821-cmems-global-sdb\01_intertidal\02_data\02_gtsm_files'

# %%
# Export stations to geojson and parquet
get_geojson_and_parquet = False
if get_geojson_and_parquet:
    # Get station files
    file_path_his_nc = file_path_his_ncs[2]

    # Open his file
    his = xr.open_dataset(file_path_his_nc)

    # Get Data
    station_idxs = his.stations.values
    lons = his.station_x_coordinate.values
    lats = his.station_y_coordinate.values

    # Get stations
    stations = [Point(lon, lat) for lon, lat in zip(lons, lats)] 
    gdf_stations = gpd.GeoDataFrame(data={'station': station_idxs, 'geometry': stations}, crs='EPSG:4326')

    # Get number of time steps per station per year
    for file_path_his_nc in file_path_his_ncs[1:]:
        # Get year
        year = int(file_path_his_nc.split('\\')[-3].split('_')[-1])
        print('Processing year: {}'.format(year))

        # Open his file
        t0 = time.time()
        his = xr.open_dataset(file_path_his_nc)
        print('{} s to open his file'.format(np.round(time.time()-t0, 2)))

        # Get his within the year (remove spin-up time)
        his_year = his.sel(time=slice('{}-01-01'.format(year), '{}-12-31'.format(year)))

        variables = ['waterlevel']
        n_timesteps = []
        for station_idx in tqdm(station_idxs, desc='Stations', leave=True):
            # Determine non-nan time indices based on first variable
            times = his_year[variables[0]].isel(stations=station_idx).dropna(dim='time').time

            # Number of time steps
            n_timesteps.append(times.size)

        # Add number of time steps to geodataframe
        gdf_stations['n_timesteps_{}'.format(year)] = n_timesteps
    
    # Export stations to geojson
    file_path_stations_geojson = os.path.join(dir_path_out, 'gtsm_stations.geojson')
    gdf_stations.to_file(file_path_stations_geojson, driver='GeoJSON')

    # Export stations to parquet
    file_path_stations_parquet = os.path.join(dir_path_out, 'gtsm_stations.parquet')
    gdf_stations.to_parquet(file_path_stations_parquet)

# %%
# Export data to csvs
get_csvs = False
if get_csvs:
    for file_path_his_nc in file_path_his_ncs[3:]:
        # Get year
        year = int(file_path_his_nc.split('\\')[-3].split('_')[-1])
        print('Processing year: {}'.format(year))
        
        # Open his file
        t0 = time.time()
        his = xr.open_dataset(file_path_his_nc)
        print('{} s to open his file'.format(np.round(time.time()-t0, 2)))

        # Get his within the year (remove spin-up time)
        his_year = his.sel(time=slice('{}-01-01'.format(year), '{}-12-31'.format(year)))
        
        # Export data to csv
        t0 = time.time()
        file_path_csv = os.path.join(dir_path_out, 'gtsm_waterlevels_{}.csv'.format(year))
        export_data(his_year, file_path_csv)
        print('{} s to export data to csv'.format(np.round(time.time()-t0, 2)))

# %%
# Copy first file to merged file
file_path_2018 = r'p:\11209821-cmems-global-sdb\01_intertidal\02_data\02_gtsm_files\gtsm_waterlevels_2018.csv'
file_path_2019 = r'p:\11209821-cmems-global-sdb\01_intertidal\02_data\02_gtsm_files\gtsm_waterlevels_2019.csv'

# Copy first x rows
CHUNK_SIZE = 105_120 * 5
chunk_container = pd.read_csv(file_path_2019, chunksize=CHUNK_SIZE)
chunk = next(chunk_container)
chunk.to_csv(file_path_2018, mode='w', header=True, index=False)

# %%
# Merge csvs files
merge_csvs = True
if merge_csvs:
    # File paths
    file_path_csvs = glob.glob(os.path.join(dir_path_out, 'gtsm_waterlevels_*.csv'))
    file_path_csvs = [file_path for file_path in file_path_csvs if 'merged' not in file_path]
    file_path_csvs.sort()
    years = [int(os.path.basename(file_path).split('_')[-1].split('.')[0]) for file_path in file_path_csvs]
    years.sort()
    file_path_csv_merged = os.path.join(dir_path_out, 'gtsm_waterlevels_merged_{}_{}_test.csv'.format(years[0], years[-1]))

    # Copy first file to merged file
    import shutil
    t0 = time.time()
    shutil.copyfile(file_path_csvs[0], file_path_csv_merged)
    print('{} s to copy {}'.format(np.round(time.time()-t0, 2), os.path.basename(file_path_csvs[0])))

    # Append other files to merged file
    for file_path_csv in file_path_csvs[1:]:
        print('Merging file: {}'.format(os.path.basename(file_path_csv)))
        t0 = time.time()

        # Merge file to merged file
        CHUNK_SIZE = 20_000_000
        CHUNK_SIZE = 105_120
        chunk_container = pd.read_csv(file_path_csv, chunksize=CHUNK_SIZE)
        i = 0
        for chunk in chunk_container:
            chunk.to_csv(file_path_csv_merged, mode='a', header=False, index=False)
            i += 1
            if i > 5:
                break

        # Print time
        print('{} s to append {}'.format(np.round(time.time()-t0, 2), os.path.basename(file_path_csv)))

# %%
# Function to upload file to Google Cloud Storage
import gcsfs
def upload_file_to_google_cloud_storage(gcs_project: str, file_path_gcc: str, bucket_name: str, bucket_proj: str, file_path: str):
    # Get file system interface for google cloud storage
    fs = gcsfs.GCSFileSystem(gcs_project, token=file_path_gcc)

    # Get target path
    target_path = 'gs://{}/{}/{}'.format(bucket_name, bucket_proj, os.path.basename(file_path))

    # Remove target file if it already exists
    if fs.exists(target_path):
        print('Removing existing file: {}'.format(target_path))
        fs.rm(target_path, recursive=True)
    
    # Saved file to google cloud
    try:
        fs.put(file_path, target_path, recursive=True, fixed_key_metadata={'cache_control': 'no-cache', 'Cache-Control': 'no-cache', 'cache-control': 'no-cache'})
        print('Successfully uploaded file: {}'.format(target_path))
    except OSError as e:
        print('Failed to upload file: {}: {}'.format(target_path, e))

# Settings
for year in years:
    gcs_project = 'cmems-sdb'
    file_path_gcc = r'p:\11209821-cmems-global-sdb\00_miscellaneous\KEYS\bathymetry-543b622ddce7.json'
    bucket_name = 'cmems-sdb'
    bucket_proj = 'gtsm'
    file_path = r'p:\11209821-cmems-global-sdb\01_intertidal\02_data\02_gtsm_files\gtsm_waterlevels_{}.csv'.format(year)

    # Upload file to Google Cloud Bucket
    upload_file_to_google_cloud_storage(gcs_project, file_path_gcc, bucket_name, bucket_proj, file_path)

# %%
# Transfer csv file from Google Cloud Bucket to a Google Earth Engine Asset
# Settings
import os
for year in tqdm(years):
    if year == 2018:
        continue
    google_cloud_bucket_file_path = r'gs://cmems-sdb/gtsm/gtsm_waterlevels_{}.csv'.format(year)
    asset_id = r'projects/bathymetry/assets/gtsm_waterlevels_{}'.format(year)
    command = r'earthengine upload table --asset_id={} {}'.format(asset_id, google_cloud_bucket_file_path)
    out = os.system(command)

    # Check if the task was successful
    if out == 0:
        print('Task submitted successfully')
    else:
        print('Task failed')

# %%
# OLD

# %%
# Get the Google Earth Engine Asset
import ee
ee.Initialize()

print('Getting the Google Earth Engine Asset')
#gtsm_wls = ee.FeatureCollection('users/rubenwhitenl/gtsm_waterlevels')
gtsm_wls = ee.FeatureCollection('projects/bathymetry/assets/gtsm_waterlevels')
gtsm_wls2 = gtsm_wls.filterDate('2021-01-01', '2021-01-02')
gtsm_wls3 = gtsm_wls2.filterBounds(ee.Geometry.Rectangle(23, 25, 33, 35))

print('Printing the Google Earth Engine Asset')
#print('Length of gtsm_wls: {}'.format(gtsm_wls.size().getInfo()))
print('Length of gtsm_wls2: {}'.format(gtsm_wls2.size().getInfo()))
print('Length of gtsm_wls3: {}'.format(gtsm_wls3.size().getInfo()))

# %%
# Calculate the highest and lowest astronomical tides per station
min_tides = np.ones(station_idxs.shape)*np.nan
max_tides = np.ones(station_idxs.shape)*np.nan

for station_idx in tqdm(station_idxs):
    # Determine the non-nan time indices based on the Waterlevel variable
    time_idxs = np.where(~np.isnan(his.isel(stations=station_idx)['waterlevel'].values))[0]

    # Get the waterlevels for the station
    waterlevels = his.isel(stations=station_idx, time=time_idxs)['waterlevel'].values

    # Get the minimum and maximum waterlevels
    min_tides[station_idx] = np.min(waterlevels)
    max_tides[station_idx] = np.max(waterlevels)

# %% =============================================================================
# OLD
# %% =============================================================================
# # %%
# # File paths
# file_path_his_nc = r'p:\1230882-emodnet_hrsm\GTSMv3.0EMODnet\CMEMS_intertidal_SDB\r001\output\gtsm_model_0000_his.nc'
# dir_path_out = r'p:\11209821-cmems-global-sdb\01_intertidal\02_data\02_gtsm_files'

# # %%
# # Open files
# t0 = time.time()
# his = xr.open_dataset(file_path_his_nc)
# print('{} s to open his file'.format(np.round(time.time()-t0, 2)))

# # %%
# # Get Data
# t0 = time.time()
# station_idxs = his.stations.values
# lons = his.station_x_coordinate.values
# lats = his.station_y_coordinate.values
# times = his.time.values
# t1 = time.time()

# # Convert times to milliseconds
# str_times = np.array([str(time) for time in times])
# int_times = np.array([time.astype('datetime64[ms]').astype('int64') for time in times])

# # Get start and end times
# stime = times[0]
# etime = times[-1]

# # %%
# # Export stations as geojson
# # Create points
# points = [Point(lon, lat) for lon, lat in zip(lons, lats)]

# # Create geodataframe
# gdf_stations = gpd.GeoDataFrame(geometry=points, crs='EPSG:4326')

# # Export to geojson
# gdf_stations.to_file(os.path.join(dir_path_out, 'gtsm_stations.geojson'), driver='GeoJSON')

# # %%
# # Write data to csv
# '''
# with open(os.path.join(dir_path_out, 'gtsm.csv'), 'w') as f:
#     # Write header
#     f.write('geometry,system:time_start,system:time_end,station,times,waterlevels\n')

#     # Write data per station
#     for idx in tqdm(station_idxs):
#         # Get waterlevels
#         waterlevels = his.waterlevel.isel(stations=idx).values

#         # Remove nans
#         no_nans = np.where(~np.isnan(waterlevels))[0]
#         waterlevels_ = waterlevels[no_nans]
#         times_ = times[no_nans]

#         # If all nans, remove station
#         if len(waterlevels_) == 0:
#             continue

#         # Write data
#         f.write('POINT ({lon} {lat}),{sys_stime},{sys_etime},{station},"{times}","{waterlevels}"\n'.format(
#             lon=lons[idx], lat=lats[idx], sys_stime=times[0], sys_etime=times[-1], 
#             station=idx, times=list(times_), waterlevels=list(waterlevels_)))
# '''


# # %%
# # Write data to csv 2
# variables = ['waterlevel']

# with open(os.path.join(dir_path_out, 'gtsm_waterlevels.csv'), 'w') as f:
#     print('Header')
#     # Write header
#     header_str = 'geometry,system:time_start,system:time_end,station,times'
#     header_str += ''.join([',{}'.format(var) for var in variables]) + '\n'
#     f.write(header_str)

#     # Get data per station
#     for station_idx in tqdm(station_idxs, desc='Stations', leave=True):
#         # Determine non-nan time indices based on first variable
#         time_idxs = np.where(~np.isnan(his.isel(stations=station_idx)[variables[0]].values))[0]

#         # Get data for station
#         data = his.isel(stations=station_idx, time=time_idxs)[variables]
        
#         # Get values
#         data = {var: data[var].values for var in variables}

#         # Get data per time step
#         for time_idx in time_idxs:
#             # Write data per station and time step
#             row_str = 'POINT ({lon} {lat}),{sys_stime},{sys_etime},{station},{time}'.format(
#                 lon=lons[station_idx], lat=lats[station_idx], sys_stime=int_times[time_idx], sys_etime=int_times[time_idx],
#                 station=station_idx, time=str_times[time_idx])
#             row_str += ''.join([','+str(data[var][time_idx].item()) for var in variables]) + '\n'
#             f.write(row_str)
