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
import numpy as np
import os
import time
import xarray as xr
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point

import ee
ee.Initialize()

# %%
# File paths
file_path_his_nc = r'p:\1230882-emodnet_hrsm\GTSMv3.0EMODnet\CMEMS_intertidal_SDB\r001\output\gtsm_model_0000_his.nc'
dir_path_out = r'p:\11209821-cmems-global-sdb\01_intertidal\02_data\02_gtsm_files'

# %%
# Open files
t0 = time.time()
his = xr.open_dataset(file_path_his_nc)
print('{} s to open his file'.format(np.round(time.time()-t0, 2)))

# %%
# Get Data
t0 = time.time()
station_idxs = his.stations.values
lons = his.station_x_coordinate.values
lats = his.station_y_coordinate.values
times = his.time.values
t1 = time.time()

# Convert times to milliseconds
str_times = np.array([str(time) for time in times])
int_times = np.array([time.astype('datetime64[ms]').astype('int64') for time in times])

# Get start and end times
stime = times[0]
etime = times[-1]

# %%
# Export stations as geojson
# Create points
points = [Point(lon, lat) for lon, lat in zip(lons, lats)]

# Create geodataframe
gdf_stations = gpd.GeoDataFrame(geometry=points, crs='EPSG:4326')

# Export to geojson
gdf_stations.to_file(os.path.join(dir_path_out, 'gtsm_stations.geojson'), driver='GeoJSON')

# %%
# Write data to csv
'''
with open(os.path.join(dir_path_out, 'gtsm.csv'), 'w') as f:
    # Write header
    f.write('geometry,system:time_start,system:time_end,station,times,waterlevels\n')

    # Write data per station
    for idx in tqdm(station_idxs):
        # Get waterlevels
        waterlevels = his.waterlevel.isel(stations=idx).values

        # Remove nans
        no_nans = np.where(~np.isnan(waterlevels))[0]
        waterlevels_ = waterlevels[no_nans]
        times_ = times[no_nans]

        # If all nans, remove station
        if len(waterlevels_) == 0:
            continue

        # Write data
        f.write('POINT ({lon} {lat}),{sys_stime},{sys_etime},{station},"{times}","{waterlevels}"\n'.format(
            lon=lons[idx], lat=lats[idx], sys_stime=times[0], sys_etime=times[-1], 
            station=idx, times=list(times_), waterlevels=list(waterlevels_)))
'''

# %%
# Write data to csv 2
variables = ['waterlevel']

with open(os.path.join(dir_path_out, 'gtsm_v2.csv'), 'w') as f:
    print('Header')
    # Write header
    header_str = 'geometry,system:time_start,system:time_end,station,times'
    header_str += ''.join([',{}'.format(var) for var in variables]) + '\n'
    f.write(header_str)

    # Get data per station
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
            
# %%
# Notes
'''
Manually upload the csv files to the Google Cloud Bucket.
'''

# %%
# Transfer csv file from Google Cloud Bucket to a Google Earth Engine Asset
google_cloud_bucket_file_path = 'gs://cmems-sdb/gtsm/gtsm_waterlevels.csv'
#asset_id = 'users/rubenwhitenl/gtsm_waterlevels'
asset_id = 'projects/bathymetry/assets/gtsm_waterlevels'
command = 'earthengine upload table --asset_id={} {}'.format(asset_id, google_cloud_bucket_file_path)
out = os.system(command)

# Check if the task was successful
if out == 0:
    print('Task submitted successfully');
else:
    print('Task failed');

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

# %%


