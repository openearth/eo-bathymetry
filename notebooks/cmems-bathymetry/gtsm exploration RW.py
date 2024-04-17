# %%
# Packages
import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import time
import xarray as xr
ee.Initialize()

# %%
# File paths
file_path_his_nc = r'p:\1230882-emodnet_hrsm\GTSMv3.0EMODnet\CMEMS_intertidal_SDB\r001\output\gtsm_model_0000_his.nc'
file_path_map_nc = r'p:\1230882-emodnet_hrsm\GTSMv3.0EMODnet\CMEMS_intertidal_SDB\r001\output\gtsm_model_0000_map.nc'

# Bounds North Sea
lon_bounds = [2, 4]
lat_bounds = [51, 53]
time_bounds = ['2021-06-01', '2021-07-01']
time_bounds = [np.datetime64(time_bounds[0]), np.datetime64(time_bounds[1])]

# %%
# Open file
t0 = time.time()
his = xr.open_dataset(file_path_his_nc)
print('{} s to open file'.format(np.round(time.time()-t0, 2)))

# %%
# Get longitude, latitude, time and water levels
t0 = time.time()
lons = his.station_x_coordinate.values
lats = his.station_y_coordinate.values
times = his.time.values
waterlevels = his.waterlevel.values

# %%
# Get station and time indices
station_idxs = np.where((lats >= lat_bounds[0]) & (lats <= lat_bounds[1]) & (lons >= lon_bounds[0]) & (lons <= lon_bounds[1]))[0]
time_idxs = np.where((times >= time_bounds[0]) & (times <= time_bounds[1]))[0]

# Get longitude [deg], latitude [deg], time [ms] and water levels [m]
lats = lats[station_idxs]
lons = lons[station_idxs]
times = times[time_idxs]
waterlevels = his.waterlevel.isel(stations=station_idxs, time=time_idxs).values
waterlevels = np.where(np.isnan(waterlevels), 0, waterlevels)

print('{} s to get longitude, latitude, time and water levels'.format(np.round(time.time()-t0, 2)))
print('Length stations:   {}'.format(len(station_idxs)))
print('Length of times:   {}'.format(len(time_idxs)))
print('Shape waterlevels: {}'.format(waterlevels.shape))

# %%
# Create image collection
t0 = time.time()
col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')

# Filter image collection for 2020 and the North Sea
col = col.filterDate(time_bounds[0].astype(str), time_bounds[1].astype(str))
col = col.filterBounds(ee.Geometry.Rectangle(lon_bounds[0], lat_bounds[0], lon_bounds[1], lat_bounds[1]))

# Get first image
image_idx = 10
image = ee.Image(col.toList(col.size()).get(image_idx))

# Convert latitude, longitude and time to earth engine arrays
ee_lats = ee.List(lats.tolist())
ee_lons = ee.List(lons.tolist())
ee_times = ee.List([np.datetime_as_string(time) for time in times]).map(lambda time: ee.Date(time))
ee_waterlevels = ee.Array(ee.List(waterlevels.tolist()))

print('{} s to create image collection'.format(np.round(time.time()-t0, 2)))
print('Length images:     {}'.format(col.size().getInfo()))
print('Shape lats:        {}'.format(np.array(ee_lats.getInfo()).shape))
print('Shape lons:        {}'.format(np.array(ee_lons.getInfo()).shape))
print('Sahpe times:       {}'.format(np.array(ee_times.getInfo()).shape))
print('Shape waterlevels: {}'.format(np.array(ee_waterlevels.getInfo()).shape))

# %%
# Add water level to image
def add_waterlevel_to_image(image, lats, lons, times, waterlevels):
    # Get latitude and longitude of the image
    image_centroid = ee.Geometry.centroid(image.geometry())
    image_lat = image_centroid.coordinates().get(1)
    image_lon = image_centroid.coordinates().get(0)

    # Get time of the image
    image_time_start = ee.Date(image.get('system:time_start'))
    image_time_end = ee.Date(image.get('system:time_end'))
    image_time = ee.Date((image_time_start.millis().add(image_time_end.millis())).divide(2))

    # Get latitude and longitude of nearest station
    dists1 = ee.Array(ee.Array(lats).subtract(image_lat).pow(2).add(ee.Array(lons).subtract(image_lon).pow(2)).toList())
    station_idx = dists1.multiply(-1).argmax().get(0)
    station_lat = lats.get(station_idx)
    station_lon = lons.get(station_idx)

    # Convert date to milliseconds
    image_time = image_time.millis()
    times = times.map(lambda time: ee.Date(time).millis())

    # Get nearest time at station
    dists2 = ee.Array(ee.Array(times).subtract(image_time)).abs()
    station_time_idx = dists2.multiply(-1).argmax().get(0)
    station_time = times.get(station_time_idx)

    # Get water level
    station_waterlevel = waterlevels.get(ee.List([station_time_idx, station_idx]))

    # Append metadata to image
    image = image.set({
        'image_lat': image_lat,
        'image_lon': image_lon,
        'image_time': image_time,
        'station_idx': station_idx,
        'station_time_idx': station_time_idx,
        'station_lat': station_lat,
        'station_lon': station_lon,
        'station_time': station_time,
        'station_waterlevel': station_waterlevel
    })

    # Return image
    return image

# Add water levels to image
t0 = time.time()
image = add_waterlevel_to_image(image, ee_lats, ee_lons, ee_times, ee_waterlevels)
print('{} s to add water levels to image'.format(np.round(time.time()-t0, 2)))

# Get image properties
image_properties = image.getInfo()['properties']
print('image_lat:          {}'.format(image_properties['image_lat']))
print('image_lon:          {}'.format(image_properties['image_lon']))
print('image_time:         {}'.format(np.datetime64(image_properties['image_time'], 'ms')))
print('station_idx:        {}'.format(image_properties['station_idx']))
print('station_time_idx:   {}'.format(image_properties['station_time_idx']))
print('station_lat:        {}'.format(image_properties['station_lat']))
print('station_lon:        {}'.format(image_properties['station_lon']))
print('station_time:       {}'.format(np.datetime64(image_properties['station_time'], 'ms')))
print('station_waterlevel: {}'.format(image_properties['station_waterlevel']))

# Get image geometry
image_geometry = image.geometry().getInfo()
image_coords = {'x': [coord[0] for coord in image_geometry['coordinates'][0]],
                'y': [coord[1] for coord in image_geometry['coordinates'][0]]}

# %%
# Plot stations and image
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
p = axs[0].tripcolor(lons, lats, waterlevels[image_properties['station_time_idx'], :], cmap='Spectral')
fig.colorbar(p, ax=axs[0], label='Water level at image time [m]')
axs[0].scatter(lons, lats, c='k', s=1, label='Stations')
axs[0].plot(image_coords['x'], image_coords['y'], 'b', label='Image')
axs[0].scatter(image_properties['image_lon'], image_properties['image_lat'], c='b', s=10, label='Image centroid')
axs[0].scatter(image_properties['station_lon'], image_properties['station_lat'], c='r', s=10, label='Nearest station')
axs[0].set_xlim(lon_bounds)
axs[0].set_ylim(lat_bounds)
axs[0].set_aspect('equal')
axs[0].grid()
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
axs[0].legend()

axs[1].plot(times, waterlevels[:, image_properties['station_idx']], 'r')
axs[1].plot(np.datetime64(image_properties['station_time'], 'ms'), image_properties['station_waterlevel'], 'bo', label='Image time')
axs[1].set_xlim([times[0], times[-1]])
axs[1].set_ylim([np.nanmin(waterlevels), np.nanmax(waterlevels)])
axs[1].grid()
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Water level at nearest station [m]')
axs[1].legend()
plt.show()

# %%
# Interactive map
# Create map
map = geemap.Map()

# Center map
map.setCenter(image_properties['image_lon'], image_properties['image_lat'], 10)

# Add image
map.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 2000}, 'image')

# Add stations
stations = ee.FeatureCollection([ee.Geometry.Point([lon, lat]) for lon, lat in zip(lons, lats)])
map.addLayer(stations, {'color': 'black'}, 'stations')

# Add nearest station
nearest_station = ee.FeatureCollection([ee.Geometry.Point([image_properties['station_lon'], image_properties['station_lat']])])
map.addLayer(nearest_station, {'color': 'red'}, 'nearest station')

# Add image centroid
image_centroid = ee.FeatureCollection([ee.Geometry.Point([image_properties['image_lon'], image_properties['image_lat']])])
map.addLayer(image_centroid, {'color': 'blue'}, 'image centroid')

# Display map
map

# %%
# Add water levels to all images
col_size = col.size().getInfo()
t0 = time.time()
col2 = col.map(lambda image: add_waterlevel_to_image(image, ee_lats, ee_lons, ee_times, ee_waterlevels))
print('{} s to add water levels to {} images'.format(np.round(time.time()-t0, 2), col_size))

# %%