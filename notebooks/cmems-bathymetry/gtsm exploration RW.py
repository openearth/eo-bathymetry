# %%
# Packages
import contextily as ctx
import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import shape
import time
import xarray as xr
ee.Initialize()

# %%
# Plot map extents
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cmap = plt.cm.get_cmap('Spectral')
colors = cmap(np.linspace(0, 1, 15))

for i in range(0, 15):
    file_path_map_nc = r'p:\1230882-emodnet_hrsm\GTSMv3.0EMODnet\CMEMS_intertidal_SDB\r001\output\gtsm_model_{:04}_map.nc'.format(i)
    print(os.path.basename(file_path_map_nc))
    map = xr.open_dataset(file_path_map_nc)
    lons = map.mesh2d_face_x.values
    lats = map.mesh2d_face_y.values
    ax.scatter(lons, lats, color=colors[i], alpha=0.5, label='Map {:04}'.format(i))
ax.legend(loc='right', bbox_to_anchor=(1.2, 0.5))
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])
ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom=1)

# %%
# File paths
file_path_his_nc = r'p:\1230882-emodnet_hrsm\GTSMv3.0EMODnet\CMEMS_intertidal_SDB\r001\output\gtsm_model_0000_his.nc'
file_path_map_nc = r'p:\1230882-emodnet_hrsm\GTSMv3.0EMODnet\CMEMS_intertidal_SDB\r001\output\gtsm_model_0014_map.nc'

# Bounds North Sea
lon_bounds = [1.5, 4.5]
lat_bounds = [51, 54]
time_bounds = ['2021-06-01', '2021-07-01']
time_bounds = [np.datetime64(time_bounds[0]), np.datetime64(time_bounds[1])]

# %%
# Open files
t0 = time.time()
his = xr.open_dataset(file_path_his_nc)
print('{} s to open his file'.format(np.round(time.time()-t0, 2)))

t0 = time.time()
map = xr.open_dataset(file_path_map_nc)
print('{} s to open map file'.format(np.round(time.time()-t0, 2)))

# %%
# Get variable names
his_vars = list(his.variables)
map_vars = list(map.variables)
'''
print('his variables: {}'.format(his_vars))
print('map variables: {}'.format(map_vars))
'''
# Get coordinates
his_coords = list(his.coords)
map_coords = list(map.coords)
'''
print('his coordinates: {}'.format(his_coords))
print('map coordinates: {}'.format(map_coords))
'''

# %%
# Get data from file
# Get longitude, latitude, time and water levels
t0 = time.time()
lons = his.station_x_coordinate.values
lats = his.station_y_coordinate.values
times = his.time.values
#waterlevels = his.waterlevel.values

# Get station and time indices
station_idxs = np.where((lats >= lat_bounds[0]) & (lats <= lat_bounds[1]) & (lons >= lon_bounds[0]) & (lons <= lon_bounds[1]))[0]
time_idxs = np.where((times >= time_bounds[0]) & (times <= time_bounds[1]))[0]

# Get longitude [deg], latitude [deg], time [ms] and water levels [m]
lats = lats[station_idxs]
lons = lons[station_idxs]
times = times[time_idxs]
waterlevels = his.waterlevel.isel(stations=station_idxs, time=time_idxs).values

# Remove poitns where water level is nan
no_nan_idxs = np.where(~np.isnan(waterlevels).all(axis=0))[0]
station_idxs = station_idxs[no_nan_idxs]
lats = lats[no_nan_idxs]
lons = lons[no_nan_idxs]
waterlevels = waterlevels[:, no_nan_idxs]

print('{} s to get data from file'.format(np.round(time.time()-t0, 2)))
'''
print('Length stations:   {}'.format(len(station_idxs)))
print('Length of times:   {}'.format(len(time_idxs)))
print('Shape waterlevels: {}'.format(waterlevels.shape))
'''

# %%
# Get earth engine image collection and arrays
# Create image collection
t0 = time.time()
col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')

# Filter image collection for 2020 and the North Sea
col = col.filterDate(time_bounds[0].astype(str), time_bounds[1].astype(str))
col = col.filterBounds(ee.Geometry.Rectangle(lon_bounds[0], lat_bounds[0], lon_bounds[1], lat_bounds[1]))

# Convert latitudes, longitudes, times and water levels to earth engine objects
ee_lats = ee.List(lats.tolist())
ee_lons = ee.List(lons.tolist())
ee_times = ee.List([np.datetime_as_string(time) for time in times]).map(lambda time: ee.Date(time))
ee_waterlevels = ee.Array(ee.List(waterlevels.tolist()))

print('{} s to create image collection'.format(np.round(time.time()-t0, 2)))
'''
print('Length images:     {}'.format(col.size().getInfo()))
print('Shape lats:        {}'.format(np.array(ee_lats.getInfo()).shape))
print('Shape lons:        {}'.format(np.array(ee_lons.getInfo()).shape))
print('Shape times:       {}'.format(np.array(ee_times.getInfo()).shape))
print('Shape waterlevels: {}'.format(np.array(ee_waterlevels.getInfo()).shape))
'''

# %%
# Add water level to images
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

# Add water levels to images
col_size = col.size().getInfo()
t0 = time.time()
col2 = col.map(lambda image: add_waterlevel_to_image(image, ee_lats, ee_lons, ee_times, ee_waterlevels))
print('{} s to add water levels to {} images'.format(np.round(time.time()-t0, 2), col_size))

# %%
# Get features from images
def get_feature_from_image(image):
    feature = ee.Feature(ee.Geometry(image.geometry()),
                         {'image_lat': image.get('image_lat'),
                          'image_lon': image.get('image_lon'),
                          'image_time': image.get('image_time'),
                          'station_idx': image.get('station_idx'),
                          'station_time_idx': image.get('station_time_idx'),
                          'station_lat': image.get('station_lat'),
                          'station_lon': image.get('station_lon'),
                          'station_time': image.get('station_time'),
                          'station_waterlevel': image.get('station_waterlevel')})	
    return feature

def get_features_from_image_collection(col):
    # Get features from image collection
    features = col.map(get_feature_from_image).getInfo()['features']

    # Convert json geometries of the features to shapely geometries
    for i, feature in enumerate(features):
        features[i]['geometry'] = shape(feature['geometry'])
    
    # Convert ms times to numpy datetime64
    for feature in features:
        feature['properties']['image_time'] = np.datetime64(feature['properties']['image_time'], 'ms')
        feature['properties']['station_time'] = np.datetime64(feature['properties']['station_time'], 'ms')
    
    return features

t0 = time.time()
features = get_features_from_image_collection(col2)
t1 = time.time()
print('{} s to get {} features'.format(np.round(t1-t0, 2), len(features)))

# %%
# Plot on interactive map
def plot_interactive_map(image_idx):
    # Get feature geometry and properties
    geometry = features[image_idx]['geometry']
    properties = features[image_idx]['properties']

    # Get image
    image = ee.Image(col2.toList(col2.size()).get(image_idx))

    # Interactive map
    # Create map
    map = geemap.Map()

    # Center map
    map.setCenter(properties['image_lon'], properties['image_lat'], 10)

    # Add image
    map.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 2000}, 'image')

    # Add stations
    stations = ee.FeatureCollection([ee.Geometry.Point([lon, lat]) for lon, lat in zip(lons, lats)])
    map.addLayer(stations, {'color': 'black'}, 'stations')

    # Add image centroid
    image_centroid = ee.FeatureCollection([ee.Geometry.Point([properties['image_lon'], properties['image_lat']])])
    map.addLayer(image_centroid, {'color': 'red', 'markerType': 'cross'}, 'image centroid')

    # Add nearest station
    nearest_station = ee.FeatureCollection([ee.Geometry.Point([properties['station_lon'], properties['station_lat']])])
    map.addLayer(nearest_station, {'color': 'blue', 'markerType': 'cross'}, 'nearest station')

    # Display map
    map

# Plot on interactive map
plot_interactive_map(image_idx=1)

# %%
# Plot on static map with time series
def plot_static_map(image_idx):
    # Get feature geometry and properties
    geometry = features[image_idx]['geometry']
    properties = features[image_idx]['properties']

    # Create figure and axis
    gridspec_kwargs = {'height_ratios': [1, 0.5]}
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw=gridspec_kwargs)

    # Plot map
    # Plot stations
    axs[0].tripcolor(lons, lats, waterlevels[properties['station_time_idx'], :],
                    cmap='Spectral', edgecolors='none', alpha=0.25, vmin=-2.5, vmax=2.5)
    p = axs[0].scatter(lons, lats, c=waterlevels[properties['station_time_idx'], :],
                    cmap='Spectral', s=20, vmin=-2.5, vmax=2.5, label='Stations')
    plt.colorbar(p, ax=axs[0], label='Water level at image time [m]')

    # Plot features
    axs[0].add_patch(plt.Polygon(geometry.exterior.coords, color='grey',
                                alpha=0.25, linewidth=1.5, label='Image extent'))
                    
    axs[0].plot(properties['image_lon'], properties['image_lat'],
                linestyle='none', marker='o', color='grey', label='Image centroid')
    axs[0].plot(properties['station_lon'], properties['station_lat'],
                linestyle='none', marker='o', markerfacecolor='none', color='black', label='Nearest station')

    # Set axis properties
    axs[0].set_xlim(lon_bounds)
    axs[0].set_ylim(lat_bounds)
    axs[0].set_aspect('equal')
    axs[0].grid()
    axs[0].set_xlabel('Longitude [deg]')
    axs[0].set_ylabel('Latitude [deg]')
    axs[0].legend(loc='upper right', framealpha=0.5)

    # Add basemap
    ctx.add_basemap(axs[0], crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom=8)

    # Plot time series
    axs[1].plot(times, waterlevels[:, properties['station_idx']],
                linestyle='-', color='r', label='Water level at nearest station')
    axs[1].plot(properties['image_time'], properties['station_waterlevel'],
                linestyle='none', marker='o', color='grey', label='Image time')
    axs[1].plot(properties['station_time'], properties['station_waterlevel'],
                linestyle='none', marker='o', markerfacecolor='none', color='black', label='Station time')

    # Set axis properties
    #axs[1].set_xlim([times[0], times[-1]])
    axs[1].set_xlim(properties['image_time'].astype('datetime64[D]') - np.timedelta64(1, 'D'), properties['image_time'].astype('datetime64[D]') + np.timedelta64(1, 'D'))
    axs[1].set_ylim(-2.5, 2.5)
    axs[1].grid()
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Water level at nearest station [m]')
    axs[1].legend(loc='upper right', framealpha=0.5)

# Plot on static map with time series
for image_idx in range(0, len(features), 1):
    plot_static_map(image_idx)
    plt.show()

# %%
# Test map dataset
t0 = time.time()
lons2 = map.mesh2d_face_x.values
lats2 = map.mesh2d_face_y.values
times2 = map.time.values
waterlevels2 = map.waterdepth.values

print('{} s to get data from file'.format(np.round(time.time()-t0, 2)))
print('Length stations:   {}'.format(len(lons2)))
print('Shape waterlevels: {}'.format(waterlevels2.shape))

# %%