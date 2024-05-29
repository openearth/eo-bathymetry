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
# Spatial and temporal bounds
lon_bounds = [0, 6]
lat_bounds = [50, 54]
time_bounds = ['2021-01-01', '2021-02-01']

# %%
# Get earth engine image collections
# Get image collection
t0 = time.time()
image_col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
image_col = image_col.filterDate(time_bounds[0], time_bounds[1])
image_col = image_col.filterBounds(ee.Geometry.Rectangle(lon_bounds[0], lat_bounds[0], lon_bounds[1], lat_bounds[1]))
print('{} s to create image collection'.format(np.round(time.time()-t0, 2)))

# Get gtsm feature collection
t0 = time.time()
gtsm_col = ee.FeatureCollection('projects/bathymetry/assets/gtsm_waterlevels')
print('{} s to get gtsm feature collection'.format(np.round(time.time()-t0, 2)))

# Get size of images and gtsm features
print('Number of images:       {}'.format(image_col.size().getInfo()))

# %%
# Add gtsm data to images
def add_gtsm_data_to_images(image, gtsm_col, max_spatial_offset=1, max_temporal_offset=10):
    ''' Add gtsm data to images.

    :param image: Image to which gtsm data is added.
    :type image: ee.Image
    :param gtsm_col: gtsm feature collection.
    :type gtsm_col: ee.FeatureCollection
    :param max_spatial_offset: Maximum spatial offset in kilometers.
    :type max_spatial_offset: float (default=1)
    :param max_temporal_offset: Maximum temporal offset in minutes
    :type max_temporal_offset: float (default=10)
    '''

    # Get area around image centroid
    image_centroid = ee.Geometry.centroid(image.geometry())
    image_footprint = ee.Geometry(image.geometry())
    image_footprint = image_footprint.buffer(max_spatial_offset*1000)

    # Get period around image time
    image_time_start = ee.Date(image.get('system:time_start'))
    image_time_end = ee.Date(image.get('system:time_end'))
    image_time = ee.Date((image_time_start.millis().add(image_time_end.millis())).divide(2))
    image_period = ee.DateRange(ee.Date(image_time.millis().subtract(max_temporal_offset*60*1000)),
                                ee.Date(image_time.millis().add(max_temporal_offset*60*1000)))
    
    # Filter gtsm data based on image footprint and period
    gtsm_col = gtsm_col.filterBounds(image_footprint)
    gtsm_col = gtsm_col.filterDate(image_period.start(), image_period.end())

    # Add spatial offset to features
    def add_spatial_offset_to_features(feature):
        return feature.set('spatial offset to image', feature.distance(ee.Feature(image_centroid)))
    gtsm_col = gtsm_col.map(add_spatial_offset_to_features)

    # Get minimum spatial offset
    min_spatial_offset = gtsm_col.reduceColumns(ee.Reducer.min(), ['spatial offset to image']).get('min')
    
    # Get features for which the spatial offset is equal to the minimum spatial offset (multiple features possible)
    gtsm_col = gtsm_col.filter(ee.Filter.eq('spatial offset to image', min_spatial_offset))

    # Add temporal offset to features
    def add_temporal_offset_to_features(feature):
        return feature.set('temporal offset to image', ee.Number(feature.get('system:time_start')).subtract(image_time.millis()).abs())
    gtsm_col = gtsm_col.map(add_temporal_offset_to_features)

    # Get minimum temporal offset
    min_temporal_offset = gtsm_col.reduceColumns(ee.Reducer.min(), ['temporal offset to image']).get('min')

    # Get features for which the temporal offset is equal to the minimum temporal offset (multiple features possible)
    gtsm_col = gtsm_col.filter(ee.Filter.eq('temporal offset to image', min_temporal_offset))

    # Check if gtsm feature collection is empty
    bool_empty = gtsm_col.size().eq(0)

    # Set gtsm data to image if gtsm feature collection is not empty
    image = ee.Algorithms.If(bool_empty,
                            image.set('gtsm_data', ee.Dictionary({
                                'is_empty': True})),
                            image.set('gtsm_data', ee.Dictionary({
                                'is_empty': False,
                                'station': gtsm_col.first().get('station'),
                                'station_lon': gtsm_col.first().geometry().coordinates().get(0),
                                'station_lat': gtsm_col.first().geometry().coordinates().get(1),
                                'time': gtsm_col.first().get('times'),
                                'waterlevel': gtsm_col.first().get('waterlevel')})))
    
    # Return image
    return image

# Add gtsm data to first image
t0 = time.time()
image1 = image_col.first()
image2 = add_gtsm_data_to_images(image1, gtsm_col)
print('{} s to add gtsm data to first image'.format(np.round(time.time()-t0, 2)))
print(image2.getInfo()['properties']['gtsm_data'])

# %%
# Add gtsm data to images
t0 = time.time()
image_col2 = image_col.map(lambda image: add_gtsm_data_to_images(image, gtsm_col))
print('{} s to add gtsm data to images'.format(np.round(time.time()-t0, 2)))

# %%
# Image index
image_idx = 0

# Convert image collection to image list
image_list = image_col.toList(image_col.size().getInfo())
image_list2 = image_col2.toList(image_col2.size().getInfo())

# Get image
print('Get image')
image = ee.Image(image_list.get(image_idx))
image2 = ee.Image(image_list2.get(image_idx))

# Get image centroid
print('Get image centroid')
image_footprint = ee.Geometry(image.geometry())
image_centroid = ee.Geometry.centroid(image_footprint)
image_centroid_coords = image_centroid.coordinates().getInfo()

# Get gtsm data
print('Get gtsm data')
gtsm_data = image2.getInfo()['properties']['gtsm_data']

# Create map
print('Create map')
map = geemap.Map()

# Center map
print('Center map')
map.setCenter(image_centroid_coords[0], image_centroid_coords[1], 11)

# Add image footprint
print('Add image footprint')
map.addLayer(image_footprint, {'color': 'black', 'width': 1}, 'Image footprint')

# Add image
print('Add image')
map.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 2000}, 'image')

# Add image centroid
print('Add image centroid')
map.addLayer(image_centroid, {'color': 'black'}, 'Image centroid')

# Add nearest gtsm_feature
print('Add nearest gtsm feature')
if not gtsm_data['is_empty']:
    gtsm_feature = ee.Geometry.Point(gtsm_data['station_lon'], gtsm_data['station_lat'])
    map.addLayer(gtsm_feature, {'color': 'red'}, 'Nearest gtsm feature')

# Display map
print('Display map')
map

# %%
