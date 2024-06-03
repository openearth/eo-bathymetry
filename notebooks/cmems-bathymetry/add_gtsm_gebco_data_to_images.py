# %%
# Packages
import ee
import geemap
import numpy as np
import shapely as shp
import time
import geopandas as gdf
import matplotlib.pyplot as plt
ee.Initialize()

import sys
local_path = r"c:\Users\white_rn\Documents\GitHub\ee-packages-py" 
sys.path.append(local_path)
from eepackages import tiler

# %%
# Define area of interest, period of interest and tiles
# Area of interest
aoi_file_path = r'p:\11209821-cmems-global-sdb\00_miscellaneous\AOIs\AOI_GER_WaddenSea.geojson'
gdf_aoi = gdf.read_file(aoi_file_path)
aoi = ee.Geometry.Polygon(gdf_aoi.geometry[0].geoms[0].__geo_interface__['coordinates'])   

# Period of interest
poi = ['2021-01-01', '2022-01-01']

# Zoom level
zoom = 10

# Get tiles
tiles = tiler.get_tiles_for_geometry(aoi, zoom)
tiles = tiles.toList(tiles.size())
print('Number of tiles: {}'.format(tiles.size().getInfo()))

# %%
# Select tile
tile_idx = 37
tile = ee.Feature(tiles.get(tile_idx))

# %%
# Plot tiles
plot_tile = False
if plot_tile:
    print('Create map')
    map = geemap.Map()

    print('Center map')
    map.setCenter(6.5, 53.5, 8)

    print('Add selected tile')
    map.addLayer(tile.geometry(), {'color': 'red'}, 'Selected tile')

    print('Display map')
    map

# %%
# Get earth engine image collections
# Get image collection
t0 = time.time()
image_col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
image_col = image_col.filterDate(poi[0], poi[1])
image_col = image_col.filterBounds(tile.geometry())
print('{} s to get image collection'.format(np.round(time.time()-t0, 2)))

# Get gtsm feature collection
t0 = time.time()
gtsm_col = ee.FeatureCollection('projects/bathymetry/assets/gtsm_waterlevels')
print('{} s to get gtsm feature collection'.format(np.round(time.time()-t0, 2)))

# Get gebco image
t0 = time.time()
gebco_image = ee.Image('projects/bathymetry/assets/gebco_2023_hat_lat')
print('{} s to get gebco image'.format(np.round(time.time()-t0, 2)))

# Get size of images and gtsm features
#print('Number of images: {}'.format(image_col.size().getInfo()))

# %%
# Add gtsm and gebco data to images
def add_gtsm_gebco_data_to_images(image, gtsm_col, gebco_image, tile=ee.Feature(None),
                                  max_spatial_offset=1, max_temporal_offset=10):
    ''' Add gtsm and gebco data to images.

    :param image: Image to which gtsm data is added.
    :type image: ee.Image
    :param gtsm_col: gtsm feature collection with water levels.
    :type gtsm_col: ee.FeatureCollection
    :param gebco_image: gebco image with highest and lowest astronomical tide.
    :type gebco_image: ee.Image
    :param tile: Tile to which image belongs (default=None, tile geometry is determined based on image geometry).
    :type tile: ee.Feature
    :param max_spatial_offset: Maximum spatial offset in kilometers.
    :type max_spatial_offset: float (default=1)
    :param max_temporal_offset: Maximum temporal offset in minutes
    :type max_temporal_offset: float (default=10)
    '''
    # If tile geometry is not provided, set tile geometry to image geometry
    tile = ee.Feature(ee.Algorithms.If(ee.Algorithms.IsEqual(tile.geometry(), None),
                                       ee.Feature(image.geometry()),
                                       tile))

    # Get area around the tile
    tile_centroid = ee.Geometry.centroid(tile.geometry(), maxError=1)
    tile_footprint = ee.Geometry(tile.geometry())
    tile_buffer = tile_footprint.buffer(max_spatial_offset*1000)

    # Get period around image time
    image_time_start = ee.Date(image.get('system:time_start'))
    image_time_end = ee.Date(ee.Algorithms.If(image.get('system:time_end'),
                                              ee.Date(image.get('system:time_end')),
                                              image_time_start))
    image_time = ee.Date(image_time_start.millis().add(image_time_end.millis()).divide(2))
    image_period = ee.DateRange(ee.Date(image_time_start.millis().subtract(max_temporal_offset*60*1000)),
                                ee.Date(image_time_end.millis().add(max_temporal_offset*60*1000)))
    
    # Filter gtsm data based on image footprint and period
    gtsm_col = gtsm_col.filterBounds(tile_buffer)
    gtsm_col = gtsm_col.filterDate(image_period.start(), image_period.end())

    # Add spatial offset to features
    def add_spatial_offset_to_features(feature):
        return feature.set('spatial offset to image', feature.distance(ee.Feature(tile_centroid)))
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

    # Get station centroid
    station_centroid = ee.Geometry(ee.Algorithms.If(bool_empty,
                                                    ee.Geometry.Point(0, 0),
                                                    gtsm_col.first().geometry()))
    station_buffer = ee.Geometry(ee.Algorithms.If(bool_empty,
                                                  ee.Geometry.Point(0, 0).buffer(max_spatial_offset*1000),
                                                  gtsm_col.first().geometry())).buffer(max_spatial_offset*1000)
    
    # Get gebco highest and lowest astronomical tide data
    gebco_data = ee.Dictionary(ee.Algorithms.If(bool_empty,
                                                ee.Dictionary({'b1': 0, 'b2': 0}),
                                                gebco_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=station_buffer, scale=30)))

    # Set gtsm en gebco data to image
    image = ee.Image(ee.Algorithms.If(bool_empty,
                                      image.set({'gtsm_gebco_data_isempty': True}),
                                      image.set({'gtsm_gebco_data_isempty': False,
                                                 'gtsm_station': gtsm_col.first().get('station'),
                                                 'gtsm_station_lon': gtsm_col.first().geometry().coordinates().get(0),
                                                 'gtsm_station_lat': gtsm_col.first().geometry().coordinates().get(1),
                                                 'gtsm_time': gtsm_col.first().get('times'),
                                                 'gtsm_waterlevel': gtsm_col.first().get('waterlevel'),
                                                 'gebco_hat': gebco_data.get('b1'),
                                                 'gebco_lat': gebco_data.get('b2')})))

    # Return image
    return image

# Add gtsm data to first image
image1 = image_col.first()
t0 = time.time()
image2 = add_gtsm_gebco_data_to_images(image1, gtsm_col, gebco_image, tile)
print('{} s to add gtsm and gebco data to the first image'.format(np.round(time.time()-t0, 2)))
#print(image1.getInfo()['properties'])
#print(image2.getInfo()['properties'])

# %%
# Add gtsm and gebco data to all images
t0 = time.time()
image_col2 = image_col.map(lambda image: add_gtsm_gebco_data_to_images(image, gtsm_col, gebco_image, tile))
print('{} s to add gtsm and gebco data to alll images'.format(np.round(time.time()-t0, 2)))

# %%
# Plot image
plot_image = False
image_idx = 0
if plot_image:
    # Convert image collection to image list
    print('Convert image collection to image list')
    image_list = image_col.toList(image_col.size().getInfo())
    image_list2 = image_col2.toList(image_col2.size().getInfo())

    # Get image
    print('Get image')
    image = ee.Image(image_list.get(image_idx))
    image2 = ee.Image(image_list2.get(image_idx))

    # Get image centroid
    print('Get image centroid')
    image_centroid = ee.Geometry.centroid(image2.geometry())
    image_centroid_coords = image_centroid.coordinates().getInfo()

    # Get image properties
    print('Get image properties')
    image_properties = image2.getInfo()['properties']

    # Get tile centroid
    print('Get tile centroid')
    tile_centroid = ee.Geometry.centroid(tile.geometry(), maxError=1)

    # Create map
    print('Create map')
    map = geemap.Map()

    # Center map
    print('Center map')
    map.setCenter(image_centroid_coords[0], image_centroid_coords[1], 11)

    # Add image
    print('Add image')
    map.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 2000}, 'image')

    # Add image centroid
    print('Add image centroid')
    map.addLayer(image_centroid, {'color': 'black'}, 'Image centroid')

    # Add tile
    print('Add tile')
    map.addLayer(tile.geometry(), {'color': 'blue'}, 'Tile')

    # Add tile centroid
    print('Add tile centroid')
    map.addLayer(tile_centroid, {'color': 'green'}, 'Tile centroid')

    # Add nearest gtsm_feature
    print('Add nearest gtsm feature')
    if not image_properties['gtsm_gebco_data_isempty']:
        gtsm_feature = ee.Geometry.Point([image_properties['gtsm_station_lon'], image_properties['gtsm_station_lat']])
        map.addLayer(gtsm_feature, {'color': 'red'}, 'Nearest gtsm feature')

    # Display map
    print('Display map')
    map

# %%
# Get gtsm and gebco data from image
def get_gtsm_gebco_data_from_image(image):
    ''' Get gtsm and gebco data from image.

    :param image: Image from which gtsm data is extracted.
    :type image: ee.Image
    :return: Dictionary with gtsm and gebco data.
    :rtype: dict
    '''

    # Initialise feature
    feature = ee.Feature(None)

    # Add image properties
    feature = ee.Feature(ee.Algorithms.If(image.get('gtsm_gebco_data_isempty'),
                                          feature.set({'gtsm_station': None,
                                                       'gtsm_station_lon': None,
                                                       'gtsm_station_lat': None,
                                                       'gtsm_time': None,
                                                       'gtsm_waterlevel': None,
                                                       'gebco_hat': None,
                                                       'gebco_lat': None}),
                                          feature.set({'gtsm_station': image.get('gtsm_station'),
                                                       'gtsm_station_lon': image.get('gtsm_station_lon'),
                                                       'gtsm_station_lat': image.get('gtsm_station_lat'),
                                                       'gtsm_time': image.get('gtsm_time'),
                                                       'gtsm_waterlevel': image.get('gtsm_waterlevel'),
                                                       'gebco_hat': image.get('gebco_hat'),
                                                       'gebco_lat': image.get('gebco_lat')})))
    
    # Return feature
    return feature

# Get gtsm and gebco data from image
print('Get data from image')
gtsm_gebco_data = image_col2.map(lambda image: get_gtsm_gebco_data_from_image(image)).getInfo()
image_idxs = list(range(len(gtsm_gebco_data['features'])))
gtsm_waterlevels = [feature['properties']['gtsm_waterlevel'] for feature in gtsm_gebco_data['features']]
gebco_hats = [feature['properties']['gebco_hat'] for feature in gtsm_gebco_data['features']]
gebco_lats = [feature['properties']['gebco_lat'] for feature in gtsm_gebco_data['features']]

# %%
# Plot gtsm waterlevels
fig, ax = plt.subplots()
ax.scatter(image_idxs, gtsm_waterlevels, color='blue', s=3, label='Gtsm water level')
ax.plot(image_idxs, gebco_hats, color='red', linewidth=3, label='Gebco highest astronomical tide')
ax.plot(image_idxs, gebco_lats, color='green', linewidth=3, label='Gebco lowest astronomical tide')
ax.set_xlabel('Image index')
ax.set_ylabel('Water level [m]')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35))
plt.show()

# %%
