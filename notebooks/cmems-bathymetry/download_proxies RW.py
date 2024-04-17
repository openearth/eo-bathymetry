# %%
# Packages
# generic imports
import sys
import os
import geopandas as gpd
import time
import geemap
import geojson
import ee
ee.Initialize()

# specific imports
from typing import Any, Dict, List, Optional
from geojson import Polygon, Feature, FeatureCollection, dump
from shapely.geometry import Polygon
from dateutil.relativedelta import *
from google.cloud import storage
from logging import Logger, getLogger
from googleapiclient.discovery import build
from re import sub
from ctypes import ArgumentError
from functools import partial
from dateutil.parser import parse

# custom functionality import without requirement to pip install package
local_path = r'c:\Users\white_rn\Documents\GitHub\ee-packages-py'  # path to local GitHub clone
sys.path.append(local_path)
from eepackages.applications.bathymetry import Bathymetry
from eepackages import tiler

logger: Logger = getLogger(__name__)

# Acknowledgements & Code references:
# - https://github.com/openearth/eo-bathymetry/
# - https://github.com/openearth/eo-bathymetry-functions/
# - https://github.com/gee-community/ee-packages-py

# See scheme at https://github.com/openearth/eo-bathymetry/blob/master/notebooks/rws-bathymetry/acces_api.pdf for a workflow visualization 

# %%
# Functions
def get_tile_subtidal_bathymetry(tile: ee.Feature, start: ee.String, stop: ee.String) -> ee.Image:
    """
    Get subtidal bathymetry based on tile geometry.
    Server-side compliant for GEE.

    args:
        tile (ee.Feature): tile geometry used to obtain bathymetry.
        start (ee.String): start date in YYYY-MM-dd format.
        stop (ee.String): stop date in YYYY-MM-dd format.
    
    returns:
        ee.Image: image containing subtidal bathymetry covering tile.
    """

    bounds: ee.Geometry = ee.Feature(tile).geometry().bounds(1)
    sdb: Bathymetry = Bathymetry()
    zoom: ee.String = ee.String(tile.get("zoom"))
    tx: ee.String = ee.String(tile.get("tx"))
    ty: ee.String = ee.String(tile.get("ty"))
    tile_name: ee.String = ee.String("z").cat(zoom).cat("_x").cat(tx).cat("_y").cat(ty).replace("\.\d+", "", "g")
    img_fullname: ee.String = ee.String(tile_name).cat("_t").cat(ee.Date(start).millis().format())
        
    image: ee.Image = sdb.compute_inverse_depth(
                bounds=bounds,
                start=start,
                stop=stop,
                scale=tiler.zoom_to_scale(ee.Number.parse(tile.get("zoom"))).multiply(5), # scale to search for clean images
                missions=["S2", "L8"],
                filter_masked=True,
                skip_neighborhood_search=False
                # cloud_frequency_threshold_data=,
                # pansharpen=,
                # skip_scene_boundary_fix=,
                # bounds_buffer=
    ).clip(bounds)

    image = image.set(
        "fullname", img_fullname,
        "system:time_start", ee.Date(start).millis(),
        "system:time_stop", ee.Date(stop).millis(),
        "zoom", zoom,
        "tx", tx,
        "ty", ty
    )
    return image

def get_tile_intertidal_bathymetry(tile: ee.Feature, start: ee.String, stop: ee.String) -> ee.Image:
    """
    Get intertidal bathymetry based on tile geometry.
    Server-side compliant for GEE.

    args:
        tile (ee.Feature): tile geometry used to obtain bathymetry.
        start (ee.String): start date in YYYY-MM-dd format.
        stop (ee.String): stop date in YYYY-MM-dd format.
    
    returns:
        ee.Image: image containing intertidal bathymetry covering tile.
    """

    bounds: ee.Geometry = ee.Feature(tile).geometry().bounds(1)
    sdb: Bathymetry = Bathymetry()
    zoom: ee.String = ee.String(tile.get("zoom"))
    tx: ee.String = ee.String(tile.get("tx"))
    ty: ee.String = ee.String(tile.get("ty"))
    tile_name: ee.String = ee.String("z").cat(zoom).cat("_x").cat(tx).cat("_y").cat(ty).replace("\.\d+", "", "g")
    img_fullname: ee.String = ee.String(tile_name).cat("_t").cat(ee.Date(start).millis().format())
        
    image: ee.Image = sdb.compute_intertidal_depth(
        bounds=bounds,
        start=start,
        stop=stop,
        scale=tiler.zoom_to_scale(ee.Number.parse(tile.get("zoom"))).multiply(5), # scale to search for clean images
        # missions=['S2', 'L8'],
        # filter: ee.Filter.dayOfYear(7*30, 9*30), # summer-only
        filter_masked=False, 
        # filterMaskedFraction = 0.5,
        # skip_scene_boundary_fix=False,
        # skip_neighborhood_search=False,
        neighborhood_search_parameters={"erosion": 0, "dilation": 0, "weight": 50},
        bounds_buffer=0,
        water_index_min=-0.05,
        water_index_max=0.15,
        # lowerCdfBoundary=45,
        # upperCdfBoundary=50,
        # cloud_frequency_threshold_data=0.15, 
        clip = True
    )# .reproject(ee.Projection("EPSG:3857").atScale(90))

    image = image.set(
        "fullname", img_fullname,
        "system:time_start", ee.Date(start).millis(),
        "system:time_stop", ee.Date(stop).millis(),
        "zoom", zoom,
        "tx", tx,
        "ty", ty
    )

    return image

def tile_to_asset(
    image: ee.Image,
    tile: ee.Feature,
    export_scale: int,
    asset_path_prefix: str,
    asset_name: str,
    overwrite: bool
) -> Optional[ee.batch.Task]:
    
    asset_id: str = f"{asset_path_prefix}/{asset_name}"
    asset: Dict[str, Any] = ee.data.getInfo(asset_id)
    if overwrite and asset:
        logger.info(f"deleting asset {asset}")
        ee.data.deleteAsset(asset_id)
    elif asset:
        logger.info(f"asset {asset} already exists, skipping {asset_name}")
        return
    task: ee.batch.Task = ee.batch.Export.image.toAsset(
        image,
        assetId=asset_id,
        description=asset_name,
        region=tile.geometry(),
        scale=export_scale,
        maxPixels= 1e10
    )
    task.start()
    logger.info(f"exporting {asset_name} to {asset_id}")

def tile_to_cloud_storage(
    image: ee.Image,
    tile: ee.Feature,
    export_scale: int,
    bucket: str,
    bucket_path: str,
    overwrite: bool
) -> Optional[ee.batch.Task]:
    with build('storage', 'v1') as storage:
        res = storage.objects().list(bucket=bucket, prefix="/".join(bucket_path.split("/")[:-1])).execute()
    if not overwrite:
        try:
            object_exists = any(map(lambda item: item.get("name").startswith(bucket_path), res.get("items")))
        except AttributeError:
            object_exists = False
        if object_exists:
            logger.info(f"object {bucket_path} already exists in bucket {bucket}, skipping")
            return
        
    task: ee.batch.Task = ee.batch.Export.image.toCloudStorage(
        image,
        bucket=bucket,
        description=bucket_path.replace("/", "_"),
        fileNamePrefix=bucket_path,
        region=tile.geometry(),
        scale=export_scale,
        fileFormat='GeoTIFF',
        formatOptions= {'cloudOptimized': True}, # enables easy QGIS plotting
        maxPixels= 1e10
    )
    task.start()
    return task

def export_sdb_tiles(
    sink: str,
    tile_list: ee.List,
    num_tiles: int,
    export_scale: int,
    sdb_tiles: ee.ImageCollection,
    name_suffix: str,
    mode: str,
    task_list: List[ee.batch.Task],
    overwrite: bool,
    bucket: Optional[str] = None
) -> List[ee.batch.Task]:
    """
    Export list of tiled images containing sub or intertidal tidal bathymetry. Fires off the tasks and adds to the list of tasks.
    based on: https://github.com/gee-community/gee_tools/blob/master/geetools/batch/imagecollection.py#L166

    args:
        sink (str): type of data sink to export to. Viable options are: "asset" and "cloud".
        tile_list (ee.List): list of tile features.
        num_tiles (int): number of tiles in `tile_list`.
        scale (int): scale of the export product.
        sdb_tiles (ee.ImageCollection): collection of subtidal bathymetry images corresponding
            to input tiles.
        name_suffix (str): unique identifier after tile statistics.
        task_list (List[ee.batch.Task]): list of tasks, adds tasks created to this list.
        overwrite (bool): whether to overwrite the current assets under the same `asset_path`.
        bucket (str): Bucket where the data is stored. Only used when sink = "cloud"
    
    returns:
        List[ee.batch.Task]: list of started tasks

    """
    if sink == "asset":
        user_name: str = ee.data.getAssetRoots()[0]["id"].split("/")[-1]
        asset_path_prefix: str = f"users/{user_name}/eo-bathymetry"
        ee.data.create_assets(asset_ids=[asset_path_prefix], asset_type="Folder", mk_parents=True)
    
    for i in range(num_tiles):
        # get tile
        temp_tile: ee.Feature = ee.Feature(tile_list.get(i))
        tile_metadata: Dict[str, Any] = temp_tile.getInfo()["properties"]
        tx: str = tile_metadata["tx"]
        ty: str = tile_metadata["ty"]
        zoom: str = tile_metadata["zoom"]
        # filter imagecollection based on tile
        filtered_ic: ee.ImageCollection = sdb_tiles \
            .filterMetadata("tx", "equals", tx) \
            .filterMetadata("ty", "equals", ty) \
            .filterMetadata("zoom", "equals", zoom)
        # if filtered correctly, only a single image remains
        img: ee.Image = ee.Image(filtered_ic.first())  # have to cast here
        img_name: str = sub(r"\.\d+", "", f"{mode}/z{zoom}/x{tx}/y{ty}/") + name_suffix 
        # Export image
        if sink == "asset":  # Replace with case / switch in python 3.10
            task: Optional[ee.batch.Task] = tile_to_asset(
                image=img,
                tile=temp_tile,
                export_scale=export_scale,
                asset_path_prefix=asset_path_prefix,
                asset_name=img_name.replace("/","_"),
                overwrite=overwrite
            )
            if task: task_list.append(task)
        elif sink == "cloud":
            if not bucket:
                raise ArgumentError("Sink option requires \"bucket\" arg.")
            task: ee.batch.Task = tile_to_cloud_storage(
                image=img,
                tile=temp_tile,
                export_scale=export_scale,
                bucket=bucket,
                bucket_path=img_name,
                overwrite=overwrite
            )
        else:
            raise ArgumentError("unrecognized data sink: {sink}")
        task_list.append(task)
    return task_list

def export_tiles(
    sink: str,
    mode: str,
    geometry: ee.Geometry,
    zoom: int,
    start: str,
    stop: str,
    scale: Optional[float] = None,
    step_months: int = 3,
    window_months: int = 24,
    overwrite: bool = False,
    bucket: Optional[str] = None
) -> None:
    """
    From a geometry, creates tiles of input zoom level, calculates subtidal bathymetry in those
    tiles, and exports those tiles.

    args:
        sink (str): type of data sink to export to. Viable options are: "asset" and "cloud".
        mode (str): either "subtidal" or "intertidal" for select type of bathymetry to export.
        geometry (ee.Geometry): geometry of the area of interest.
        zoom (int): zoom level of the to-be-exported tiles.
        start (ee.String): start date in YYYY-MM-dd format.
        stop (ee.String): stop date in YYYY-MM-dd format.
        scale Optional(float): scale of the product to be exported. Defaults tiler.zoom_to_scale(zoom).getInfo().
        step_months (int): steps with which to roll the window over which the subtidal bathymetry
            is calculated.
        windows_months (int): number of months over which the bathymetry is calculated.
    """

    def create_year_window(year: ee.Number, month: ee.Number) -> ee.Dictionary:
        t: ee.Date = ee.Date.fromYMD(year, month, 1)
        d_format: str = "YYYY-MM-dd"
        return ee.Dictionary({
            "start": t.format(d_format),
            "stop": t.advance(window_months, 'month').format(d_format)
            })
    
    window_length: int = (parse(stop).year-parse(start).year)*12+(parse(stop).month-parse(start).month)
    dates: ee.List = ee.List.sequence(parse(start).year, parse(stop).year-window_months/12).map(
        lambda year: ee.List.sequence(1, None, step_months, int((window_length-window_months)/step_months)+1).map(partial(create_year_window, year))
    ).flatten()
    
    # Get tiles
    tiles: ee.FeatureCollection = tiler.get_tiles_for_geometry(geometry, ee.Number(zoom))

    if scale == None:
        scale: float = tiler.zoom_to_scale(zoom).getInfo() # not specified, defaults to pre-set float
    task_list: List[ee.batch.Task] = []
    num_tiles: int = tiles.size().getInfo()
    tile_list: ee.List = tiles.toList(num_tiles)

    for date in dates.getInfo():
        if mode == "subtidal":
            sdb_tiles: ee.ImageCollection = tiles.map(
                lambda tile: get_tile_subtidal_bathymetry(
                    tile=tile,
                    start=ee.String(date["start"]),
                    stop=ee.String(date["stop"])
                ).clip(geometry) # clip individual tiles to match geometry of aoi
            )
        elif mode == "intertidal":
            sdb_tiles: ee.ImageCollection = tiles.map(
                lambda tile: get_tile_intertidal_bathymetry(
                    tile=tile,
                    start=ee.String(date["start"]),
                    stop=ee.String(date["stop"])
                ).clip(geometry).select('ndwi').rename('water_score') # clip individual tiles to match geometry of aoi, select ndwi and rename
            )

        # Now export tiles
        export_sdb_tiles(
            sink=sink,
            tile_list=tile_list,
            num_tiles=num_tiles,
            mode=mode,
            export_scale=scale,
            sdb_tiles=sdb_tiles,
            name_suffix=f"t{date['start']}_{date['stop']}",
            task_list=task_list,
            overwrite=overwrite,
            bucket=bucket
        )

# %%
# Main
# Settings
print('Settings')
# Folder and file paths
main_fol = r"p:\11209821-cmems-global-sdb" # name of the main local folder 
aoi_fol = os.path.join(main_fol, "00_miscellaneous", "AOIs") # name of the folder containing the AoI
aoi_polygon_fol = os.path.join(main_fol, "00_miscellaneous", "AOI_polygons") # name of the folder containing the AoI polygons
output_fol = os.path.join(main_fol, "01_intertidal", "02_data") # name of the folder containing the output

bucket = "cmems-sdb" # name of the Google Cloud Storage bucket to store files in the cloud

# Area of Interest (AoI) settings
project_name = "AOI_GER_WaddenSea" # name of the project AoI

# Composite image settings
mode = "intertidal" # specify mode, either "intertidal" or "subtidal"
start_date = "2021-01-01" # start date of the composites
stop_date = "2022-01-01" # end date of the composites
compo_int = 12 # composite interval [months]
compo_len = 12 # composite length [months]
scale = 10  # output resolution of the image [m]

# Tiling settings
zoomed_list = [9, 10, 11] # list with zoom levels to be inspected
sel_tile = 2 # idx of chosen tile level in zoomed_list (inspect the map to chose it accordingly), z9 too big for in memory computations

# Load google credentials
credential_file = os.path.join(main_fol, "00_miscellaneous", "KEYS", "bathymetry-543b622ddce7.json") # Cloud Storage credential key
if not credential_file == "":  
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_file)

# Get the Area of Interest (AoI)
# Open Area of Interest (AoI)
print('Area of Interest (AoI)')
with open(os.path.join(aoi_fol, project_name + ".geojson")) as f:
    AoIjson = geojson.loads(f.read())

# Get the bounds of the AoI
print('Bounds')
bounds = ee.Geometry(AoIjson["features"][0]["geometry"])

# Get the tiles for the AoI
print('Tiles')
tiles = list(tiler.get_tiles_for_geometry(bounds, zoom) for zoom in zoomed_list)

# %%
# Debugging
# Part of export_tiles function
tiles = tiler.get_tiles_for_geometry(geometry=bounds, zoom=ee.Number(zoomed_list[sel_tile]))

# Get tile with tx='1065.0' and ty='660.0'
tile = ee.Feature(tiles.filterMetadata("tx", "equals", "1065.0").filterMetadata("ty", "equals", "660.0").first())

# Get start and stop dates
start = ee.String(start_date)
stop = ee.String(stop_date)

# %%
# Part of get_tile_intertidal_bathymetry function
bounds: ee.Geometry = ee.Feature(tile).geometry().bounds(1)
sdb: Bathymetry = Bathymetry()
zoom: ee.String = ee.String(tile.get("zoom"))
tx: ee.String = ee.String(tile.get("tx"))
ty: ee.String = ee.String(tile.get("ty"))
tile_name: ee.String = ee.String("z").cat(zoom).cat("_x").cat(tx).cat("_y").cat(ty).replace("\.\d+", "", "g")
img_fullname: ee.String = ee.String(tile_name).cat("_t").cat(ee.Date(start).millis().format())

image: ee.Image = sdb.compute_intertidal_depth(
    bounds=bounds,
    start=start,
    stop=stop,
    scale=tiler.zoom_to_scale(ee.Number.parse(tile.get("zoom"))).multiply(5), # scale to search for clean images
    # missions=['S2', 'L8'],
    # filter: ee.Filter.dayOfYear(7*30, 9*30), # summer-only
    filter_masked=False, 
    # filterMaskedFraction = 0.5,
    # skip_scene_boundary_fix=False,
    # skip_neighborhood_search=False,
    neighborhood_search_parameters={"erosion": 0, "dilation": 0, "weight": 50},
    bounds_buffer=0,
    water_index_min=-0.05,
    water_index_max=0.15,
    # lowerCdfBoundary=45,
    # upperCdfBoundary=50,
    # cloud_frequency_threshold_data=0.15, 
    clip = True
)# .reproject(ee.Projection("EPSG:3857").atScale(90))

image = image.set(
        "fullname", img_fullname,
        "system:time_start", ee.Date(start).millis(),
        "system:time_stop", ee.Date(stop).millis(),
        "zoom", zoom,
        "tx", tx,
        "ty", ty
    )

# %%
# Get date of the processed image
ee_date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd HH:mm:ss")
date = ee_date.getInfo()
print(date)

# %%
# Get dates of all raw images
ee_dates = sdb._raw_images.aggregate_array("system:time_start").map(lambda date: ee.Date(date).format("YYYY-MM-dd HH:mm:ss"))
dates = ee_dates.getInfo()
print(dates)

# %%
# Plot the processed image
Map = geemap.Map()
Map.centerObject(bounds, 9)
Map.addLayer(bounds, {}, "AoI")
Map.addLayer(image.clip(bounds), {"bands": ["ndwi"], "min": -0.05, "max": 0.15}, "intertidal bathymetry")
Map

# %%
# Plot all raw images
Map = geemap.Map()
Map.centerObject(bounds, 9)
Map.addLayer(bounds, {}, "AoI")
from tqdm import tqdm
for idx in tqdm(range(sdb._raw_images.size().getInfo())):
    image_ = ee.Image(sdb._raw_images.toList(sdb._raw_images.size()).get(idx))
    Map.addLayer(image_.clip(bounds), {"bands": ["red", "green", "blue"], "min": 0, "max": 0.3}, f"image_{idx} (date: {dates[idx]})")
    if idx > 20:
        break
Map

# %%
# Export images for tiles
#export_tiles(sink="cloud", mode=mode, geometry=bounds, zoom=zoomed_list[sel_tile], start=start_date, stop=stop_date, 
#         scale=scale, step_months=compo_int, window_months=compo_len, overwrite=True, bucket=bucket)

# %%
