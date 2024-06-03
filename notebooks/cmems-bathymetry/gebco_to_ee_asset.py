# %%
# Notes
'''
Google Earth Engine accepts the following types: tif, shp and csv
- Tif files are used for raster data, and cannot be created row by row, structured <--
- Shp files have a maximum size of 2GB, and cannot be created row by row
- Csv files can be created row by row, and can be used for vector data
'''

# %%
# Packages
import rioxarray as rxr
import os
import glob
from tqdm import tqdm
from osgeo import gdal
gdal.UseExceptions() # Allow GDAL to throw Python Exceptions

# %%
# File paths
dir_path_data = r'p:\11209821-cmems-global-sdb\01_intertidal\02_data\03_gebco_files'
file_path_ncs = glob.glob(os.path.join(dir_path_data, '*.nc'))
file_path_tifs = [os.path.join(dir_path_data, os.path.basename(file_path).replace('.nc', '.tif')) for file_path in file_path_ncs]

file_path_hat_tifs = [file_path for file_path in file_path_tifs if 'hat.tif' in file_path]
file_path_lat_tifs = [file_path for file_path in file_path_tifs if 'lat.tif' in file_path]

file_path_hat_merge_tif = os.path.join(dir_path_data, 'gebco_2023_hat.tif')
file_path_lat_merge_tif = os.path.join(dir_path_data, 'gebco_2023_lat.tif')

file_path_merge_tif = os.path.join(dir_path_data, 'gebco_2023_hat_lat.tif')

# %%
# Function to merge multiple tifs into one
def merge_grids_by_xy(file_paths_in, file_path_out, **kwargs):
    """Merge multiple tifs into one.

    :param file_paths_in: List of paths to tifs.
    :type file_paths_in:  list
    :param file_path_out: Path to output tif.
    :type file_path_out:  str
    :param kwargs:        Keyword arguments for :func:`gdal.Translate`.
    :type kwargs:         dict, optional
    :return:              None
    :rtype:               None

    See also: `gdal.BuildVRT() <https://gdal.org/programs/gdalbuildvrt.html#gdalbuildvrt>`_,
    `gdal.Translate() <https://gdal.org/programs/gdal_translate.html#gdal-translate>`_
    """

    # Get file path vrt
    file_path_vrt = file_path_out.replace('.tif', '.vrt')

    # Get keyword arguments
    creationOptions=['DRIVER=GTiff','COMPRESS=LZW','INTERLEAVE=BAND','ZLEVEL=9','PREDICTOR=1', 'BIGTIFF=YES']
    kwargs.setdefault('creationOptions', creationOptions)

    # Merge tifs
    vrt = gdal.BuildVRT(file_path_vrt, file_paths_in)
    ds = gdal.Translate(file_path_out, vrt, **kwargs)
    vrt = ds = None

    # Remove vrt
    if os.path.exists(file_path_vrt):
        os.remove(file_path_vrt)

def merge_grids_by_band(file_paths_in, file_path_out, **kwargs):
    """Merge multiple tifs into one by band.

    :param file_paths_in: List of paths to tifs.
    :type file_paths_in:  list
    :param file_path_out: Path to output tif.
    :type file_path_out:  str
    :param kwargs:        Keyword arguments for :func:`gdal.Translate`.
    :type kwargs:         dict, optional
    :return:              None
    :rtype:               None

    See also: `gdal.BuildVRT() <https://gdal.org/programs/gdalbuildvrt.html#gdalbuildvrt>`_,
    `gdal.Translate() <https://gdal.org/programs/gdal_translate.html#gdal-translate>`_
    """

    # Get file path vrt
    file_path_vrt = file_path_out.replace('.tif', '.vrt')

    # Get keyword arguments
    creationOptions=['DRIVER=GTiff','COMPRESS=LZW','INTERLEAVE=BAND','ZLEVEL=9','PREDICTOR=1', 'BIGTIFF=YES']
    kwargs.setdefault('creationOptions', creationOptions)

    # Merge tifs
    vrt = gdal.BuildVRT(file_path_vrt, file_paths_in, separate=True)
    ds = gdal.Translate(file_path_out, vrt, **kwargs)
    vrt = ds = None

    # Remove vrt
    if os.path.exists(file_path_vrt):
        os.remove(file_path_vrt)

def name_bands(file_path_in, names):
    """Name bands in a tif file.

    :param file_path_in: Path to tif file.
    :type file_path_in:  str
    :param names:        List of names for the bands.
    :type names:         list
    :return:             None
    :rtype:              None
    """
    
    # Open file
    ds = gdal.Open(file_path_in, gdal.GA_Update)

    # Name bands
    for i, name in enumerate(names):
        band = ds.GetRasterBand(i+1)
        band.SetDescription(name)

    # Close file
    ds = None

# %%
# Convert the nc files to tif files
for file_path_nc, file_path_tif in tqdm(zip(file_path_ncs, file_path_tifs), total=len(file_path_ncs)):
    # Skip if the tif file already exists
    if os.path.exists(file_path_tif):
        continue

    # Open file
    ds = rxr.open_rasterio(file_path_nc)

    # Write file
    ds.rio.to_raster(file_path_tif)

# Merge the hat and lat tifs by xy
print('Merging the hat and lat tifs by xy')
if not os.path.exists(file_path_hat_merge_tif):
    merge_grids_by_xy(file_path_hat_tifs, file_path_hat_merge_tif)

if not os.path.exists(file_path_lat_merge_tif):
    merge_grids_by_xy(file_path_lat_tifs, file_path_lat_merge_tif)

# Merge the hat and lat tifs by band
print('Merging the hat and lat tifs by band')
if not os.path.exists(file_path_merge_tif):
    merge_grids_by_band([file_path_hat_merge_tif, file_path_lat_merge_tif], file_path_merge_tif)

# Name the bands
print('Naming the bands')
name_bands(file_path_merge_tif, ['hat', 'lat'])

# %%
# Notes
'''
Manually upload the tif files to the Google Cloud Bucket.
'''

# %%
# Transfer tif files from Google Cloud Bucket to a Google Earth Engine Asset
google_cloud_bucket_file_path = 'gs://cmems-sdb/gebco/gebco_2023_hat_lat.tif'
asset_id = 'projects/bathymetry/assets/gebco_2023_hat_lat'
command = 'earthengine upload image --asset_id={} {}'.format(asset_id, google_cloud_bucket_file_path)
out = os.system(command)

# Check if the tasks were successfully submitted
print('Task submitted successfully') if out == 0 else print('Task failed')

# %%
# Get the Google Earth Engine Asset
import ee
ee.Initialize()

print('Getting the Google Earth Engine Asset')
gebco_image = ee.Image('projects/bathymetry/assets/gebco_2023_hat_lat')

# %%
