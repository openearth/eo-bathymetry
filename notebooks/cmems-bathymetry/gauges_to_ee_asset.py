# %%
# Notes
'''
Google Earth Engine accepts the following types: tif, shp and csv
- Tif files are used for raster data, and cannot be created row by row, structured
- Shp files have a maximum size of 2GB, and cannot be created row by row
- Csv files can be created row by row, and can be used for vector data
'''

# %%
# Packages
import os
import glob
import matplotlib.pyplot as plt
import xarray as xr
import dfm_tools as dfmt
from tqdm import tqdm
import numpy as np

# %%
# Settings
dir_output = r'p:\11209821-cmems-global-sdb\01_intertidal\02_data\04_gauge_files'
os.makedirs(dir_output, exist_ok=True)
time_min, time_max = '2021-01-01', '2022-01-01'

# %%
# Get gauge datasets
gdf_gesla = dfmt.ssh_catalog_subset(source='gesla3', time_min=time_min, time_max=time_max)
gdf_ioc = dfmt.ssh_catalog_subset(source='ioc', time_min=time_min, time_max=time_max)
gdf_cmems = dfmt.ssh_catalog_subset(source='cmems', time_min=time_min, time_max=time_max)
#gdf_uhslc_rqds = dfmt.ssh_catalog_subset(source='uhslc-rqds', time_min=time_min, time_max=time_max)
#gdf_uhslc_fast = dfmt.ssh_catalog_subset(source='uhslc-fast', time_min=time_min, time_max=time_max)
gdf_psmsl_gnssir = dfmt.ssh_catalog_subset(source='psmsl-gnssir', time_min=time_min, time_max=time_max)
gdf_rwsddl = dfmt.ssh_catalog_subset(source='rwsddl')

gdf_list = [gdf_gesla, gdf_ioc, gdf_cmems,
            #gdf_uhslc_rqds, gdf_uhslc_fast,
            gdf_psmsl_gnssir,
            gdf_rwsddl]

# %%
# plot gauge locations
fig, ax = plt.subplots(figsize=(12,8))
for gdf in tqdm(gdf_list):
    gdf.geometry.plot(ax=ax, marker="x", label='{} ({})'.format(gdf.iloc[0]["source"], len(gdf)))
ax.legend(loc=3)
dfmt.plot_coastlines(ax=ax, min_area=1000, linewidth=0.5, zorder=0)
dfmt.plot_borders(ax=ax, zorder=0)

# %%
# Retrieve gauge datasets (Note not all rwsddl gauges contain data)
download_data = False
if download_data:
    for gdf in tqdm(gdf_list):
        dfmt.ssh_retrieve_data(gdf, dir_output, time_min=time_min, time_max=time_max)

# %%
# plot the retrieved datasets
fig,ax = plt.subplots(figsize=(12,8))
file_path_ncs = glob.glob(os.path.join(dir_output, "*.nc"))
file_path_ncs.sort()
xs = np.ones(len(file_path_ncs))*np.nan
ys = np.ones(len(file_path_ncs))*np.nan
for idx, file_path_nc in tqdm(enumerate(file_path_ncs), total=len(file_path_ncs)):
    ds = xr.open_dataset(file_path_nc)
    xs[idx] = float(ds.station_x_coordinate)
    ys[idx] = float(ds.station_y_coordinate)
    station_name = os.path.basename(file_path_nc).strip(".nc")
    ds.waterlevel.values = np.where(np.abs(ds.waterlevel.values) > 100, np.nan, ds.waterlevel.values)
    ds.waterlevel.plot(ax=ax, label=station_name)
    del ds

# %%
# Plot the gauge locations
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(xs, ys)
dfmt.plot_coastlines(ax=ax, min_area=1000, linewidth=0.5, zorder=0)
dfmt.plot_borders(ax=ax, zorder=0)

# %%


# %%
# create overview plot and statistics csv
import shutil
import matplotlib.dates as md
import pandas as pd

def ssh_netcdf_overview(dir_netcdf, perplot=30, time_min=None, time_max=None, yearstep=None):
    """
    reads all netcdf files in a directory and makes figures of the non-nan waterlevel time availability
    it also writes a csv file with statistics
    """
    dir_output = os.path.join(dir_netcdf, "overview")
    if os.path.isdir(dir_output):
        shutil.rmtree(dir_output)
    os.makedirs(dir_output, exist_ok=False)
    
    file_list = glob.glob(os.path.join(dir_netcdf, "*.nc"))
    file_list = sorted(file_list, key=str.casefold)
    
    print(f"creating overview for {len(file_list)} files: ", end="")
    fig, ax = plt.subplots(figsize=(15,8))
    stats_list = []
    fig_file_list = []
    for ifile, file_nc in enumerate(file_list):
        print(f"{ifile+1}/{len(file_list)}", end=" ")
        
        fname = os.path.basename(file_nc)
        print(f"{ifile+1} ", end="")
        print(fname)
        
        ds = xr.open_dataset(file_nc)
        ds = ds.sortby("time") # necessary for BODC data
        
        fname_clean = fname.replace(".nc","")
        longitude = float(ds.station_x_coordinate)
        latitude = float(ds.station_y_coordinate)
        
        fig_file_list.append(fname_clean)
        
        # stats
        ds_ndays = round(int(ds.time.max() - ds.time.min())/1e9/3600/24, 2)
        nvalues = len(ds.waterlevel)
        nnan = int(ds.waterlevel.isnull().sum())
        time_diff_min = ds.time.to_pandas().diff().dt.total_seconds()/60
        
        stats_one = {"fname_clean":fname_clean,
                     "longitude":longitude,
                     "latitude":latitude,
                     "tstart":str(ds.time[0].dt.strftime("%Y-%m-%d").values),
                     "tstop":str(ds.time[-1].dt.strftime("%Y-%m-%d").values),
                     "ndays": ds_ndays,
                     "#values": nvalues,
                     "#nan": nnan,
                     "%nan": (nnan/nvalues)*100,
                     "dt min [min]":int(time_diff_min.min()),
                     "dt max [min]":int(time_diff_min.max()),
                     "dt mean [min]":time_diff_min.mean(),
                     "dt mode [min]":time_diff_min.mode().iloc[0],
                     "ndupl": ds.time.to_pandas().duplicated().sum(),
                     "min": float(ds.waterlevel.min()),
                     "max": float(ds.waterlevel.max()),
                     "contributor": ds.attrs["contributor_abbreviated"]
                     }
        stats_one_pd = pd.DataFrame(stats_one, index=[fname])
        stats_list.append(stats_one_pd)
        
        # derive unique hourly times with non-nan values
        bool_nan = ds.waterlevel.isnull()
        ds_nonan = ds.sel(time=~bool_nan)
        ds_slice = ds_nonan.sel(time=slice(time_min, time_max))
        # take unique timestamps after rounding to hours, this is faster and consumes less memory
        time_hr_uniq = ds_slice.time.to_pandas().index.round("h").drop_duplicates()
        time_yaxis_value = pd.Series(index=time_hr_uniq)
        time_yaxis_value[:] = -(ifile%perplot)
        if len(time_yaxis_value) != 0:
            time_yaxis_value.plot(ax=ax, marker='s', linestyle='none', markersize=1, color="r")
        
        # clear file links
        del ds
        
        bool_lastinrange = (ifile%perplot) == (perplot-1)
        bool_lastfile = ifile == (len(file_list)-1)
        if bool_lastinrange | bool_lastfile:
            # finish and save figure
            nlines = len(fig_file_list)
            ax.set_yticks(range(0,-nlines,-1), fig_file_list)
            figname = f"overview_availability_{ifile-nlines+2:04d}_{ifile+1:04d}"
            ax.set_xlim(time_min, time_max)
            if yearstep is not None:
                # set xtick steps
                ax.xaxis.set_major_locator(md.YearLocator(base=yearstep))
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
            ax.grid()
            ax.set_xlabel(None)
            fig.tight_layout()
            fig.savefig(os.path.join(dir_output, figname), dpi=200)
        
        if bool_lastinrange:
            # reset figure
            ax.cla()
            fig_file_list = []
    print()
    
    # write statistics csv
    stats = pd.concat(stats_list)
    stats.index.name = "file_name"
    file_csv = os.path.join(dir_output, "waterlevel_data_netcdf_overview.csv")
    stats.to_csv(file_csv, float_format="%.2f")

# create overview plot and statistics csv
ssh_netcdf_overview(dir_output, perplot=30)

# %%
# Packages
import pandas as pd
import xarray as xr
import glob
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%
# File paths
dir_output = r'p:\11209821-cmems-global-sdb\01_intertidal\02_data\04_gauge_files'
file_path_ncs = glob.glob(os.path.join(dir_output, "*.nc"))
file_path_csv = os.path.join(dir_output, 'overview', 'waterlevel_data_netcdf_overview.csv')

# %%
# Read nc
ds = xr.open_dataset(file_path_ncs[0])

# %%
# Read csv
df = pd.read_csv(file_path_csv)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

# Plot csv
fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree()})
gdf.plot(ax=ax, marker=".", color="red", markersize=20, zorder=10)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.COASTLINE)
ax.gridlines(draw_labels=True)
fig.savefig(os.path.join(dir_output, 'overview', 'gauge_locations.png'))

# Plot csv zoom Netherlands
fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree()})
gdf.plot(ax=ax, marker=".", color="red", markersize=20, zorder=10)
ax.set_extent([3, 8, 51, 54])
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.gridlines(draw_labels=True)
fig.savefig(os.path.join(dir_output, 'overview', 'gauge_locations_NL.png'))


# %%
# %%
