from dataclasses import dataclass
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from yaml.error import YAMLError

from dacite import from_dict
from ee import ImageCollection, Filter, Geometry, Number, Reducer
from yaml import safe_load

from eobathymetry.cloud_mask_algorithms import *
from eobathymetry.missions import *


logger: Logger = getLogger(__name__)


@dataclass
class BandMapping:
    source: List[str]
    dest: List[str] = ['swir', 'nir', 'red', 'green', 'blue']
    temperature_mapping: Optional[Tuple[str]] = None


@dataclass
class AssetData:
    s2_merge_by_time: bool = True
    cloud_mask: bool = True
    resample: bool = False
    include_tier2: bool = False
    clip_buffer_size: int = 6000
    clip_buffer_size_l7: Optional[int] = None
    scale: int = 100
    filter: Optional[Filter] = None
    # TODO: bandsAll
    missions: List[Mission] = [s2, l8]
    band_mappings: Dict[Mission, BandMapping] = {
        s2: BandMapping(
                source=['B11', 'B8', 'B4', 'B3', 'B2']
        ),
        l8: BandMapping(
                source=['B6', 'B5', 'B4', 'B3', 'B2'],
                temperature_mapping=("B10", "temp")
        ),
        l7: BandMapping(
                source=['B5', 'B4', 'B3', 'B2', 'B1'],
                temperature_mapping=("B6_VCID_1", "temp")
        ),
        l5: BandMapping(
                source=['B5', 'B4', 'B3', 'B2', 'B1'],
                temperature_mapping=("B6", "temp")
        ),
        l4: BandMapping(
                source=['B5', 'B4', 'B3', 'B2', 'B1'],
                temperature_mapping=("B6", "temp")
        ),
    }


def getImages(g, options: Optional[Dict[Any]] = None):
    geometry: Geometry = Geometry(g)
    # Validate schema of options and get defaults
    asset_data: AssetData = from_dict(options)

    for mission in asset_data.missions:
        ic: ImageCollection = ImageCollection(mission.collection_id)
        if asset_data.include_tier2:
            ic = ic.merge(ImageCollection(mission.tier2_collection_id))
        ic = ic.filterBounds(geometry=geometry)
        if asset_data.filter:
            ic = ic.filter(asset_data.filter)
        if asset_data.cloud_mask:
            ic = ic.map(mission.cloud_mask_algorithm)
            band_mapping: BandMapping = asset_data.band_mappings[mission]
            band_mapping.source = band_mapping.source + ["cloud"]
            band_mapping.dest = band_mapping.dest + ["cloud"]
        ic.select(band_mapping.source, band_mapping.dest)
        if asset_data.clip_buffer_size_l7 and mission.name == MissionEnum.L7:
            def f(i):
                mask = i.select(['green', 'red', 'nir', 'swir']).mask().reduce(Reducer.allNonZero())
                mask = mask.focal_min(asset_data.clip_buffer_size_l7, 'square', 'meters').reproject(i.select('nir').projection())
                return i.updateMask(mask)

            ic = ic.map(f)
            def clipNegativeFootprint(i):
                return i.clip(i.select(0).geometry().buffer(Number(asset_data.clip_buffer_size).multiply(-1), 1000))

            l4 = l4.map(clipNegativeFootprint)
            l5 = l5.map(clipNegativeFootprint)
            l7 = l7.map(clipNegativeFootprint)


        
