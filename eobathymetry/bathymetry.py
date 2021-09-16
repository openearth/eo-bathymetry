from re import I
from typing import Any, Dict, List, Optional
from datetime import datetime

import ee

from eobathymetry.missions import Mission, s2, l8
from eobathymetry import assets
from eobathymetry import gl

class Bathymetry(object):
    def __init__(self, missions: List[Mission] = [s2, l8], waterIndexMin: float = -0.15, waterIndexMax: float = 0.35):
        self.missions = missions
        self.waterIndexMin = waterIndexMin
        self.waterIndexMax = waterIndexMax
    
    @property
    def missions(self) -> List[Mission]:
        return self._missions

    @missions.setter
    def missions(self, values) -> None:
        self._missions = values

    # TODO: Check if we need getters setters in the next few methods, or in other classes, otherwise leave out.

    @staticmethod
    def _remove_all_zero_images(image: ee.Image):
        mask: ee.Image = image.select(['blue', 'green', 'red', 'nir', 'swir']).mask().reduce(ee.Reducer.allNonZero())
        return image.updateMask(mask)

    def compute_inverse_depth(
        self,
        bounds,
        start: datetime,
        stop: datetime,
        filter_masked: bool,
        scale: float,
        missions: List[str] = ["S2", "L8"],
        cloud_frequency_threshold_data: float = 0.15,
        pansharpen: bool = False,
        skip_neighborhood_searh: bool = False,
        bounds_buffer: int = 10000,
    ) -> ee.Image:
        images: ee.ImageCollection = self.get_images(
            bounds=bounds,
            start=start,
            stop=stop,
            filter_masked=filter_masked,
            scale=scale,
            missions=missions,
            cloud_frequency_threshold_delta=cloud_frequency_threshold_data
        )
        # save loaded images in class as raw_images
        self.raw_images = images
        images = images.map(self._remove_all_zero_images)
        # Always has a value?
        if bounds_buffer:
            bounds = bounds.buffer(bounds_buffer, bounds_buffer / 10)

        return self._compute_inverse_depth(
            images=images,
            bounds=bounds,
            scale=scale,
            pansharpen=pansharpen,
            skip_neighborhood_search=skip_neighborhood_searh
        )


    def _compute_inverse_depth(self, images: ee.ImageCollection, bounds, scale: int, pansharpen: bool, skip_neighborhood_search: bool) -> ee.Image:
        bands: List[str] = ["red", "green", "blue"]
        green_max: float = 0.4

        def _set_image_area_properties(image: ee.Image) -> ee.Image:
            water: ee.Image = image.normalizedDifference(['green', 'nir']).rename('water').unitScale(0, 0.1)
            
            if pansharpen:
                image = pansharpen(image)

            water_area: float = water.gt(0.01).multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=bounds,
                scale=scale * 5,
                tileScale=4
            ).values().get(0)

            land_area: float = water.lt(0).multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=bounds,
                scale=scale * 5,
                tileScale=4
            ).values().get(0)

            dark = image

            dark = dark \
                .updateMask(water.gt(0)) \
                .reduceRegion(
                    reducer=ee.Reducer.percentile([0]),
                    geometry=bounds,
                    scale=scale,
                    maxPixels=1e10,
                    tileScale=4
                )
            
            image = image \
                .set(dark) \
                .set({
                    "water": water,
                    "waterArea": water_area,
                    "landArea": land_area
                })
            
            return image

        images: ee.ImageCollection = images.map(_set_image_area_properties)
        self.images_step_1 = images
        
        # Filter images with negative RGB values
        images = images.filter(ee.Filter.And(
            ee.Filter.gt(bands[0], 0),
            ee.Filter.gt(bands[1], 0),
            ee.Filter.gt(bands[2], 0)
        ))

        if skip_neighborhood_search:
            images = assets.addCdfQualityScore(images=images, opt_thresholdMin=70, opt_thresholdMax=80, opt_includeNeighborhood=False)
        else:
            images = assets.addCdfQualityScore(
                images=images,
                opt_thresholdMin=70,
                opt_thresholdMax=80,
                opt_includeNeighborhood=True,
                opt_neighborhoodOptions={
                    "erosion": 0,
                    "dilation": 0,
                    "weight": 200
                })

        def image_map_func(i: ee.Image):
            t: str = i.get("system:time_start")
            weight: ee.Image = i.select("weight")

            dark_image: ee.Image = ee.Image.constant(list(map(lambda n: i.get(n), bands))).rename(bands)
            mission: str = i.get("MISSION")
            scale_water_to: str = "percentiles"
            scale_land_to: str = "percentiles"

            range_percentiles_water: List[int] = [2, 98]
            range_percentiles_land: List[int] = [2, 98]
            
            range_sigma_water: List[int] = [1, 1]
            range_sigma_land: List[int] = [2, 2]

            water: ee.Image = ee.Image(i.get("water"))
            non_water: ee.Image = water.subtract(1).multiply(-1)

            i = i.select(bands).subtract(dark_image).max(0.0001)

            i_all: ee.Image = i

            water2: ee.Image = gl.smoothStep(-0.05, 0.2, water)
            non_water2: ee.Image = water2.subtract(1).multiply(-1)

            i = i.log()

            stat1: ee.Image = i

            stat1 = stat1.updateMask(water2.multiply(i_all.select("green").lt(green_max)))

            if scale_water_to=="percentiles":
                stat1 = stat1.reduceRegion(
                    reducer= ee.Reducer.percentile(range_percentiles_water),
                    geometry=bounds,
                    scale=scale * 3,
                    maxPixels=1e10,
                )
                
                min1: List[str] = [ee.String(stat1.get(band)).cat("_p").cat(str(range_percentiles_water[0])) for band in bands]
                max1: List[str] = [ee.String(stat1.get(band)).cat("_p").cat(str(range_percentiles_water[1])) for band in bands]

            if scale_water_to=="sigma":
                stat1mean = stat1.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=bounds,
                    scale=scale*3,
                    maxPixels=1e10
                )

                stat1sigma = stat1.reduceregion(  # not being used.
                    reducer=ee.Reducer.stDev(),
                    geometry=bounds,
                    scale=scale*3,
                    maxPixels=1e10
                )

                # Not sure whether this si tested, min1 should always be zero
                min1: List[ee.Number] = [ee.Number(stat1mean.get(band)).subtract(ee.Number(stat1mean.get(band)).multiply(range_sigma_water[0])) for band in bands]
                max1: List[ee.Number] = [ee.Number(stat1mean.get(band)).add(ee.Number(stat1mean.get(band)).multiply(range_sigma_water[1])) for band in bands]
            
            min1 = self._fix_null(min1, 0)
            max1 = self._fix_null(max1, 0.001)

            stat2: ee.Image = i_all.updateMask(non_water2.multiply(i_all.select("green").lt(green_max)))

            if scale_land_to=="percentiles":
                stat2 = stat2.reduceRegion(
                    reducer=ee.Reducer.percentile(range_percentiles_land),
                    geometry=bounds,
                    scale=scale*3,
                    maxPixels=1e10
                )

                min2 = [ee.String(stat2.get(band)).cat("_p").cat(str(range_percentiles_land[0])) for band in bands]
                max2 = [ee.String(stat2.get(band)).cat("_p").cat(str(range_percentiles_land[1])) for band in bands]
                
            if scale_land_to=="sigma":
                stat2mean = stat2.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=bounds,
                    scale=scale*3,
                    maxPixels=1e10
                )
                stat2sigma = stat2.reduceRegion(  # not used?
                    reducer=ee.Reducer.stDev(),
                    geometry=bounds,
                    scale=scale*3,
                    maxPixels=1e10
                )

                min2: List[ee.Number] = [ee.Number(stat2mean).get(band).subtract(ee.Number(stat2mean.get(band)).multiply(range_sigma_land[0])) for band in bands]
                max2: List[ee.Number] = [ee.Number(stat2mean.get(band)).add(ee.Number(stat2mean.get(band)).multiply(range_sigma_land[1])) for band in bands]

            min2 = self._fix_null(min2, 0)
            max2 = self._fix_null(max2, 0.001)

            i_water = self._unit_scale(i.select(bands), min1, max1).updateMask(water2)

            i_land = self._unit_scale(i_all.select(bands), min2, max2).updateMask(non_water2)

            i = i_water.blend(i_land).addBands(water)

            i = i.addBands(weight)

            return i.set({
                "label": ee.Date(t).format().cat(", ").cat(mission),
                "system:time_start": t
            })
        
        images = images.map(image_map_func)
        self.images_step_2 = images

        # mean = sum(w * x) / sum(w)
        image: ee.Image = images.map(lambda i: i.select(bands.append("water")).multiply(i.select("weight"))) \
            .sum().divide(images.select("weight").sum())
        
        return image

    def compute_intertidal_depth(options: Dict[str, Any]) -> ee.Image:
        pass

    @staticmethod
    def get_images(
        bounds,
        start: datetime,
        stop: datetime,
        filter_masked: bool,
        scale: float,
        missions: List[str],
        filter_masked_fraction: Optional[float] = None,
        cloud_frequency_threshold_delta: float = 0.15,
        filter: Optional[ee.Filter] = None 
    ) -> ee.ImageCollection:
        date_filter: ee.Filter = ee.Filter.date(start, stop)
        if filter:
            filter: ee.Filter = ee.Filter.And(filter, date_filter)
        else:
            filter: ee.Filter = ee.Filter.date(date_filter)

        options_get_images: Dict[str, Any] = {
            "missions": missions,
            "filter": filter,
            "filterMasked": filter_masked,
            "filterMaskedFraction": filter_masked_fraction,
            "scale": scale * 10,  # why *10?
            "resample": True
        }

        images: ee.ImageCollection = assets.getImages(bounds, options_get_images)

        options_get_mostly_clean_images: Dict[str, Any] = {
            "cloudFrequencyThresholdData": cloud_frequency_threshold_delta
        }

        return assets.getMostlyCleanImages(images, bounds, options_get_mostly_clean_images)

    @staticmethod
    def _fix_null(values, v) -> ee.List:
        return ee.List(values).map(lambda o: ee.Algorithms.If(ee.Algorithms.IsEqual(o, None), v, o))
    
    @staticmethod
    def _unit_scale(image: ee.Image, min: float, max: float) -> ee.Image:
        min_image: ee.Image = ee.Image.constant(min)
        max_image: ee.Image = ee.Image.constant(max)
        return image.subtract(min_image).divide(max_image.subtract(min_image))
        