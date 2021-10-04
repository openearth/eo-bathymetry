import math
from typing import Optional, List, Tuple

import ee

TILE_SIZE = 256
ORIGIN = math.pi * 6378137 # earth depth, Origin is: 2 * Math.PI * 6378137 / 2.0
C: float = 40075016.686  # earth circumpherence
PROJECTION: str = "EPSG:3857"

def zoom_to_scale(zoom: int) -> float:
    
    tile_width: float = C / math.pow(2, zoom)  # Origin is: tileWidth = C * Math.cos(0) / Math.pow(2, zoom)
    return tile_width / TILE_SIZE

def scale_to_zoom(scale: float) -> int:
    tile_width: float = scale * TILE_SIZE
    return math.ceil(math.log(C / tile_width) / math.log(2))

def pixels_to_meters(px: float, py: float, zoom: int) -> List[float]:
    resolution: float = zoom_to_scale(zoom)
    x: float = px * resolution - ORIGIN
    y: float = py * resolution - ORIGIN

    return [x, y]

def meters_to_pixels(x: float, y: float, zoom: int) -> Tuple[float]:
    resolution: float = zoom_to_scale(zoom)
    px: float = (x + ORIGIN) / resolution
    py: float = (x + ORIGIN) / resolution
    return px, py

def to_radians(degrees: float) -> float:
    return degrees * math.pi / 180

def to_degrees(radians: float) -> float:
    return radians * 180 / math.pi

def degrees_to_tiles(lon: float, lat: float, zoom: int) -> List[int]:
    tx: int = math.floor((lon + 180) / 360 * math.pow(2, zoom))
    ty: int = math.floor(
        (1 - math.log(math.tan(to_radians(lat)) + 1 / math.cos(to_radians(lat))) / math.pi) /2 * math.pow(2, zoom)
    )
    return [tx, ty]

def tiles_to_degrees(tx: int, ty: int, zoom: int) -> Tuple[float]:
    lon: float = tx / math.pow(2, zoom) * 360 - 180
    n: float = math.pi - 2 * math.pi * ty / math.pow(2, zoom)
    lat: float = to_degrees(math.atan(0.5 * (math.exp(n) - math.exp(-n))))

    return [lon, lat]

def get_tile_bounds(tx: int, ty: int, zoom: int) -> List[float]:
    ty = math.pow(2, zoom) - ty - 1 # TMS -> XYZ, flip y index
    min: float = pixels_to_meters(tx * TILE_SIZE, ty * TILE_SIZE, zoom)
    max: float = pixels_to_meters((tx + 1) * TILE_SIZE, (ty + 1) * TILE_SIZE, zoom)
    return [min, max]

def get_tiles_for_geometry(geometry: ee.Geometry, zoom: int, opt_bounds: Optional[ee.Geometry] = None) -> ee.FeatureCollection:
    bounds: ee.Geometry = geometry
    if opt_bounds:
        bounds = opt_bounds
    bounds_list: ee.List = ee.List(bounds.bounds().coordinates().get(0))

    ll: List[float] = bounds_list.get(0).getInfo() # TODO: port to server-side
    ur: List[float] = bounds_list.get(2).getInfo()

    tmin = degrees_to_tiles(ll[0], ll[1], zoom)
    tmax = degrees_to_tiles(ur[0], ur[1], zoom)

    tiles: List = []
    for tx in range(tmin[0], tmax[0]+1):
        for ty in range(tmax[1], tmin[1]+1):
            temp_bounds: List[float] = get_tile_bounds(tx, ty, zoom)
            rect: ee.Geometry = ee.Geometry.Rectangle(temp_bounds, PROJECTION, False)
            tiles.append(ee.Feature(rect).set({"tx": tx, "ty": ty, "zoom": zoom}))
    
    return ee.FeatureCollection(tiles).filterBounds(geometry)