from dataclasses import dataclass
from typing import Callable, Optional

from ee import Image

# TODO: Ask Gena if cloudMask is always:
# check cloud bit(s)
# check entire image on cloud bits
# return mask with name "cloud"
# In that case this can be more generic
def cloud_mask_algorithm_L8(image: Image) -> Image:
  qa: Image = image.select('BQA')
  # /// Check that the cloud bit is off.
  # // See https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band
  mask: Image = qa.bitwiseAnd(1 << 4).eq(0)
    
  return image.addBands(mask.rename('cloud'))


def cloud_mask_algorithm_S2(image: Image) -> Image:
  qa: Image = image.select('QA60')
  #  Bits 10 and 11 are clouds and cirrus, respectively.
  cloudBitMask: int = 1 << 10
  cirrusBitMask: int = 1 << 11
  # Both flags should be set to zero, indicating clear conditions.
  mask: Image = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
  mask: Image = mask.rename('cloud')
    
  return image.addBands(mask)


def cloud_mask_algorithm_default(image: Image) -> Image:
    cloudMask: Image = Image.constant(0).rename('cloud')
    
    # TODO: implement cloud masking
    
    return image.addBands(cloudMask)


@dataclass
class Mission:
    name: str
    collection_id: str
    tier2_collection_id: Optional[str] = None
    cloud_mask_algorithm: Callable = cloud_mask_algorithm_default


s2: Mission = Mission(
    name="S2",
    collection_id="COPERNICUS/S2",
    cloud_mask_algorithm=cloud_mask_algorithm_S2
)

l8: Mission = Mission(
    name="L8",
    collection_id="LANDSAT/LC08/C01/T1_RT_TOA",
    tier2_collection_id="LANDSAT/LC08/C01/T2_TOA",
    cloud_mask_algorithm=cloud_mask_algorithm_L8
)

l7: Mission = Mission(
    name="L7",
    collection_id="LANDSAT/LE07/C01/T1_RT_TOA",
    tier2_collection_id="LANDSAT/LE07/C01/T2_TOA"
)

l5: Mission = Mission(
    name="L5",
    collection_id="LANDSAT/LT05/C01/T1_TOA",
    tier2_collection_id="LANDSAT/LT05/C01/T2_TOA"
)

l4: Mission = Mission(
    name="L4",
    collection_id="LANDSAT/LT04/C01/T1_TOA",
    tier2_collection_id="LANDSAT/LT04/C01/T2_TOA"
)

@dataclass
class CloudMaskAlgorithm:
    mission: Mission
    algorithm: Callable
