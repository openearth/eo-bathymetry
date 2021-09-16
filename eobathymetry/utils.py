import ee

def focalMin(image: ee.Image, radius: float):
    erosion: ee.Image = image.Not().fastDistanceTransform(radius).sqrt().lte(radius).Not()
    return erosion

def focalMax(image: ee.Image, radius: float):
    dilation: ee.Image = image.fastDistanceTransform(radius).sqrt().lte(radius)
    return dilation

def focalMaxWeight(image: ee.Image, radius: float):
    distance: ee.Image = image.fastDistanceTransform(radius).sqrt()
    dilation: ee.Image = distance.where(distance.gte(radius), radius)
    dilation = ee.Image(radius).subtract(dilation).divide(radius)
    return dilation