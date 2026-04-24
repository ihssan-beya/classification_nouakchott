import ee

def run_analysis(roi_coords):
    roi = ee.Geometry.Rectangle(roi_coords)m
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate('2023-01-01', '2023-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    image = collection.median().clip(roi)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    input_img = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).addBands([ndvi, ndwi, ndbi])
    eau = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-15.98, 18.10]), {'class': 0})])
    vegetation = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-15.95, 18.08]), {'class': 1})])
    urbain = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-15.97, 18.09]), {'class': 2})])
    sable = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-16.00, 18.12]), {'class': 3})])
    classes = eau.merge(vegetation).merge(urbain).merge(sable)
    samples = input_img.sampleRegions(collection=classes, properties=['class'], scale=10)
    classifier = ee.Classifier.smileRandomForest(50).train(samples, 'class', input_img.bandNames())
    classified = input_img.classify(classifier)
    return classified
