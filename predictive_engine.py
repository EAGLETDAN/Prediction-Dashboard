import ee
import geemap
import pandas as pd
import numpy as np

# Initialize GEE
ee.Initialize(project='mygismapping')

# 1. Define Mau Forest Complex Boundary (Simplified)
mau_aoi = ee.Geometry.Polygon([
    [[35.20, -0.80], [35.90, -0.80], [35.95, -0.40], 
     [35.85, 0.10], [35.50, 0.20], [35.25, 0.05], 
     [35.15, -0.40], [35.20, -0.80]]
])

def get_landsat_forest_mask(year):
    """Generates a binary Forest (1) / Non-Forest (0) mask for a given year."""
    start_date = f'{year}-06-01'
    end_date = f'{year}-09-30'
    
    if year <= 2000: col = 'LANDSAT/LT05/C02/T1_L2'
    elif year == 2010: col = 'LANDSAT/LE07/C02/T1_L2'
    else: col = 'LANDSAT/LC08/C02/T1_L2'
    
    img = ee.ImageCollection(col).filterBounds(mau_aoi).filterDate(start_date, end_date).median()
    
    # Calculate NDVI
    nir = 'SR_B5' if year >= 2013 else 'SR_B4'
    red = 'SR_B4' if year >= 2013 else 'SR_B3'
    ndvi = img.normalizedDifference([nir, red]).rename('NDVI')
    
    # Forest threshold (NDVI > 0.45 for dense canopy)
    return ndvi.gt(0.45).selfMask().rename('Forest')

def get_driving_factors():
    """Calculates spatial variables that drive deforestation."""
    # 1. Topography (SRTM)
    dem = ee.Image("USGS/SRTMGL1_003").clip(mau_aoi)
    slope = ee.Terrain.slope(dem).rename('Slope')
    elevation = dem.rename('Elevation')
    
    # 2. Human Activity Proxy (VIIRS Nighttime Lights - annual composite)
    # Using 2020 as the baseline for human pressure
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG") \
        .filterDate('2020-01-01', '2020-12-31') \
        .select('cf_cvg').median().clip(mau_aoi).rename('Human_Pressure')
        
    # 3. Distance to Forest Edge (Crucial for edge-effect fragmentation)
    forest_2020 = get_landsat_forest_mask(2020)
    # Invert forest to get non-forest, then calculate distance
    non_forest = forest_2020.Not().selfMask()
    dist_to_edge = ee.Image(non_forest).fastDistanceTransform().sqrt().multiply(30).rename('Dist_to_Edge')
    
    return ee.Image.cat([elevation, slope,dist_to_edge])

def train_and_predict():
    """Trains the ML model on 2010-2020 changes and predicts 2030."""
    print("Extracting historical forest masks...")
    forest_2010 = get_landsat_forest_mask(2010)
    forest_2020 = get_landsat_forest_mask(2020)
    
    # Create the Target Variable: 1 = Deforested, 0 = Survived
    # Deforested = Was forest in 2010, but not in 2020
    deforested = forest_2010.And(forest_2020.Not()).rename('Target')
    # Survived = Was forest in 2010, and still forest in 2020
    survived = forest_2010.And(forest_2020).rename('Target')
    
    print("Calculating driving factors...")
    drivers = get_driving_factors()
    
    # Combine drivers and target
    training_image = drivers.addBands(deforested)
    
    print("Sampling training data (Stratified)...")
    # Sample 1000 pixels of deforestation, and 1000 pixels of surviving forest
    sample_def = training_image.stratifiedSample(
        numPoints=1000, classBand='Target', region=mau_aoi, scale=30, 
        geometries=False, seed=42
    )
    
    # For survived, we just sample from the 2010 forest mask to get '0' targets
    training_survived = drivers.addBands(survived).stratifiedSample(
        numPoints=1000, classBand='Target', region=mau_aoi, scale=30,
        geometries=False, seed=43
    )
    
    # Merge samples
    training_data = sample_def.merge(training_survived)
    
    print("Training Random Forest Classifier...")
    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=50, variablesPerSplit=2, minLeafPopulation=2
    ).train(
        features=training_data, 
        classProperty='Target', 
        inputProperties=drivers.bandNames()
    )
    
    print("Predicting 2030 Deforestation Probability...")
    # Apply model to 2020 drivers to get probability of deforestation (0 to 1)
    probability_map = drivers.classify(classifier, 'Deforestation_Probability')
    
    # Calculate Markov Quantity (How much forest was actually lost 2010-2020?)
    lost_area_ha = deforested.multiply(ee.Image.pixelArea()).divide(10000) \
        .reduceRegion(ee.Reducer.sum(), mau_aoi, 30).get('Target')
    
    return probability_map, lost_area_ha.getInfo()

# Run the engine
if __name__ == "__main__":
    prob_map, lost_ha = train_and_predict()
    print(f"Model trained. Historical loss (2010-2020): {lost_ha} Ha")
    
    # Export the probability map to Google Drive for the dashboard to use
    task = ee.batch.Export.image.toDrive(
        image=prob_map,
        description='Mau_2030_Probability',
        folder='Mau_Forest_Dashboard',
        region=mau_aoi,
        scale=30,
        maxPixels=1e9
    )
    task.start()
    print("Exporting prediction map to Google Drive... Check GEE Tasks tab.")