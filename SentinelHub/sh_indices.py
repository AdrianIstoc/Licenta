import numpy as np

def calc_ndvi(red, nir):
    return (nir - red) / (nir + red + 1e-6)

def calc_ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-6)

def calc_ndbi(swir, nir):
    return (swir - nir) / (swir + nir + 1e-6)

def calculate_ndvi(image):
    red = image[:,:,0].astype(float)
    nir = image[:,:,1].astype(float)
    cloud = image[:,:,2]

    cloud_mask = cloud == 1

    ndvi = ( (nir-red) / (nir+red+1e-6) )

    ndvi[cloud_mask] = np.nan

    return ndvi