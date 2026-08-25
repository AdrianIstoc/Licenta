import numpy as np

def calculate_ndvi(image):
    red = image[:,:,0].astype(float)
    nir = image[:,:,1].astype(float)
    cloud = image[:,:,2]

    cloud_mask = cloud == 1

    ndvi = ( (nir-red) / (nir+red+1e-6) )

    ndvi[cloud_mask] = np.nan

    return ndvi

def calculate_ndwi(image):
    green = image[:,:,0].astype(float)
    nir = image[:,:,1].astype(float)
    cloud= image[:,:,2]

    cloud_mask = cloud == 1

    ndwi = ((green - nir) / (green + nir+1e-6))

    ndwi[cloud_mask] = np.nan
    return ndwi