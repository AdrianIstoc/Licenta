import numpy as np

def calc_ndvi(red, nir):
    return (nir - red) / (nir + red + 1e-6)

def calc_ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-6)

def calc_ndbi(swir, nir):
    return (swir - nir) / (swir + nir + 1e-6)