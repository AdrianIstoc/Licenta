from utils import plot_image
from sh_connection import SentinelHubConnection
from sentinelhub import (
    DataCollection,
    SentinelHubRequest,
    BBox,
    bbox_to_dimensions,
    CRS,
    MimeType,
)
from sh_indices import (
    calc_ndvi,
    calc_ndwi,
    calc_ndbi
)
import sh_collections
import sh_download
import matplotlib.pyplot as plt
import numpy as np

connection = SentinelHubConnection()
config = connection.get_config()

collection = sh_collections.get_sentinel2_l2a(config)

bbox = BBox(bbox=[ 26.792222, 47.007858, 26.848514, 46.972363], crs=CRS.WGS84)
dimensions = bbox_to_dimensions(bbox, resolution=10)
time_interval=("2022-07-01", "2022-07-15")

bands = ["B03", "B04", "B08", "B11"]


image = sh_download.download_bands(
    config,
    collection,
    bbox,
    time_interval,
    bands
)

green = image[:,:,0].astype(float)
red = image[:,:,1].astype(float)
nir = image[:,:,2].astype(float)
swir = image[:,:,3].astype(float)

ndvi = calc_ndvi(red, nir)
ndwi = calc_ndwi(green, nir)
ndbi = calc_ndbi(swir, nir)

classification = np.zeros(ndvi.shape, dtype=np.uint8)
classification[ndwi > 0.3] = 1
classification[ndvi > 0.6] = 2
classification[ndbi > 0.2] = 3

plt.figure(figsize=(8,6))
plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(label="NDVI")
plt.title("NDVI")
plt.show()

plt.figure(figsize=(8,6))
plt.imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
plt.colorbar(label="NDWI")
plt.title("NDWI")
plt.show()

plt.figure(figsize=(8,6))
plt.imshow(ndbi, cmap="OrRd", vmin=-1, vmax=1)
plt.colorbar(label="NDBI")
plt.title("NDBI")
plt.show()

# plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
# plt.colorbar(label="NDVI")
# plt.show()
