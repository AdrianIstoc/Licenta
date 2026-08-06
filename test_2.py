from SentinelHub.sh_connection import Connection
from utils import plot_image

import numpy as np
from datetime import date
from sentinelhub import (
    SHConfig,
    DataCollection,
    BBox,
    CRS,
    SentinelHubRequest,
    MimeType,
    bbox_to_dimensions
)

config = Connection()
S2_L2A_CDSE = DataCollection.SENTINEL2_L2A.define_from(
    "S2_L2A_CDSE",
    service_url=config.sh_base_url
)
print(S2_L2A_CDSE.service_url)

bbox = BBox(bbox=[25.5, 47.45, 25.55, 47.50], crs=CRS.WGS84)
dimensions = bbox_to_dimensions(bbox, resolution=10)
time_interval = ("2024-05-01", "2024-05-31")

evalscript_test = """
//VERSION=3
function setup() {
    return {
        input: ["B01"],
        output: { bands: 1}
    };
}
function evaluatePixel(sample) {
    return [
        sample.B01
    ];
}
"""

def get_ndvi():
    request = SentinelHubRequest(
        evalscript=evalscript_test,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=S2_L2A_CDSE,
                time_interval=time_interval
            )
        ],
        responses=[
            SentinelHubRequest.output_response(
                "default",
                MimeType.PNG
            )
        ],
        bbox=bbox,
        size=dimensions,
        config=config,
        data_folder="./data"
    )


    image = request.get_data()[0]
    return image

image = get_ndvi()
    
plot_image(image, factor=1/255, clip_range=(0, 1))


# def get_ndvi(an: int):
#     request = SentinelHubRequest(
#         evalscript=evalscript_brut,
#         input_data=[
#             SentinelHubRequest.input_data(
#                 data_collection=DataCollection.SENTINEL2_L2A,
#                 time_interval=(f"{an}-07-01", f"{an}-08-31"),
#                 mosaicking_order="leastCC"
#             )
#         ],
#         responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
#         bbox=bbox,
#         size=dimensions,
#         config=config
#     )

#     satelite_data = request.get_data()[0]
#     banda_rosie = satelite_data[:, :, 0]
#     banda_nir = satelite_data[:, :, 1]

#     ndvi = (banda_nir - banda_rosie) / (banda_nir + banda_rosie + 0.0001)
#     return ndvi

# ndvi_2022= get_ndvi(2022)
# ndvi_2025 = get_ndvi(2025)

# padure_2022 = ndvi_2022 > 0.6
# pixeli_defrisati = (ndvi_2022 > 0.6) & (ndvi_2025 < 0.4)

# total_pixeli_padure_initiala = np.sum(padure_2022)
# total_pixeli_taiati = np.sum(pixeli_defrisati)

# if total_pixeli_padure_initiala > 0:
#     procent_defrisare = (total_pixeli_taiati / total_pixeli_padure_initiala) * 100
# else:
#     procent_defrisare = 0.0

# print(f"Padure initiala: {total_pixeli_padure_initiala}")
# print(f"Padure defrisata: {total_pixeli_taiati}")
# print(f"procent total de defrisare: {procent_defrisare:.2f}%")