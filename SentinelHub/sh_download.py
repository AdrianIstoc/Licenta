from datetime import date
from SentinelHub.sh_indices import (
    calculate_ndvi,
    calculate_ndwi,
)
from sentinelhub import (
    SentinelHubRequest,
    MimeType,
    bbox_to_dimensions
)
from SentinelHub.sh_collections import COLLECTIONS
import numpy as np


def choose_resolution(bbox, resolution, max_pixels=2000):
    resolution = resolution

    width, height = bbox_to_dimensions(bbox=bbox, resolution=resolution)

    while width>max_pixels or height>max_pixels:
        resolution *= 2

        width,height= bbox_to_dimensions(bbox=bbox, resolution=resolution)

    return resolution

def download(
        config,
        collection,
        bbox,
        time_interval,
        evalscript,
        resolution = 10,
        mime_type = MimeType.TIFF,
        mosaicking_order=None,
        maxcc=None,
        data_folder="./data"
):
    size = bbox_to_dimensions(bbox, resolution=resolution)
    input_param = {
        "data_collection": collection,
        "time_interval": time_interval
        }
    
    if mosaicking_order is not None:
        input_param["mosaicking_order"] = mosaicking_order

    if maxcc is not None:
        input_param["maxcc"] = maxcc

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(**input_param)
        ],
        responses=[
            SentinelHubRequest.output_response(
                "default",
                mime_type
            )
        ],
        bbox=bbox,
        size=size,
        config=config,
        data_folder=data_folder
    )
    
    return request.get_data()[0]
    
def build_evalscript(
        bands,
        clm = False,
        scl = False,
        data_mask = False,
        sample_type="AUTO"
        ):
    inputs = list(bands)
    if clm:
        inputs.append("CLM")
    if scl:
        inputs.append("SCL")
    if data_mask:
        inputs.append("dataMask")
    input_string = ",".join(f'"{band}"' for band in inputs)
    output_string = ",\n\t".join(f"sample.{band}" for band in inputs)


    return f"""
//VERSION=3

function setup() {{
    return {{
        input: [{input_string}],
        output: {{
            bands: {len(inputs)},
            sampleType: "{sample_type}"
        }}
    }};
}}

function evaluatePixel(sample) {{
    return [
        {output_string}
    ];
}}
"""

def download_bands(
        config,
        collection,
        bbox,
        time_interval,
        bands,
        resolution=10,
        mime_type=MimeType.TIFF,
        mosaicking_order=None,
        maxcc=None,
        clm=False,
        scl=False,
        data_mask=False,
        sample_type="AUTO"
):
    evalscript = build_evalscript(
        bands=bands,
        clm=clm,
        scl=scl,
        data_mask=data_mask,
        sample_type=sample_type
    )
    resolution = choose_resolution(bbox=bbox, resolution=resolution)
    
    return download(
        config=config,
        collection=collection,
        bbox=bbox,
        time_interval=time_interval,
        evalscript=evalscript,
        resolution=resolution,
        mime_type=mime_type,
        mosaicking_order=mosaicking_order,
        maxcc=maxcc
    )

def create_periods(start_date, end_date):
        periods = []

        current = start_date

        while current < end_date:
            if current.month==12:
                next_date = date(current.year+1,1,1)
            else:
                next_date = date(current.year, current.month+1,1)

            if next_date>end_date:
                next_date=end_date

            periods.append((current, next_date))

            current=next_date

        return periods

def download_ndvi_data(config, collection, bbox, start_date, end_date):
    results = []

    periods = create_periods(start_date=start_date, end_date=end_date)


    for period_start, period_end in periods:
        try:
            image = download_bands(
                config=config,
                collection=collection,
                bbox=bbox,
                time_interval=(period_start, period_end),
                bands=["B04", "B08"],
                mosaicking_order="leastCC",
                maxcc=0.2,
                clm=True,
                scl=True,
                data_mask=True,
                sample_type="FLOAT32"
            )
            ndvi=calculate_ndvi(image=image)

            slc=image[:,:,3]
            data_mask=image[:,:,4]

            vegetation_mask=(
                (slc == 4) #vegetatie
                & (data_mask == 1) #exista valori
            )
            masked_ndvi = np.where(vegetation_mask, ndvi, np.nan)

            vegetation_ndvi = ndvi[vegetation_mask]
            if vegetation_ndvi.size == 0:
                print(f"No valid vegetation for {period_start}")
                continue

            mean_ndvi = np.nanmean(vegetation_ndvi)

            vegetation_percentage = (np.sum(vegetation_mask)/vegetation_mask.size)*100

            results.append({
                "image": masked_ndvi,
                "date": period_start,
                "value": mean_ndvi,
                "percentage": vegetation_percentage
            })
            
            print(f"Downloaded vegetation for {period_start}")
        except Exception as e:
            print(f"Faild for {period_start}: {e}")

    return {"option": "NDVI", "results": results}

def download_ndwi_data(config, collection, bbox, start_date, end_date):
    results = []

    periods = create_periods(start_date=start_date, end_date=end_date)

    for period_start, period_end in periods:
        try:
            image = download_bands(
                config=config,
                collection=collection,
                bbox=bbox,
                time_interval=(period_start, period_end),
                bands=["B03", "B08"],
                mosaicking_order="leastCC",
                maxcc=0.2,
                clm=True,
                scl=True,
                data_mask=True,
                sample_type="FLOAT32"
            )
            ndwi = calculate_ndwi(image)

            scl = image[:,:,3]
            data_mask = image[:,:,4]

            water_mask = (
                (scl == 6) #water
                & (data_mask == 1) 
            )
            masked_ndwi = np.where(water_mask, ndwi, np.nan)

            water_ndwi = ndwi[water_mask]
            if water_ndwi.size ==0:
                print(f"No valid water for {period_start}")
                continue

            mean_ndwi = np.nanmean(water_ndwi)

            water_percentage=(np.sum(water_mask)/water_mask.size)*100

            results.append({
                "image": masked_ndwi,
                "date": period_start,
                "value": mean_ndwi,
                "percentage": water_percentage
            })
            
            print(f"Downloaded water for {period_start}")


        except Exception as e:
            print(f"Faild for {period_start}: {e}")

    return {"option": "NDWI", "results": results}


def download_rgb_data(config, collection, bbox, start_date, end_date):
    results =[]

    periods = create_periods(start_date=start_date, end_date=end_date)

    for period_start, period_end in periods:
        try:
            image =download_bands(
                config=config,
                collection=collection,
                bbox=bbox,
                time_interval=(period_start, period_end),
                bands=["B04", "B03", "B02"],
                mosaicking_order="leastCC",
                #maxcc=0.2,
                sample_type="FLOAT32"
            )

            results.append({
                "image": image,
                "date": period_start
            })
            print(f"Downloaded image for {period_start}")
        except Exception as e:
            print(f"Failed for {period_start}: {e}")

    return {"option": "RGB", "results": results}


DOWNLOAD_OPTIONS={
    "RGB": download_rgb_data,
    "NDVI": download_ndvi_data,
    "NDWI": download_ndwi_data,
}

def download_data(config, bbox, start_date, end_date, option):
    function = DOWNLOAD_OPTIONS.get(option)
    if function is None:
        print(f"Unkown download option {option}")
        return 

    collection_function = COLLECTIONS.get(option)
    collection = collection_function(config=config)
    
    return function(config=config, collection=collection, bbox=bbox, start_date=start_date, end_date=end_date)