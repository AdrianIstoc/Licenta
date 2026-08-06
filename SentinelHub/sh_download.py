from sentinelhub import (
    SentinelHubRequest,
    MimeType,
    bbox_to_dimensions
)

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
        input_param["mosaiking_order"] = mosaicking_order

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
    # print (evalscript)
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