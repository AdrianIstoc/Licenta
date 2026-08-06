from sentinelhub import DataCollection

def get_sentinel2_l2a(config):
    return DataCollection.SENTINEL2_L2A.define_from(
        "S2_L2A_CDSE",
        service_url=config.sh_base_url
    )