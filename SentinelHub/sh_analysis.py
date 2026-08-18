from datetime import date
from SentinelHub.sh_download import download_bands
from SentinelHub.sh_indices import calculate_ndvi

import numpy as np

import matplotlib.pyplot as plt

class VegetationAnalysis:

    def __init__(self, bbox, start_date, end_date, config, collection):
        self.bbox = bbox
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        self.collection = collection
        self.results = []

    def create_periods(self):
        periods = []

        current = self.start_date

        while current < self.end_date:
            if current.month==12:
                next_date = date(current.year+1,1,1)
            else:
                next_date = date(current.year, current.month+1,1)

            if next_date>self.end_date:
                next_date=self.end_date

            periods.append((current, next_date))

            current=next_date

        self.periods = periods

    def download_observation(self, period):
        image = download_bands(
            config=self.config,
            collection=self.collection,
            bbox=self.bbox,
            time_interval=period,
            bands=["B04","B08"],
            clm= True,
            scl= True,
            data_mask=True,
            maxcc=0.2,
            mosaicking_order="leastCC",
            sample_type="FLOAT32"
        )

        return image

    def calculate_mean_ndvi(self, ndvi):
        return np.nanmean(ndvi)
        

    def run(self):
        self.create_periods()

        for period_start, period_end in self.periods:
            try:
                image = self.download_observation(period=(period_start, period_end))
                ndvi = calculate_ndvi(image)

                scl = image[:,:,3]
                data_mask = image[:,:,4]

                vegetation_mask=( 
                    (scl == 4) #vegetatie
                    & (data_mask == 1) #exista valori 
                    )

                masked_ndvi = np.where(vegetation_mask, ndvi, np.nan)

                vegetation_ndvi = ndvi[vegetation_mask]
                if vegetation_ndvi.size == 0:
                    print(f"No valid vegetation for {period_start}")
                    continue

                # plt.figure(figsize=(8,6))
                # plt.imshow(masked_ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
                # plt.colorbar(label="NDVI")
                # plt.title("NDVI")
                # plt.show()


                mean_ndvi = self.calculate_mean_ndvi(vegetation_ndvi)

                self.results.append({
                    "date": period_start,
                    "ndvi": mean_ndvi
                    })
            except Exception as e:
                print(f"Faild for {period_start}: {e}")

    def graph_ndvi(self):
        dates=[result["date"] for result in self.results]
        ndvi=[result["ndvi"] for result in self.results]

        plt.figure(figsize=(10, 5))

        plt.plot(dates, ndvi, marker="o")

        plt.xlabel("Date")
        plt.ylabel("Mean NDVI")
        plt.title("NDVI evolution")

        plt.ylim(-1,1)

        plt.grid(True)
        # plt.xticks(rotation=45)
        plt.xticks(dates, rotation=45)

        plt.tight_layout()
        plt.show()