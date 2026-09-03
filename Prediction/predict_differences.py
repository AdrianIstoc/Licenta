import numpy as np
import pandas as pd
import math


SIGMA = 4.0



def month_distance(m1, m2):
    diff = abs(m1-m2)
    return  min(diff, 12 - diff)


def month_weight(distance, sigma=2.0):
    return math.exp(-(distance**2)/(2*sigma**2))


def get_month_differences(df, target_month, value_column="value"):
    results = []

    for year, year_df in df.groupby("year"):
        target = year_df.loc[year_df["month"]==target_month, value_column]

        if target.empty or pd.isna(target.iloc[0]):
            continue

        target_value = target.iloc[0]

        for _, row in year_df.iterrows():
            value = row[value_column]
            if pd.isna(value):
                continue
            if row["month"]==target_month:
                continue

            results.append({
                "year": year,
                "month": int(row["month"]),
                "target_value": target_value,
                "other_value": value,
                "difference": target_value-value,
                "distance": month_distance(m1=target_month, m2=int(row["month"]))
            })

    return pd.DataFrame(results)

def predict_month_from_differences(df, target_month, value_column="value", sigma=SIGMA):
    differences = get_month_differences(df, target_month=target_month, value_column=value_column)

    if differences.empty:
        return None

    differences["weight"]=differences["distance"].apply(lambda d: month_weight(d,sigma=sigma))
    

    weighted_difference=(
        (differences["difference"]*differences["weight"]).sum()
        /differences["weight"].sum()
    )

    return weighted_difference

def predict_next_months_differences(df, n, sigma=SIGMA):
    df = df.copy()

    last_real_date = df.loc[df["value"].notna(),"date"].max()

    predictions=[]

    current_date = last_real_date

    for _ in range(n):
        next_date = current_date+pd.DateOffset(months=1)
        target_month = next_date.month

        delta = predict_month_from_differences(df=df, target_month=target_month, value_column="value", sigma=sigma)

        if delta is None:
            value_prediction = None
        else:
            last_real_value = df.loc[df["date"] == last_real_date, "value"].iloc[0]
            value_prediction = last_real_value+delta

        percentage_prediction = None

        if ("percentage" in df.columns and df["percentage"].notna().any()):
            percentage_delta=predict_month_from_differences(df=df, target_month=target_month, value_column="percentage", sigma=SIGMA)

            if percentage_delta is not None:
                last_real_percentage = df.loc[df["date"]==last_real_date, "percentage"].iloc[0]

                if pd.notna(last_real_percentage):
                    percentage_prediction = (last_real_percentage +percentage_delta)

                    percentage_prediction = np.clip(percentage_prediction, 0, 100)

        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "value": value_prediction,
            "percentage": percentage_prediction
        })

        current_date = next_date

    return predictions

