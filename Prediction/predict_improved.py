import numpy as np
import pandas as pd

ALPHA = 0.05
BETA = 10.0


def predict_month_improved(df, target_date, value_column="value", alpha=ALPHA, beta=BETA):
    df = df.copy()

    target_month = target_date.month

    valid = df[df[value_column].notna()].copy()

    if valid.empty:
        return None

    same_month = valid[valid["date"].dt.month == target_month].copy()

    if same_month.empty:
        return None
    target_year = target_date.year

    same_month["year_distance"]= (target_year-same_month["date"].dt.year)

    same_month = same_month[same_month["year_distance"]>0]

    if same_month.empty:
        return None

    same_month["year_weight"]=np.exp(-same_month["year_distance"]/beta)

    if same_month.empty:
        return None

    seasonal_value=(same_month[value_column]*same_month["year_weight"]).sum()/same_month["year_weight"].sum()

    if len(same_month)>= 2:
        same_month=same_month.sort_values("date")
        differences=same_month[value_column].diff().dropna()

        trend = differences.mean()
    else:
        trend=0.0

    prediction = seasonal_value + alpha*trend

    return prediction

def predict_next_months_improved(df, n, alpha=ALPHA, beta=BETA):
    df = df.copy()
    df = df.sort_values("date")

    valid = df[df["value"].notna()]

    if valid.empty:
        return None

    last_real_date=valid["date"].max()

    predictions=[]

    for i in range(1, n+1):
        target_date=last_real_date+pd.DateOffset(months=i)
        value_prediction=predict_month_improved(df=df, target_date=target_date, value_column="value", alpha=alpha, beta=beta)
        percentage_prediction=predict_month_improved(df=df, target_date=target_date, value_column="percentage", alpha=alpha, beta=beta)

        if percentage_prediction is not None:
            percentage_prediction = np.clip(percentage_prediction, 0, 100)

        predictions.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "value": value_prediction,
            "percentage": percentage_prediction
        })

    return predictions