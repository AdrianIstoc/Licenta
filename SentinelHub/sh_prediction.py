import math
import pandas as pd
import numpy as np

def month_distance(m1, m2):
    diff = abs(m1-m2)
    return  min(diff, 12 - diff)

def month_weight(distance, sigma=2.0):
    return math.exp(-(distance**2)/(2*sigma**2))

def get_weights(month_wanted, sigma=2.0):
    weights = {}
    for month in range(1,13):
        distance = month_distance(month_wanted, month)
        weights[month] = month_weight(distance=distance, sigma=sigma)

    total = sum(weights.values())

    for month in weights:
        weights[month] /= total

    return weights

def results_to_dataFrame(results):
    df = pd.DataFrame([
        {
            "date": item["date"],
            "value": item["value"]
        }
        for item in results
    ])

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    return df

def complete_calendar(df):
    df = df.copy()

    df= df.set_index("date")

    df= df.asfreq("MS")

    df= df.reset_index()

    return df

def add_calendar_features(df):
    df = df.copy()

    df["year"]=df["date"].dt.year
    df["month"]=df["date"].dt.month

    return df

def get_month_history(df, month):
    history = df[
        (df["month"]==month) &
        (df["value"].notna())
    ].copy()

    return history



def predict_month(df, target_month, sigma=2.0):
    weights = get_weights(target_month, sigma=sigma)

    weighted_sum=0.0
    weight_sum=0.0

    for month, weight in weights.items():
        history = get_month_history(df, month=month)
        if history.empty:
            continue

        value_mean = history["value"].mean()

        weighted_sum += value_mean*weight
        weight_sum +=weight

    if weight_sum==0:
        return None

    return weighted_sum/weight_sum

def get_month_differences(df, target_month):
    results = []

    for year, year_df in df.groupby("year"):
        target = year_df.loc[year_df["month"]==target_month, "value"]

        if target.empty or pd.isna(target.iloc[0]):
            continue

        target_value = target.iloc[0]

        for _, row in year_df.iterrows():
            if pd.isna(row["value"]):
                continue
            if row["month"]==target_month:
                continue

            results.append({
                "year": year,
                "month": int(row["month"]),
                "target_value": target_value,
                "other_value": row["value"],
                "difference": target_value-row["value"],
                "distance": month_distance(m1=target_month, m2=int(row["month"]))
            })

    return pd.DataFrame(results)

def predict_month_from_differences(df, target_month, sigma=2.0):
    differences = get_month_differences(df, target_month=target_month)

    if differences.empty:
        return None

    differences["weight"]=differences["distance"].apply(lambda d: month_weight(d,sigma=sigma))
    

    weighted_difference=(
        (differences["difference"]*differences["weight"]).sum()
        /differences["weight"].sum()
    )

    return weighted_difference


def predict_next_months(df, n, sigma=2.0):
    df = df.copy()

    last_real_date = df.loc[df["value"].notna(),"date"].max()

    predictions=[]

    current_value=df.loc[df["date"]==last_real_date, "value"].iloc[0]
    current_date = last_real_date

    for _ in range(n):
        next_date = current_date+pd.DateOffset(months=1)
        target_month = next_date.month

        delta = predict_month_from_differences(df, target_month, sigma)

        if delta is None:
            prediction = current_value
        else:
            prediction = current_value+delta

        predictions.append({
            "date": next_date,
            "value": prediction
        })

        current_date = next_date
        current_value = prediction

    return predictions


def prdct(results):
    df = results_to_dataFrame(results=results)
    cc = complete_calendar(df)
    acf = add_calendar_features(cc)

    # sigmas=[
    #     0.5,
    #     1.0,
    #     1.5,
    #     2.0,
    #     2.5,
    #     3.0,
    #     3.5,
    #     4.0,
    #     4.5,
    #     5.0
    # ]
    # for sigma in sigmas:
    #     bt = backtest(df=acf, horizon=6, sigma=sigma)
    #     metrics=evaluate_backtest(bt)
    #     print(
    #         f"sigma={sigma:.1f} "
    #         f"MAE={metrics['MAE']:.5f} "
    #         f"RMSE={metrics['RMSE']:.5f}"
    #     )


    # bt = backtest(df=acf, horizon=6, sigma=2.5)
    # metrics=evaluate_backtest(bt)
    # print(
    #     f"MAE={metrics['MAE']:.5f} "
    #     f"RMSE={metrics['RMSE']:.5f}"
    # )


    result_model = backtest_model(acf, predict_next_months, horizon=6, sigma=2.5)
    metrics_model = evaluate_backtest_2(result_model)

    result_persistance = backtest_model(acf, predict_persistence, horizon=6)
    metrics_persistance = evaluate_backtest_2(result_persistance)

    result_seasonal = backtest_model(acf, predict_seasonal_naive, horizon=6)
    metrics_seasonal = evaluate_backtest_2(result_seasonal)

    print("Persistance:")
    print(metrics_persistance)
    print()
    print("Seasonal naive:")
    print(metrics_seasonal)
    print()
    print("Weighted model:")
    print(metrics_model)


    return predict_next_months(df=acf, n=24, sigma=2.5)












def backtest(df, horizon=6, step=1, sigma=2.0):
    df = df.sort_values("date").copy()

    real_dates= df.loc[df["value"].notna(), "date"].sort_values().unique()

    results=[]

    for i in range(0, len(real_dates)-horizon, step):
        cutoff_date = real_dates[i]
        train_df = df[df["date"] <= cutoff_date].copy()

        future_dates=real_dates[
            (real_dates>cutoff_date)
        ][:horizon]

        if len(future_dates)<horizon:
            break

        predictions = predict_next_months(df=train_df, n=horizon, sigma=sigma)

        prediction_df = pd.DataFrame(predictions)

        actual_df = df[df["date"].isin(future_dates)][["date","value"]].copy()

        merged = actual_df.merge(
            prediction_df,
            on="date",
            suffixes=("_actual", "_predicted")
        )

        merged["error"] = (
            merged["value_predicted"]
            -merged["value_actual"]
        )

        merged["absolute_error"]=merged["error"].abs()

        results.append(merged)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)

def evaluate_backtest(results):
    mae = results["absolute_error"].mean()

    rmse= (results["error"]**2).mean()**0.5

    return {
        "MAE": mae,
        "RMSE": rmse
    }


def predict_persistence(df, n):
    df= df.copy()

    valid = df[df["value"].notna()]

    if valid.empty:
        return None

    last_real_date = valid["date"].max()
    current_value=valid.loc[valid["date"]==last_real_date, "value"].iloc[0]

    predictions=[]
    current_date=last_real_date

    for _ in range(n):
        next_date = current_date+pd.DateOffset(months=1)

        predictions.append({
            "date": next_date,
            "value": current_value
        })

        current_date=next_date

    return predictions

def predict_seasonal_naive(df, n):
    df = df.copy()

    valid = df[df["value"].notna()].copy()

    if valid.empty:
        return None
    last_real_date = valid["date"].max()

    predictions=[]

    current_date=last_real_date

    for _ in range(n):
        next_date = current_date+pd.DateOffset(months=1)
        target_month = next_date.month

        history = valid[valid["date"].dt.month == target_month]

        if history.empty:
            prediction = None
        else:
            prediction = history.sort_values("date")["value"].iloc[-1]

        predictions.append({
            "date": next_date,
            "value": prediction
        })

        current_date=next_date

    return predictions


def backtest_model(df, predict_function, horizon=6, **kwargs):
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    valid_indices=df.index[df["value"].notna()].tolist()

    results = []

    for end_idx in valid_indices:
        train = df.iloc[:end_idx+1].copy()

        if len(train) < 6:
            continue
        predictions=predict_function(train, n=horizon, **kwargs)

        if predictions is None:
            continue

        for prediction in predictions:
            prediction_date = prediction["date"]

            actual = df.loc[df["date"]==prediction_date, "value"]

            if actual.empty:
                continue

            actual_value = actual.iloc[0]

            if pd.isna(actual_value):
                continue

            if prediction["value"] is None:
                continue

            results.append({
                "date": prediction_date,
                "predicted": prediction["value"],
                "actual": actual_value
            })

    return pd.DataFrame(results)

def evaluate_backtest_2(results):
    if results.empty:
        return{
            "MAE": None,
            "RMSE": None
        }

    errors = (results["predicted"] - results["actual"])

    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())

    return {
        "MAE": mae,
        "RMSE": rmse
    }