import math
import pandas as pd
import numpy as np
from utils import backtest_model, evaluate_backtest_2, predict_persistence, predict_seasonal_naive


SIGMA = 5.0
ALPHA = 0.1
BETA = 4.5
HORRIZON = 12



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

    current_date = last_real_date

    for _ in range(n):
        next_date = current_date+pd.DateOffset(months=1)
        target_month = next_date.month

        delta = predict_month_from_differences(df, target_month, sigma)

        if delta is None:
            prediction = None
        else:
            last_real_value = df.loc[df["date"] == last_real_date, "value"].iloc[0]
            prediction = last_real_value+delta

        predictions.append({
            "date": next_date,
            "value": prediction
        })

        current_date = next_date

    return predictions


def predict_month_improved(df, target_date, alpha=0.5, beta=2.0):
    df = df.copy()

    target_month = target_date.month

    valid = df[df["value"].notna()].copy()

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

    seasonal_value=(same_month["value"]*same_month["year_weight"]).sum()/same_month["year_weight"].sum()

    if len(same_month)>= 2:
        same_month=same_month.sort_values("date")
        differences=same_month["value"].diff().dropna()

        trend = differences.mean()
    else:
        trend=0.0

    prediction = seasonal_value + alpha*trend

    return prediction

def predict_next_months_improved(df, n, alpha=0.5, beta=2.0):
    df = df.copy()
    df = df.sort_values("date")

    valid = df[df["value"].notna()]

    if valid.empty:
        return None

    last_real_date=valid["date"].max()

    predictions=[]

    for i in range(1, n+1):
        target_date=last_real_date+pd.DateOffset(months=i)
        prediction=predict_month_improved(df=df, target_date=target_date, alpha=alpha, beta=beta)

        predictions.append({
            "date": target_date, #.strftime("%Y-%m-%d"),
            "value": prediction
        })

    return predictions



def prdct(results, n):
    df = results_to_dataFrame(results=results)
    cc = complete_calendar(df)
    acf = add_calendar_features(cc)

    # sigma_values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # sigma_results=[]

    # for sigma in sigma_values:
    #     results_sigma = backtest_model(acf, predict_next_months, horizon=HORRIZON, sigma=sigma)
    #     metrics_sigma=evaluate_backtest_2(results_sigma)
    #     print(
    #         f"sigma={sigma}, "
    #         f"rows={len(results_sigma)}, "
    #         f"MAE={metrics_sigma['MAE']}, "
    #         f"RMSE={metrics_sigma['RMSE']}"
    #     )
    #     sigma_results.append({
    #         "sigma": sigma,
    #         "MAE": metrics_sigma["MAE"],
    #         "RMSE": metrics_sigma["RMSE"]
    #     })

    # print()
    # print("sigma results:")
    # print(sigma_results)

    # if not sigma_results:
    #     print("NU S-A PUTUT DETERMINA UN sigma!")
    #     return None

    # best_sigma_results = min(sigma_results,key=lambda x:x["MAE"])

    # best_sigma = best_sigma_results["sigma"]

    # print()
    # print("Best sigma:")
    # print(best_sigma)
    # print("MAE:", best_sigma_results["MAE"])
    # print("RMSE:", best_sigma_results["RMSE"])




    # alpha_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # beta_values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    # both_results=[]

    # for alpha in alpha_values:
    #     for beta in beta_values:
    #         results_both = backtest_model(acf, predict_next_months_improved, horizon=HORRIZON, alpha=alpha, beta=beta)
    #         metrics_both=evaluate_backtest_2(results_both)
    #         print(
    #             f"alpha={alpha}, "
    #             f"beta={beta}, "
    #             f"rows={len(results_both)}, "
    #             f"MAE={metrics_both['MAE']}, "
    #             f"RMSE={metrics_both['RMSE']}"
    #         )
    #         both_results.append({
    #             "alpha": alpha,
    #             "beta": beta,
    #             "MAE": metrics_both["MAE"],
    #             "RMSE": metrics_both["RMSE"]
    #         })

    # print()
    # print("Both results:")
    # print(both_results)

    # if not both_results:
    #     print("NU S-A PUTUT DETERMINA O COMBINATIE alpha/beta!")
    #     return None

    # best_both = min(both_results,key=lambda x:x["MAE"])

    # best_alpha = best_both["alpha"]
    # best_beta = best_both["beta"]

    # print()
    # print("Best combination:")
    # print("Alpha:", best_alpha)
    # print("Beta:", best_beta)
    # print("MAE:", best_both["MAE"])
    # print("RMSE:", best_both["RMSE"])


    result_model = backtest_model(acf, predict_next_months, horizon=6, sigma=SIGMA)
    metrics_model = evaluate_backtest_2(result_model)

    result_persistance = backtest_model(acf, predict_persistence, horizon=6)
    metrics_persistance = evaluate_backtest_2(result_persistance)

    result_seasonal = backtest_model(acf, predict_seasonal_naive, horizon=6)
    metrics_seasonal = evaluate_backtest_2(result_seasonal)

    results_new_model = backtest_model(acf, predict_next_months_improved, horizon=6, alpha=ALPHA, beta=BETA)
    metrics_new=evaluate_backtest_2(results_new_model)

    print("Persistance:")
    print(metrics_persistance)
    print()
    print("Seasonal naive:")
    print(metrics_seasonal)
    print()
    print("Weighted model:")
    print(metrics_model)
    print()
    print("New model:")
    print(metrics_new)

    # return predict_next_months(df=acf, n=n, sigma=SIGMA)
    return predict_next_months_improved(df=acf, n=n, alpha=ALPHA, beta=BETA)