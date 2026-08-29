import math
import pandas as pd
import numpy as np
from utils import backtest_model, evaluate_backtest, predict_persistence, predict_seasonal_naive


SIGMA = 10.0
ALPHA = 0.05
BETA = 10.0
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
            "value": item["value"],
            "percentage": item.get("percentage")
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


def predict_month_from_differences(df, target_month, value_column="value", sigma=2.0):
    differences = get_month_differences(df, target_month=target_month, value_column=value_column)

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

        delta = predict_month_from_differences(df=df, target_month=target_month, value_column="value", sigma=sigma)

        if delta is None:
            value_prediction = None
        else:
            last_real_value = df.loc[df["date"] == last_real_date, "value"].iloc[0]
            value_prediction = last_real_value+delta

        percentage_prediction = None

        if ("percentage" in df.columns and df["percentage"].notna().any()):
            percentage_delta=predict_month_from_differences(df=df, target_month=target_month, value_column="percentage", sigma=sigma)

            if percentage_delta is not None:
                last_real_percentage = df.loc[df["date"]==last_real_date, "percentage"].iloc[0]

                if pd.notna(last_real_percentage):
                    percentage_prediction = (last_real_percentage +percentage_delta)

                    percentage_prediction = np.clip(percentage_prediction, 0, 100)

        predictions.append({
            "date": next_date,
            "value": value_prediction,
            "percentage": percentage_prediction
        })

        current_date = next_date

    return predictions


def predict_month_improved(df, target_date, value_column="value", alpha=0.5, beta=2.0):
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



def predict(results, n):
    df = results_to_dataFrame(results=results)
    cc = complete_calendar(df)
    acf = add_calendar_features(cc)

    # find_best_prediction_model(df=acf, horizon=HORRIZON)

    return predict_next_months_improved(df=acf, n=n, alpha=ALPHA, beta=BETA)






def find_best_sigma(df, sigmas, horizon=6):
    best_sigma=None
    best_metrics=None
    best_score=float("inf")

    for sigma in sigmas:
        backtest = backtest_model(df=df, predict_function=predict_next_months, horizon=horizon, sigma=sigma)
        metrics = evaluate_backtest(backtest)
        score=metrics["score"]

        if score is None:
            continue

        print(f"sigma={sigma} "
              f"score={score}")

        if score<best_score:
            best_score=score
            best_sigma=sigma
            best_metrics=metrics

    return {
        "sigma": best_sigma,
        "metrics": best_metrics
        }

def find_best_alpha_beta(df, alphas, betas, horizon=6):
    best_alpha=None
    best_beta=None
    best_metrics=None
    best_score=float("inf")

    for alpha in alphas:
        for beta in betas:
            backtest = backtest_model(df=df, predict_function=predict_next_months_improved, horizon=horizon, alpha=alpha, beta=beta)
            metrics=evaluate_backtest(backtest)
            score=metrics["score"]

            if score is None:
                continue

            print(f"alpha={alpha} "
                  f"beta={beta} "
                  f"score={score}")

            if score < best_score:
                best_score=score
                best_alpha=alpha
                best_beta=beta
                best_metrics=metrics

    return {
        "alpha": best_alpha,
        "beta": best_beta,
        "metrics": best_metrics
    }

def find_best_prediction_model(df, horizon=6):
    sigmas=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0, 12.0, 15.0, 20.0]
    alphas=[0.05, 0.1, 0.15, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    betas=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,12.0,15.0,20.0]

    result_persistence =backtest_model(df=df, predict_function=predict_persistence, horizon=horizon)
    metrics_persistence = evaluate_backtest(results=result_persistence)

    result_seasonal = backtest_model(df=df, predict_function=predict_seasonal_naive, horizon=horizon)
    metrics_seasonal=evaluate_backtest(results=result_seasonal)

    best_weighted=find_best_sigma(df=df, sigmas=sigmas, horizon=horizon)

    best_improved=find_best_alpha_beta(df=df, alphas=alphas, betas=betas, horizon=horizon)

    models= {
        "Persistence": metrics_persistence,
        "Seasonal Naive": metrics_seasonal,
        "Weighted": best_weighted["metrics"],
        "Improved": best_improved["metrics"]
    }

    for name, metrics in models.items():
        print(f"{name}: "
              f"score={metrics['score']}")