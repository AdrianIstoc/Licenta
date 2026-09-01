import math
import pandas as pd
import numpy as np


SIGMA = 4.0
ALPHA = 0.05
BETA = 10.0
HORRIZON = 12

def log(text):
    with open("prediction_results.txt", "a", encoding="utf-8") as f:
        f.write(str(text)+"\n")

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

    df["value"] = df["value"].interpolate(method="linear")

    if "percentage" in df.columns:
        df["percentage"] = df["percentage"].interpolate(method="linear")

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


PREDICTION_OPTION={
    "DIFF": predict_next_months_differences,
    "IMP": predict_next_months_improved,
}

def predict(results, n, option):
    df = results_to_dataFrame(results=results)
    cc = complete_calendar(df)
    acf = add_calendar_features(cc)

    # find_best_prediction_model(df=acf, horizon=HORRIZON)

    function = PREDICTION_OPTION.get(option)

    if function is None:
        print(f"Unknown predict option {option}")
        return

    
    return function(df=acf, n=n)









def predict_persistence(df, n):
    df= df.copy()

    valid = df[df["value"].notna()]

    if valid.empty:
        return None

    last_real_date = valid["date"].max()
    current_value=valid.loc[valid["date"]==last_real_date, "value"].iloc[0]

    current_percentage = None
    if "percentage" in df.columns:
        percentage=df.loc[df["date"] == last_real_date, "percentage"]

        if not percentage.empty and pd.notna(percentage.iloc[0]):
            current_percentage = percentage.iloc[0]

    predictions=[]
    current_date=last_real_date

    for _ in range(n):
        next_date = current_date+pd.DateOffset(months=1)

        predictions.append({
            "date": next_date,
            "value": current_value,
            "percentage": current_percentage
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

        history_value = valid[valid["date"].dt.month == target_month]

        if history_value.empty:
            value_prediction = None
        else:
            value_prediction = history_value.sort_values("date")["value"].iloc[-1]

        percentage_prediction = None

        if "percentage" in df.columns:
            history_percentage = df[(df["date"].dt.month == target_month) & (df["percentage"].notna())]

            if not history_percentage.empty:
                percentage_prediction=(history_percentage.sort_values("date")["percentage"].iloc[-1])
        

        predictions.append({
            "date": next_date,
            "value": value_prediction,
            "percentage": percentage_prediction
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
            actual = df.loc[df["date"]==prediction_date]

            if actual.empty:
                continue

            actual = actual.iloc[0]

            predicted_value=prediction.get("value")
            actual_value = actual["value"]

            predicted_percentage = prediction.get("percentage")
            if "percentage" in df.columns:
                actual_percentage=actual["percentage"]
            else:
                actual_percentage =None

            if predicted_value is None and predicted_percentage is None:
                continue


            results.append({
                "date": prediction_date,
                "predicted": predicted_value,
                "actual": actual_value,
                "predicted_percentage": predicted_percentage,
                "actual_percentage": actual_percentage
            })


    return pd.DataFrame(results)

def evaluate_backtest(results):
    if results.empty:
        return{
            "value": {
                "MAE": None,
                "RMSE": None
            },
            "percentage": {
                "MAE": None,
                "RMSE": None
            },
            "score": None
        }

    value_data=results[
        results["predicted"].notna() &
        results["actual"].notna()
    ]

    if value_data.empty:
        value_mae=None
        value_rmse=None
    else:
        value_errors=(value_data["predicted"]-value_data["actual"])
        value_mae=np.abs(value_errors).mean()
        value_rmse=np.sqrt((value_errors**2).mean())

    percentage_data=results[
        results["predicted_percentage"].notna() &
        results["actual_percentage"].notna()
    ]

    if percentage_data.empty:
        percentage_mae=None
        percentage_rmse=None
    else:
        percentage_errors =(percentage_data["predicted_percentage"] - percentage_data["actual_percentage"])
        percentage_mae=np.abs(percentage_errors).mean()
        percentage_rmse=np.sqrt((percentage_errors**2).mean())

    scores=[]

    if value_mae is not None:
        normalized_value_mae=value_mae/2.0
        normalized_value_rmse=value_rmse/2.0

        value_score=(0.5*normalized_value_mae + 0.5 * normalized_value_rmse)

        scores.append(value_score)

    if percentage_mae is not None:
        normalized_percentage_mae=percentage_mae/100.0
        normalized_percentage_rmse = percentage_rmse/100.0

        percentage_score=(0.5 * normalized_percentage_mae + 0.5*normalized_percentage_rmse)
        scores.append(percentage_score)

    if scores:
        score = np.mean(scores)
    else:
        score=None

    return {
        "value": {
            "MAE": value_mae,
            "RMSE": value_rmse
        },
        "percentage": {
            "MAE": percentage_mae,
            "RMSE": percentage_rmse
        },
        "score": score
    }



def find_best_sigma(df, sigmas, horizon=6):
    best_sigma=None
    best_metrics=None
    best_score=float("inf")

    for sigma in sigmas:
        backtest = backtest_model(df=df, predict_function=predict_next_months_differences, horizon=horizon, sigma=sigma)
        metrics = evaluate_backtest(backtest)
        score=metrics["score"]

        if score is None:
            continue

        log(f"sigma={sigma} "
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

            log(f"alpha={alpha} "
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
        log(f"{name}: "
              f"score={metrics['score']}")