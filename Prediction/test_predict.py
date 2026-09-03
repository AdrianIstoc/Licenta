from Prediction.predict_improved import predict_next_months_improved, ALPHA, BETA
from Prediction.predict_differences import predict_next_months_differences, SIGMA
from Prediction.predict_patchTST import predict_patchtst
from itertools import product

import pandas as pd
import numpy as np

def log(text):
    with open("prediction_results.txt", "a", encoding="utf-8") as f:
        f.write(str(text)+"\n")


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




def find_best_sigma(df, sigmas=(SIGMA,), horizon=6):
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

def find_best_alpha_beta(df, alphas=(ALPHA,), betas=(BETA,), horizon=6):
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


def find_best_patchtst(
        df,
        horizon=6,
        contexts=(42, 48, 54),
        patch_lengths=(6, 9, 12),
        patch_strides=(3,),
        learning_rates=(0.0002,),
        epochs=30,
        batch_size=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
):
    configurations=list(product(
        contexts,
        patch_lengths,
        patch_strides,
        learning_rates
    ))
    results = []

    log(f"Testing {len(configurations)} PatchTST configurations...")

    for index, (context_length, patch_length, patch_stride, learning_rate) in enumerate(configurations, start=1):
        if patch_length>context_length:
            continue
        if patch_stride>context_length:
            continue

        log(
            f"\n[{index}/{len(configurations)}] "
            f"context={context_length}, "
            f"pathc={patch_length}, "
            f"stride={patch_stride}, "
            f"lr={learning_rate}"
        )

        def predict_function(train_df, n):
            return predict_patchtst(df=train_df, n=n, context_length=context_length, patch_length=patch_length, patch_stride=patch_stride, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,dropout=dropout)

        backtest_results= backtest_model(df=df, predict_function=predict_function, horizon=horizon)
        if backtest_results.empty:
            log("No backtest results.")
            continue

        metrics = evaluate_backtest(backtest_results)
        score=metrics["score"]

        if score is None:
            log("Score=None")
            continue

        results.append({
            "context_length": context_length,
            "patch_length": patch_length,
            "patch_stride": patch_stride,
            "learning_rate": learning_rate,
            "value_mae": metrics["value"]["MAE"],
            "value_rmse": metrics["value"]["RMSE"],
            "percentage_mae": metrics["percentage"]["MAE"],
            "percentage_rmse": metrics["percentage"]["RMSE"],
            "score": score
        })

        log(f"Score: {score}")

    results_df= pd.DataFrame(results)

    if results_df.empty:
        return results_df

    results_df =results_df.sort_values("score", ascending=True).reset_index(drop=True)

    log("\n==== BEST PATCHTST CONFIGURATIONS =====")

    for _, row in results_df.head(5).iterrows():
        log(
            f"context={row['context_length']}, "
            f"patch={row['patch_length']}, "
            f"stride={row['patch_stride']}, "
            f"lf={row['learning_rate']}, "
            f"score={row['score']}"
        )

    return results_df

def find_best_prediction_model(df, horizon=6):
    # sigmas=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0, 12.0, 15.0, 20.0]
    # alphas=[0.05, 0.1, 0.15, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # betas=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,12.0,15.0,20.0]

    contexts=(42, 48, 54, 60, 66)
    patch_lengths=(6, 9, 12, 15)
    patch_strides=(3, 6)
    learning_rates=(0.0001, 0.00015, 0.0002, 0.00025, 0.0003)



    result_persistence =backtest_model(df=df, predict_function=predict_persistence, horizon=horizon)
    metrics_persistence = evaluate_backtest(results=result_persistence)

    result_seasonal = backtest_model(df=df, predict_function=predict_seasonal_naive, horizon=horizon)
    metrics_seasonal=evaluate_backtest(results=result_seasonal)

    best_weighted=find_best_sigma(df=df, horizon=horizon) # sigmas=sigmas, horizon=horizon)

    best_improved=find_best_alpha_beta(df=df, horizon=horizon) # alphas=alphas, betas=betas, horizon=horizon)


    best_patchtst = find_best_patchtst(df=df, horizon=horizon, contexts=contexts, patch_lengths=patch_lengths, patch_strides=patch_strides, learning_rates=learning_rates, epochs=200)

    models= {
        "Persistence": metrics_persistence,
        "Seasonal Naive": metrics_seasonal,
        "Weighted": best_weighted["metrics"],
        "Improved": best_improved["metrics"]
    }

    if not best_patchtst.empty:
        patchtst_best = best_patchtst.iloc[0]
        models["PatchTST"] ={
            "value": {
                "MAE": patchtst_best["value_mae"],
                "RMSE": patchtst_best["value_rmse"]
            },
            "percentage": {
                "MAE": patchtst_best["percentage_mae"],
                "RMSE": patchtst_best["percentage_rmse"]
            },
            "score": patchtst_best["score"]
        }

    for name, metrics in models.items():
        if metrics is None:
            log(f"{name}: metrics=None")
            continue

        if metrics["score"] is None:
            log(f"{name}: score=None")
            continue
        
        log(f"{name}: "
            f"score={metrics['score']}")



