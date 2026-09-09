from Prediction.predict_differences import predict_next_months_differences
from Prediction.predict_improved import predict_next_months_improved
from Prediction.predict_patchTST import predict_patchtst
from Prediction.test_predict import find_best_prediction_model, find_best_patchtst
import pandas as pd


PREDICTION_OPTION={
    "DIFF": predict_next_months_differences,
    "IMP": predict_next_months_improved,
    "PATCHTST": predict_patchtst,
}



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




def predict(results, n, option):
    df = results_to_dataFrame(results=results)
    cc = complete_calendar(df)
    acf = add_calendar_features(cc)

    # find_best_prediction_model(df=acf)

    function = PREDICTION_OPTION.get(option)

    if function is None:
        print(f"Unknown predict option {option}")
        return

    
    return function(df=acf, n=n)







