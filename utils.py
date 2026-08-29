import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rgb(image, ax):
    rgb = image.astype(np.float32).copy()

    for channel in range(3):
        low = np.nanpercentile(rgb[:,:, channel],2)
        high = np.nanpercentile(rgb[:,:,channel],98)

        if np.isfinite(low) and np.isfinite(high) and high>low:
            rgb[:,:,channel]=(rgb[:,:,channel]-low) / (high-low) 
        else:
            rgb[:,:,channel]=0


    rgb = np.clip(rgb, 0, 1)
    ax.imshow(rgb)
    ax.set_title("RGB")
    ax.axis("off")

def plot_ndvi(image, ax):
    im = ax.imshow(
        image,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1
    )

    ax.set_title("NDVI")
    ax.axis("off")

def plot_ndwi(image, ax):
    im = ax.imshow(
        image,
        cmap="RdYlBu",
        vmin=-1,
        vmax=1
    )

    ax.set_title("NDWI")
    ax.axis("off")

PLOT_FUNCTIONS={
    "RGB": plot_rgb,
    "NDVI": plot_ndvi,
    "NDWI": plot_ndwi,
}
def plot_image(data):
    if data is None:
        return

    option= data["option"]
    results= data["results"]

    plot_function = PLOT_FUNCTIONS.get(option)

    if plot_function is None:
        print(f"Unknown option: {option}")
        return

    if not results:
        print("No images to display.")
        return

    current_index = 0
    fig, ax = plt.subplots(figsize=(10,7))

    def show_image(index):
        ax.clear()

        result = results[index]

        image = result["image"]
        date= result["date"]

        plot_function(image, ax)

        ax.set_title(
            f"{option} - {date}"
            f"({index+1}/{len(results)})"
        )
        fig.canvas.draw_idle()

    def next_image(event):
        nonlocal current_index

        if current_index < len(results) - 1:
            current_index += 1
            show_image(current_index)

    def previous_image(event):
        nonlocal current_index

        if current_index > 0:
            current_index -= 1
            show_image(current_index)

    show_image(current_index)

    previous_button = plt.axes([0.25, 0.03, 0.15, 0.05])
    next_button = plt.axes([0.60, 0.03, 0.15, 0.05])

    button_previous = plt.Button(previous_button, "Previous")
    button_next = plt.Button(next_button, "Next")

    button_previous.on_clicked(previous_image)
    button_next.on_clicked(next_image)

    plt.show() 

def graph_data(dates, indice, indice_name, percentage=None):
    fig, ax1 = plt.subplots(figsize=(10,5))

    line1, = ax1.plot(
        dates,
        indice,
        marker="o",
        color="blue",
        label=indice_name
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel(indice_name)
    ax1.set_ylim(-1,1)
    ax1.grid(True)
    ax1.set_xticks(dates)
    ax1.set_xticklabels(dates, rotation=90)

    lines = [line1]

    if percentage is not None:
        ax2 = ax1.twinx()   
        line2, = ax2.plot(
            dates,
            percentage,
            marker="o",
            color="yellow",
            label=f"{indice_name} percentage"
        )

        lines.append(line2)


        ax2.set_ylabel("Percentage (%)")
        ax2.set_ylim(0,100)

    ax1.legend(handles=lines, loc="best")

    plt.title(indice_name)
    fig.tight_layout()
    plt.show()










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
        scores.append(normalized_value_mae)

    if percentage_mae is not None:
        normalized_percentage_mae=percentage_mae/100.0
        scores.append(normalized_percentage_mae)

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

