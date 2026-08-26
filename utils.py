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


    rgb = np.clip(rgb, 0, 1) #image * 3.5, 0, 1)

    for channel, name in enumerate(["r", "g", "b"]):
        values = image[:,:,channel]

        print(
            name,
            "min =", np.nanmin(values),
            "p2 =", np.nanpercentile(values, 2),
            "p98 =", np.nanpercentile(values, 98),
            "max = ", np.nanmax(values),
            "NaN =", np.isnan(values).sum()
        )

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

        last_train_date = train["date"].max()


        predictions=predict_function(train, n=horizon, **kwargs)

        if predictions is None:
            continue


        for prediction in predictions:
            prediction_date = prediction["date"]
            prediction_value = prediction["value"]


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

def evaluate_backtest(results):
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