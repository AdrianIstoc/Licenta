"""
Utility functions used by example notebooks
"""

from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_rgb(image, ax):
    rgb = np.clip(image * 3.5, 0, 1)

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

PLOT_FUNCTIONS={
    "RGB": plot_rgb,
    "NDVI": plot_ndvi
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

    ax1.plot(dates, indice, marker= "o", label=indice_name)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(indice_name)
    ax1.set_ylim(-1,1)
    ax1.grid(True)
    ax1.set_xticks(dates)
    ax1.set_xticklabels(dates, rotation=45)
    

    if percentage is not None:
        ax2 = ax1.twinx()

        ax2.plot(dates, percentage, marker="o", color="yellow", label="Percentage")
        ax2.set_ylabel("%")
        ax2.set_ylim(0,100)

    plt.title(indice_name)
    # plt.xticks(dates, rotation=45)

    fig.tight_layout()
    plt.show()