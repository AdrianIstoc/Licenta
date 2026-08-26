import tkinter as tk
from tkintermapview import TkinterMapView
from tkcalendar import Calendar
from sentinelhub import (
    BBox,
    CRS
)
from SentinelHub.sh_collections import get_sentinel2_l2a
from SentinelHub.sh_download import (
    download_rgb_data,
    download_ndvi_data,
    download_ndwi_data
)
from utils import graph_data, plot_image
from SentinelHub.sh_prediction import predict

class Application(tk.Tk):
    def __init__(self, config):
        super().__init__()

        self.download_option = tk.StringVar(value="None")
        self.prediction_months=tk.IntVar(value=12)

        self.title("Environmental Monitoring")
        self.geometry("1280x720")
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.create_widgets()

        self.config = config
        self.collection = get_sentinel2_l2a(config)

        self.dragging_mouse = False
        self.selecting_area = False

        self.starting_point = None
        self.current_point = None

        self.polygon = None

        self.start_date = None
        self.end_date = None

        self.downloaded_data = None


    def create_widgets(self):
        self.toolbar = tk.Frame(self, height=30, bd=2, relief="raised")
        self.toolbar.grid(row=0,column=0,columnspan=2,sticky="ew")


        self.map = TkinterMapView(self)
        self.map.grid(row=1,column=0,sticky="nsew")
        self.map.set_position(46.9973931, 26.8239746) #Romania-Pildesti
        self.map.set_zoom(14)

        
        self.side_menu = tk.Frame(self, width=250, bd=2, relief="raised")
        self.side_menu.grid(row=1, column=1, sticky="ns")

        self.status_label = tk.Label(self, text="", anchor="w", bd=1, relief="sunken")
        self.status_label.grid(row=2, column=0,columnspan=2, sticky="ew")


        self.select_button = tk.Button(self.toolbar, text="Select", command=self.toggle_area_selection)
        self.select_button.pack(side="left")

        self.download_button = tk.Button(self.toolbar, text="Download", command=self.download_selected_data)
        self.download_button.pack(side="left")

        self.image_button = tk.Button(self.toolbar, text="Image", command=self.show_image)
        self.image_button.pack(side="left")

        self.graph_button = tk.Button(self.toolbar, text="Graph", command=self.graph_data)
        self.graph_button.pack(side="left")

        self.predict_button = tk.Button(self.toolbar, text="Prediction", command = self.prediction)
        self.predict_button.pack(side="left")


        self.label_start = tk.Label(self.side_menu, text="Start date")
        self.label_start.pack()

        self.calendar_start = Calendar(self.side_menu, selectmode="day")
        self.calendar_start.pack()

        self.label_end = tk.Label(self.side_menu, text="End date")
        self.label_end.pack()

        self.calendar_end = Calendar(self.side_menu, selectmode="day")
        self.calendar_end.pack()

        self.confirm_date_button = tk.Button(self.side_menu, text="Confirm date", command=self.confirmDate)
        self.confirm_date_button.pack()


        self.option_label = tk.Label(self.side_menu, text="Download:")
        self.option_label.pack(pady=(20, 5))

        self.rgb_radio = tk.Radiobutton(self.side_menu, text="RGB", variable=self.download_option, value="RGB")
        self.rgb_radio.pack(anchor="w")

        self.ndvi_radio = tk.Radiobutton(self.side_menu, text="NDVI", variable=self.download_option, value="NDVI")
        self.ndvi_radio.pack(anchor="w")

        self.ndwi_radio = tk.Radiobutton(self.side_menu, text="NDWI", variable=self.download_option, value="NDWI")
        self.ndwi_radio.pack(anchor="w")


        self.prediction_label=tk.Label(self.side_menu, text="Prediction months:")
        self.prediction_label.pack(pady=(20, 5))

        self.prediction_spinbox=tk.Spinbox(self.side_menu, from_=1, to=60, textvariable=self.prediction_months, width=10)
        self.prediction_spinbox.pack()


    def prediction(self):
        if self.downloaded_data is None:
            self.show_status(text="No data to predict!", duration=3000)
            return
        elif self.downloaded_data["option"]=="RGB":
            self.show_status(text="Cannot predict RGB!", duration=3000)
            return


        predictions = predict(results=self.downloaded_data["results"], n=self.prediction_months.get())

        dates= [result["date"] for result in predictions]
        indice= [result["value"] for result in predictions]

        graph_data(dates=dates, indice=indice, indice_name=self.downloaded_data["option"])        

    def show_image(self):
        if self.downloaded_data is None:
            self.show_status(text="No data to show!", duration=3000)
            return

        plot_image(self.downloaded_data)

    def graph_data(self):
        if self.downloaded_data is None:
            self.show_status(text="No data to graph!", duration=3000)
            return
        elif self.downloaded_data["option"] == "RGB":
            self.show_status(text="Cannot graph RGB!", duration=3000)
            return

        dates = [result["date"] for result in self.downloaded_data["results"]]
        indice = [result["value"] for result in self.downloaded_data["results"]]
        percentage = [result["percentage"] for result in self.downloaded_data["results"]]
        
        graph_data(dates=dates, indice=indice, indice_name=self.downloaded_data["option"], percentage=percentage)

    def download_selected_data(self):
        if self.download_option.get() == "None":
            self.show_status(text="No download option selected!", duration=3000)
            return
        elif self.starting_point is None:
            self.show_status(text="No area selected!", duration=3000)
            return
        elif self.start_date is None:
            self.show_status(text="No time window selected!", duration=3000)
            return
        elif self.start_date > self.end_date:
            self.show_status(text="Starting date must be befor the end date!", duration=3000)
            return
        
        bbox = BBox(bbox=[self.starting_point[1], self.starting_point[0], self.current_point[1], self.current_point[0]], crs=CRS.WGS84)

        match self.download_option.get():
            case "RGB":
                self.downloaded_data = download_rgb_data(config=self.config, collection=self.collection, bbox=bbox, start_date=self.start_date, end_date=self.end_date)
            case "NDVI":
                self.downloaded_data = download_ndvi_data(config=self.config, collection=self.collection, bbox=bbox, start_date=self.start_date, end_date=self.end_date)
            case "NDWI":
                self.downloaded_data = download_ndwi_data(config=self.config, collection=self.collection, bbox=bbox, start_date=self.start_date, end_date=self.end_date)


        self.show_status(text="!!!Data downloaded!!!")


    def confirmDate(self):
        self.start_date = self.calendar_start.selection_get()
        self.end_date = self.calendar_end.selection_get()
        self.show_status(text="Date selected!", duration=3000)




    def toggle_area_selection(self):
        self.selecting_area = not self.selecting_area

        if self.selecting_area:
            self.select_button.config(text="Cancel", relief=tk.SUNKEN)
            self.enable_selection()
            self.show_status("Selection mode: drag with the right mouse button.", duration=3000)
        else:
            self.select_button.config(text="Select", relief=tk.RAISED)
            self.disable_selection()

    def enable_selection(self):
        canvas = self.map.canvas
        canvas.bind("<ButtonPress-3>", self.on_mouse_down)
        canvas.bind("<B3-Motion>", self.on_mouse_drag)
        canvas.bind("<ButtonRelease-3>", self.on_mouse_up)

    def disable_selection(self):
        canvas = self.map.canvas
        canvas.unbind("<ButtonPress-3>")
        canvas.unbind("<B3-Motion>")
        canvas.unbind("<ButtonRelease-3>")

    def on_mouse_down(self, event):
        if not self.selecting_area:
            return

        self.dragging_mouse = True

        self.starting_point = self.map.convert_canvas_coords_to_decimal_coords(event.x, event.y)

    def on_mouse_drag(self, event):
        if not self.dragging_mouse:
            return

        self.current_point = self.map.convert_canvas_coords_to_decimal_coords(event.x, event.y)

        self.draw_selection()

    def draw_selection(self):
        if self.polygon:
            self.polygon.delete()

        lat1, lon1 = self.starting_point
        lat2, lon2 = self.current_point

        self.polygon = self.map.set_polygon([(lat1, lon1), (lat1, lon2), (lat2, lon2), (lat2, lon1)])

    def on_mouse_up(self, event):
        if not self.dragging_mouse:
            return

        self.dragging_mouse = False

        self.current_point = self.map.convert_canvas_coords_to_decimal_coords(event.x, event.y)
        self.draw_selection()

        if self.starting_point is not None:
            self.show_status("Selection locked.", duration=3000)
        

    def show_status(self, text, duration=None):
        self.status_label.config(text=text)

        if duration is not None:
            self.after(duration, self.clear_status)

    def clear_status(self):
        self.status_label.config(text="")
