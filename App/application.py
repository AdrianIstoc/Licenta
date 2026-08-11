import tkinter as tk
from tkintermapview import TkinterMapView
from tkcalendar import Calendar
from SentinelHub.sh_analysis import VegetationAnalysis
from sentinelhub import (
    BBox,
    CRS
)
from SentinelHub.sh_collections import get_sentinel2_l2a


class Application(tk.Tk):
    def __init__(self, config):
        super().__init__()

        self.title("Titlu obscur") #Rename window
        self.geometry("1280x720")
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
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

    def create_widgets(self):
        self.toolbar = tk.Frame(self, height=30, bd=2, relief="raised")
        self.toolbar.grid(row=0,column=0,sticky="ew")

        self.status_label = tk.Label(self, text="", anchor="w", bd=1, relief="sunken")
        self.status_label.grid(row=2, column=0, sticky="ew")

        self.select_button = tk.Button(self.toolbar, text="Select", command=self.toggle_area_selection)
        self.select_button.pack(side="left")

        self.date_button = tk.Button(self.toolbar, text="Date", command=self.popupDate)
        self.date_button.pack(side="right")

        self.analysis_button = tk.Button(self.toolbar, text="Analysis", command = self.print_analysis)
        self.analysis_button.pack(side="left")

        self.map = TkinterMapView(self)
        self.map.grid(row=1,column=0,sticky="nsew")
        self.map.set_position(46.9973931, 26.8239746) #Romania-Pildesti
        self.map.set_zoom(14)

    def print_analysis(self):
        if self.start_date is None or self.starting_point is None:
            return
        bbox = BBox(bbox=[self.starting_point[1], self.starting_point[0], self.current_point[1], self.current_point[0]], crs=CRS.WGS84)

        analysis = VegetationAnalysis(config=self.config, collection=self.collection, bbox=bbox, start_date=self.start_date, end_date=self.end_date)

        analysis.run()
        print(analysis.results)
        analysis.graph_ndvi()

    def popupDate(self):
        def print_dates():
            self.start_date = calendar_start.selection_get()
            self.end_date = calendar_end.selection_get()
            print(self.start_date)
            print(self.end_date)
            # popupWindow.destroy()

        popupWindow = tk.Toplevel(self)
        popupWindow.title("Date selection")
        # popupWindow.grab_set()

        label_start = tk.Label(popupWindow, text="Start Date")
        label_start.grid(row=0, column=0)

        label_end = tk.Label(popupWindow, text="End Date")
        label_end.grid(row=0, column=1)

        calendar_start = Calendar(popupWindow, selectmode="day", cursor="hand1", year=2020, month=2, day=5)
        calendar_start.grid(row=1, column=0)

        calendar_end = Calendar(popupWindow, selectmode="day", cursor="hand1", year=2021, month=2, day=5)
        calendar_end.grid(row=1, column=1)

        button_date = tk.Button(popupWindow, text="Ok", command=print_dates)
        button_date.grid(row=2, column=0)


    def toggle_area_selection(self):
        self.selecting_area = not self.selecting_area

        if self.selecting_area:
            self.select_button.config(text="Cancel", relief=tk.SUNKEN)
            self.enable_selection()
            self.show_status("Selection mode: drag with the middle mouse button.", duration=3000)
        else:
            self.select_button.config(text="Select", relief=tk.RAISED)
            self.disable_selection()
            if self.starting_point is not None:
                self.show_status("Selection locked.", duration=3000)

    def enable_selection(self):
        canvas = self.map.canvas
        canvas.bind("<ButtonPress-2>", self.on_mouse_down)
        canvas.bind("<B2-Motion>", self.on_mouse_drag)
        canvas.bind("<ButtonRelease-2>", self.on_mouse_up)

    def disable_selection(self):
        canvas = self.map.canvas
        canvas.unbind("<ButtonPress-2>")
        canvas.unbind("<B2-Motion>")
        canvas.unbind("<ButtonRelease-2>")

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

        print(self.starting_point)
        print(self.current_point)

    def show_status(self, text, duration=None):
        self.status_label.config(text=text)

        if duration is not None:
            self.after(duration, self.clear_status)

    def clear_status(self):
        self.status_label.config(text="")
