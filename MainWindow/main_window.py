import tkinter as tk
from tkintermapview import TkinterMapView
from tkcalendar import Calendar
from sentinelhub import (
    BBox,
    CRS
)
from SentinelHub.sh_download import download_data
from SentinelHub.sh_visualiztion import graph_data, plot_image
from SentinelHub.sh_prediction import predict



class MainWindow(tk.Tk):
    def __init__(self, config):
        super().__init__()

        self.download_option = tk.StringVar(value="None")
        self.prediction_option = tk.StringVar(value="None")
        self.prediction_months=tk.IntVar(value=12)

        self.title("Monitorizare indici de mediu")
        self.geometry("1280x720")
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.create_widgets()

        self.config = config
        self.disable_selection()

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
        self.map.set_position(44.9913976, 29.0827044) #Romania-Rezervatia Biosferei Delta Dunarii
        self.map.set_zoom(12)

        
        self.side_menu = tk.Frame(self, width=250, bd=2, relief="raised")
        self.side_menu.grid(row=1, column=1, sticky="ns")

        self.status_label = tk.Label(self, text="", anchor="w", bd=1, relief="sunken")
        self.status_label.grid(row=2, column=0,columnspan=2, sticky="ew")



        self.download_button = tk.Button(self.toolbar, text="Descarcă", command=self.download_selected_data)
        self.download_button.pack(side="left")

        self.image_button = tk.Button(self.toolbar, text="Afișează imagini", command=self.show_image)
        self.image_button.pack(side="left")

        self.graph_button = tk.Button(self.toolbar, text="Afișează grafic", command=self.graph_data)
        self.graph_button.pack(side="left")

        self.predict_button = tk.Button(self.toolbar, text="Prezice", command = self.prediction)
        self.predict_button.pack(side="left")



        self.select_button = tk.Button(self.side_menu, text="Selectează zona", command=self.toggle_area_selection)
        self.select_button.pack(fill="x")

        self.label_start = tk.Label(self.side_menu, text="Data de început")
        self.label_start.pack(pady=(10,0))

        self.calendar_start = Calendar(self.side_menu, selectmode="day")
        self.calendar_start.pack()

        self.label_end = tk.Label(self.side_menu, text="Data de final")
        self.label_end.pack()

        self.calendar_end = Calendar(self.side_menu, selectmode="day")
        self.calendar_end.pack()

        self.confirm_date_button = tk.Button(self.side_menu, text="Confirmă data", command=self.confirmDate)
        self.confirm_date_button.pack()


        self.option_label = tk.Label(self.side_menu, text="Opțiuni pentru descărcare:")
        self.option_label.pack(pady=(10, 5))

        self.download_option_frame = tk.Frame(self.side_menu)
        self.download_option_frame.pack()

        self.rgb_radio = tk.Radiobutton(self.download_option_frame, text="RGB", variable=self.download_option, value="RGB")
        self.rgb_radio.pack(side="left")

        self.ndvi_radio = tk.Radiobutton(self.download_option_frame, text="NDVI", variable=self.download_option, value="NDVI")
        self.ndvi_radio.pack(side="left")

        self.ndwi_radio = tk.Radiobutton(self.download_option_frame, text="NDWI", variable=self.download_option, value="NDWI")
        self.ndwi_radio.pack(side="left")


        self.prediction_label=tk.Label(self.side_menu, text="Opțiuni pentru prezicere:")
        self.prediction_label.pack(pady=(20, 5))

        self.months_frame = tk.Frame(self.side_menu)
        self.months_frame.pack()

        self.months_label = tk.Label(self.months_frame, text="Luni:")
        self.months_label.pack(side="left")

        self.prediction_spinbox=tk.Spinbox(self.months_frame, from_=1, to=60, textvariable=self.prediction_months, width=10)
        self.prediction_spinbox.pack(side="left", padx=(5,0))

        self.predict_diff_radio = tk.Radiobutton(self.side_menu, text="Modelul diferențelor", variable=self.prediction_option, value="DIFF")
        self.predict_diff_radio.pack(anchor="w")

        self.predict_imp_radio = tk.Radiobutton(self.side_menu, text="Modelul îmbunătățit", variable=self.prediction_option, value="IMP")
        self.predict_imp_radio.pack(anchor="w")


    def prediction(self):
        if self.downloaded_data is None or self.downloaded_data["results"]==[]:
            self.show_status(text="Nu există date pentru predicție!")
            return
        elif self.downloaded_data["option"]=="RGB":
            self.show_status(text="Nu se pot prezice date RGB!")
            return
        try:
            months = self.prediction_months.get()
        except Exception as e:
            print(f"Prediction failed: {e}")
            self.show_status("Numărul de luni prezise trebuie să fie un număr întreg mai mare de 0!")
            return
        
        if months <= 0 or not isinstance(months, int):
            self.show_status(text="Numărul de luni prezise trebuie să fie un număr întreg mai mare de 0!")
            return
        elif months > 60:
            self.show_status(text="Numărul de luni este prea mare! Alege până în 60 de luni de prezis (5 ani)")
            return
        elif self.prediction_option.get() == "None":
            self.show_status(text="Nu s-a ales o opțiune de predicție")
            return

        predictions = predict(results=self.downloaded_data["results"], n=months, option=self.prediction_option.get())

        if predictions is None:
            self.show_status(text="Nu exista predicție!")
            return

        dates= [result["date"] for result in predictions]
        indice= [result["value"] for result in predictions]
        percentage= [result["percentage"] for result in predictions]

        graph_data(dates=dates, indice=indice, indice_name=self.downloaded_data["option"], percentage=percentage)        

    def show_image(self):
        if self.downloaded_data is None or self.downloaded_data["results"]==[]:
            self.show_status(text="Nu există date!")
            return

        plot_image(self.downloaded_data)

    def graph_data(self):
        if self.downloaded_data is None or self.downloaded_data["results"]==[]:
            self.show_status(text="Nu există date!")
            return
        elif self.downloaded_data["option"] == "RGB":
            self.show_status(text="Nu se poate face un grafic pentru RGB!")
            return

        dates = [result["date"] for result in self.downloaded_data["results"]]
        indice = [result["value"] for result in self.downloaded_data["results"]]
        percentage = [result["percentage"] for result in self.downloaded_data["results"]]
        
        graph_data(dates=dates, indice=indice, indice_name=self.downloaded_data["option"], percentage=percentage)

    def download_selected_data(self):
        option = self.download_option.get()
        if option == "None":
            self.show_status(text="Nu s-a ales o opțiune pentru descărcarea datelor!")
            return
        elif self.starting_point is None:
            self.show_status(text="Nu s-a selectat o zonă!")
            return
        elif self.start_date is None:
            self.show_status(text="Nu s-a selectat o perioadă de timp!")
            return
        elif self.start_date > self.end_date:
            self.show_status(text="Data de început trebuie să fie înainte de data de final!")
            return
        
        bbox = BBox(bbox=[self.starting_point[1], self.starting_point[0], self.current_point[1], self.current_point[0]], crs=CRS.WGS84)
        

        self.downloaded_data = download_data(config=self.config, bbox=bbox, start_date=self.start_date, end_date=self.end_date, option=option)

        if self.downloaded_data is None or self.downloaded_data["results"] == []:
            self.show_status(text="!!!Descărcare eșuată! Încearcă o perioadă te timp mai largă")
        else:
            self.show_status(text="!!!Date descărcate!!!", duration=3000)


    def confirmDate(self):
        self.start_date = self.calendar_start.selection_get()
        self.end_date = self.calendar_end.selection_get()
        self.show_status(text="Perioadă de timp selectată!")




    def toggle_area_selection(self):
        self.selecting_area = not self.selecting_area

        if self.selecting_area:
            self.select_button.config(text="Nu mai selecta", relief=tk.SUNKEN)
            self.enable_selection()
            self.show_status("Selectează: folosește clic dreapta pentru a selecta!")
        else:
            self.select_button.config(text="Selectează zona", relief=tk.RAISED)
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
            self.show_status("Selecție salvată.")
        

    def show_status(self, text, duration=None):
        self.status_label.config(text=text)

        if duration is not None:
            self.after(duration, self.clear_status)

    def clear_status(self):
        self.status_label.config(text="")
