import tkinter as tk
from tkintermapview import TkinterMapView

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Titlu obscur")
        self.geometry("1280x720")
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.create_widgets()

        self.selecting_area = False
        self.first_corner = None
        self.second_corner = None

    def create_widgets(self):
        self.toolbar = tk.Frame(self, height=30, bd=2, relief="raised")
        self.toolbar.grid(row=0,column=0,sticky="ew")

        self.select_button = tk.Button(self.toolbar, text="Select", command=self.toggle_area_selection)
        self.select_button.pack(side="left")

        self.map = TkinterMapView(self)
        self.map.grid(row=1,column=0,sticky="nsew")
        self.map.set_position(46, 25) #Romania
        self.map.set_zoom(6)

        self.map.add_left_click_map_command(self.on_map_click)

    def toggle_area_selection(self):
        self.selecting_area = not self.selecting_area

        if self.selecting_area:
            self.select_button.config(text="Cancel", relief=tk.SUNKEN)
        else:
            self.select_button.config(text="Select", relief=tk.RAISED)

    def on_map_click(self, coordinates):
        if not self.selecting_area:
            return

        if self.first_corner is None:
            self.first_corner = coordinates
            print("First:", coordinates)
        else:
            self.second_corner = coordinates
            print("Second:", coordinates)

            self.finish_selection()

    def finish_selection(self):
        lat1, lon1 = self.first_corner
        lat2, lon2 = self.second_corner

        self.polygon = self.map.set_polygon([(lat1, lon1), (lat1, lon2), (lat2, lon2), (lat2, lon1)])

        self.selecting_area = False
        self.first_corner = None
        self.second_corner = None

        self.select_button.config(text="Select", relief=tk.RAISED)
        



def test():
    app = Application()
    app.mainloop()
