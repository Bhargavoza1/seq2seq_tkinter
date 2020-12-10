
import tkinter as tk
from config.logger import logger
from ttkthemes import themed_tk

from pages.page1 import Page1
from pages.page2 import Page2
from pages.page3 import Page3

from tkinter import ttk, VERTICAL, HORIZONTAL, N, S, E, W


class MainView(tk.Frame):
    def __init__(self, frame):
        tk.Frame.__init__(self, frame)

        self.root = frame
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        # Create the panes and frames

        verticle_pane = ttk.PanedWindow(self.root, orient=VERTICAL)
        verticle_pane.grid(column=0, row=0, sticky=(W, E, N, S))
        horizontal_pane = ttk.PanedWindow(self.root, orient=HORIZONTAL)
        horizontal_pane.grid(column=0, row=1, sticky=(W, E, N, S))
        horizontal_pane.grid_columnconfigure(2, weight=2)

        container = tk.Frame(verticle_pane)
        container.columnconfigure(0, weight=1)
        verticle_pane.add(container, weight=1)

        self.b1 = tk.Button(horizontal_pane, text="Predict",font=('Helvetica', 12))
        self.b2 = tk.Button(horizontal_pane, text="Next", font=('Helvetica', 12))

        self.p1 = Page1(container, self.b1 ,self.b2)
        self.p2 = Page2(container,page1=self.p1)
        self.p3 = Page3(container,page1=self.p1,page2=self.p2)

        container.pack(side="top", fill="both", expand=True)

        self.p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.p3.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        self.b1.config(command=self.gotopred)
        self.b2.config( command=self.page)

        horizontal_pane.add(self.b1)
        horizontal_pane.add(self.b2)

        self.b1.pack(side="left", fill="both", expand=True)
        self.b2.pack(side="right", fill="both", expand=True)

        self.p1.show()
        self.i = 0

    def page(self):
        #load default data sets
        if self.i == 0:
            self.p1.load_data()


        pages = [self.p1, self.p2, self.p3]
        self.i += 1
        if self.i == 3:
            self.i = 0
            pages[0].show()
            self.b2["text"] = "Next"
        elif self.i == 2:
            self.b2["text"] = "Back to home"
            pages[self.i].show()
        else:
            pages[self.i].show()
            self.b2["text"] = "Next"


            #self.b2["state"] = "disabled"
            #self.b2["text"] = "disable"

    def gotopred(self):
        self.p1.load_data()
        self.p2.datasplit()
        self.p2.sethyperparam()
        self.p2.modelassemble()
        self.p3.show()
        self.i = 2
        self.b2["text"] = "Back to home"

'''
if __name__ == "__main__":
    root = tk.Tk()
    main = MainView(root)
    main.pack(side="top", fill="both", expand=True)
    root.wm_geometry("400x400")
    root.mainloop()
'''