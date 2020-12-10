

import tkinter as tk
from tkinter import ttk


class Page3(ttk.Frame):
    Layout = "place"
    Title = "Home"

    #def __init__(self, parent, controller, SQL):
    def __init__(self, parent,page1):
        ttk.Frame.__init__(self, parent)
        self.page1 = page1


        self.label = ttk.Label(self, text="Translate english to " , font=('Helvetica', 12))
        self.label.pack()

        ttk.Label(self , text='Enter text in english:').place(relx=.1, rely=.1)

        self.message = tk.StringVar()
        ttk.Entry(self ,textvariable=self.message,   width=70).place(relx=.1, rely=.2)

        ttk.Label(self, text='Translate to  :').place(relx=.1, rely=.3)

        self.translate = tk.Label(self, text="" , font=('Helvetica', 12))
        self.translate.place(relx=.1, rely=.4)
        self.translate_button = tk.Button(self, text="Translate", font=('Helvetica', 12),command=self.evalutemode)
        self.translate_button.pack( side = 'bottom', fill='both')


    def evalutemode(self):
        self.translate.config(text=self.message.get())

    def show(self):
        strg = self.page1.getcombobox()
        strg = strg.lower()
        self.label.config(text="Translate english to "+strg)
        import config.config as cf
        print(cf.path_to_file)
        self.tkraise()