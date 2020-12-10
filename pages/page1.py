# -*- coding: utf-8 -*-

import os

from tkinter import ttk, VERTICAL, HORIZONTAL, N, S, E, W

import tkinter as tk
from tkinter import ttk

import config.config  as config
from config.logger import logger







class Page1(ttk.Frame):
    Layout = "place"
    Title = "Home"

    #def __init__(self, parent, controller, SQL):
    def __init__(self, parent,b1,b2):
        ttk.Frame.__init__(self, parent)
        self.selectedlang= 'DEBUG'
        self.frame = self
        self.button1 = b1
        self.button2 = b2

        # Create a combobbox to select the logging level
        models = ['BASIC','BAHDANAU','LUONG']
        self.level2 = tk.StringVar()
        ttk.Label(self.frame, text='Select model:', font=('Helvetica', 10)).place(relx=.1, rely=.1)
        self.combobox2 = ttk.Combobox(
            self.frame, font=('Helvetica', 10),
            textvariable=self.level2,
            width=25,
            state='readonly',
            values=models
        )
        self.combobox2.current(0)
        self.combobox2.place(relx=.3, rely=.1)


        values = ['FRENCH', 'SPANISH']
        self.level = tk.StringVar()
        ttk.Label(self.frame, text='Translate english to:' ,font=('Helvetica', 10)).place(relx=.1, rely=.2)

        self.label1 = ttk.Label(self.frame,  foreground="red",
                                font=('Helvetica', 12))
        self.label1.pack(side="bottom")

        self.label2 = ttk.Label(self.frame, foreground="blue",
                                font=('Helvetica', 12))

        self.level.trace('w', self.OptionCallBack)
        self.combobox = ttk.Combobox(
            self.frame,font=('Helvetica', 10),
            textvariable=self.level,
            width=25,
            state='readonly',
            values=values
        )
        self.downlaod_data = tk.Button(self.frame, text="Download dataset",  )

        #config.loaddata(self.combobox.get())
        self.combobox.current(0)
        self.combobox.place(relx=.3, rely=.2)
        self.downlaod_data.place(relx=.3, rely=.3)
        self.downlaod_data.config(command=self.downlaod_data_call)

        self.label2.place(relx=.3, rely=.4)


    #self.button1.config(command=self.loaddata)

    def OptionCallBack(self,*args ):
        if not os.path.isdir('./assets/'+config.lagselector[self.getcombobox()][1]):
            self.button1["state"] = "disabled"
            self.button1["text"] = "disable"
        elif not os.path.isdir('./chekpoints/') :
            self.button1["state"] = "disabled"
            self.button1["text"] = "disable"
            self.label1.config(text='*You have to train the model',)

        else:
            self.button1["state"] = "normal"
            self.button1["text"] = "Predict"
            self.label1.config(text='', )

        #print(os.path.isdir('./assets/' + config.lagselector[self.getcombobox()][1]))
        if  os.path.isdir('./assets/'+config.lagselector[self.getcombobox()][1]):
            self.downlaod_data["state"] = "disabled"
            self.button2["state"] = "normal"
            self.label2.config(text='data set exists', foreground="blue" )
        else:
            self.downlaod_data["state"] = "normal"
            self.button2["state"] = "disabled"
            self.downlaod_data["text"] = "Download dataset"
            self.label2.config(text='data set does not exists',foreground="red" )


    def downlaod_data_call(self):
        self.load_data()
        logger.info("data set downaloded")
        self.downlaod_data["state"] = "disabled"

        self.label2.config(text='data set exists', foreground="blue")
        self.button2["state"] = "normal"

    def load_data(self):
        config.loaddata(self.getcombobox())

    def getcombobox(self):
        return self.combobox.get()

    def getcombobox2(self):
        return self.combobox2.get()

    def show(self):
        self.tkraise()



