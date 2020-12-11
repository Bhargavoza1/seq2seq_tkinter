

import logging
import signal

import tkinter as tk

from tkinter import ttk, VERTICAL, HORIZONTAL, N, S, E, W

from config.logger import logger
from extras.clokck import Clock
from pages.console import ConsoleUi
from pages.mainframe import MainView


class FormUi:

    def __init__(self, frame):
        self.frame = frame
        # Create a combobbox to select the logging level
        values = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.level = tk.StringVar()
        ttk.Label(self.frame, text='Level:').grid(column=0, row=0, sticky=W)
        self.combobox = ttk.Combobox(
            self.frame,
            textvariable=self.level,
            width=25,
            state='readonly',
            values=values
        )
        self.combobox.current(0)
        self.combobox.grid(column=1, row=0, sticky=(W, E))
        # Create a text field to enter a message
        self.message = tk.StringVar()
        ttk.Label(self.frame, text='Message:').grid(column=0, row=1, sticky=W)
        ttk.Entry(self.frame, textvariable=self.message, width=25).grid(column=1, row=1, sticky=(W, E))
        # Add a button to log the message
        self.button = ttk.Button(self.frame, text='Submit', command=self.submit_message)
        self.button.grid(column=1, row=2, sticky=W)

    def submit_message(self):
        # Get the logging level numeric value
        lvl = getattr(logging, self.level.get())
        logger.log(lvl, self.message.get())




class App:

    def __init__(self, root):
        self.root = root
        root.title('Logging Handler')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        # Create the panes and frames

        horizontal_pane = ttk.PanedWindow(self.root, orient=HORIZONTAL)
        horizontal_pane.pack(side="top", fill="both", expand=True)

        form_frame = ttk.PanedWindow(horizontal_pane )
        form_frame.columnconfigure(0, weight=1)
        form_frame.rowconfigure(0, weight=1)
        horizontal_pane.add(form_frame, weight=2)

        console_frame = ttk.Labelframe(horizontal_pane, text="Console")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        horizontal_pane.add(console_frame, weight=0)

        # Initialize all frames
        #self.form = FormUi(form_frame)
        self.form = MainView(form_frame  )
        self.console = ConsoleUi(console_frame)

        #self.clock = Clock()
        #self.clock.start()

        self.root.protocol('WM_DELETE_WINDOW', self.quit)
        self.root.bind('<Control-q>', self.quit)
        signal.signal(signal.SIGINT, self.quit)

    def quit(self, *args):
        #self.clock.stop()
        self.root.destroy()


def main():
    logging.basicConfig(level=logging.DEBUG)
    root = tk.Tk()
    app = App(root)
    root.geometry("1300x500")
    app.root.mainloop()


if __name__ == '__main__':
    main()
