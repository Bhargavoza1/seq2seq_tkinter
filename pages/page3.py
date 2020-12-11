import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf
import config.preprocess_data as preproc
from config.logger import logger
class Page3(ttk.Frame):
    Layout = "place"
    Title = "Home"

    #def __init__(self, parent, controller, SQL):
    def __init__(self, parent,page1,page2):
        ttk.Frame.__init__(self, parent)
        self.page1 = page1
        self.page2 = page2

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

        self.translate.config(text=self.translateword(self.message.get()))

    def show(self):
        strg = self.page1.getcombobox()
        strg = strg.lower()
        self.label.config(text="Translate english to "+strg)

        a = tf.train.latest_checkpoint(self.page2.checkpoint_dir, latest_filename=None)
        self.page2.checkpoint.restore(a)
        #logger.info(self.page2.targ_lang.index_word[1])
        print(self.page1.getcombobox2( ) )
        self.tkraise()

    def evaluate(self,sentence):
        attention_plot = np.zeros((self.page2.max_length_targ, self.page2.max_length_inp))

        sentence =preproc.preproc_sentence(sentence)

        inputs = [self.page2.inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=self.page2.max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.page2.units))]
        enc_out, enc_hidden = self.page2.page_encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.page2.targ_lang.word_index['<start>']], 0)

        for t in range(self.page2.max_length_targ):
            predictions, dec_hidden, attention_weights =self.page2.decoder_page(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += self.page2.targ_lang.index_word[predicted_id] + ' '

            if self.page2.targ_lang.index_word[predicted_id] == '<end>':
                return result

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result
    def translateword(self,sentence):
        result  = self.evaluate(sentence)


        print('Predicted translation: {}'.format(result))
        return result
