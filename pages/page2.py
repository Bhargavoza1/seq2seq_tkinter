import tkinter as tk
from tkinter import ttk, VERTICAL, HORIZONTAL, N, S, E, W
from config.logger import logger
from sklearn.model_selection import train_test_split
import config.preprocess_data as preproc_data
import config.config  as config
import tensorflow as tf
import models.encoder as encoder
import models.basic as decoder
import models.bahdanau as bahdanau_decoder
import models.luong as luong_decoder
import os
import asyncio
import threading

import time

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)




class Page2(ttk.Frame):
    Layout = "place"
    Title = "Home"

    # def __init__(self, parent, controller, SQL):
    def __init__(self, parent,page1):
        ttk.Frame.__init__(self, parent)
        self.models={'encoder':encoder,
                     'decoders':{'BASIC':decoder,'BAHDANAU':bahdanau_decoder,'LUONG':luong_decoder } }
        self.page1 = page1
        self.modelname =self.page1.getcombobox2()
        self.loadedstringcheck = " "
        label = ttk.Label(self, text="Adjust some hyperparameter", font=('Helvetica', 12))
        label.pack()

        label2 = ttk.Label(self,
                           text='Batch size', font=('Helvetica', 10))

        label2.place(relx=.1, rely=.2)

        self.batch_size = tk.Scale(self, from_=0, to=64, showvalue=False, orient=HORIZONTAL, command=self.ShowTheValue)
        self.batch_size.set(64)
        self.batch_size.place(relx=.3, rely=.2)

        self.value_label = ttk.Label(self, text="0")
        self.value_label.place(relx=.6, rely=.2)

        label3 = ttk.Label(self,
                           text='Embedding Dim', font=('Helvetica', 10))

        label3.place(relx=.1, rely=.3)

        self.embedding_dim = tk.Scale(self, from_=150, to=300, showvalue=False, orient=HORIZONTAL,
                                      command=self.ShowTheValue2)
        self.embedding_dim.place(relx=.3, rely=.3)
        self.embedding_dim.set(256)
        self.value_label2 = ttk.Label(self, text="0")
        self.value_label2.place(relx=.6, rely=.3)

        label4 = ttk.Label(self,
                           text='Rnn units', font=('Helvetica', 10))

        label4.place(relx=.1, rely=.4)

        self.rnn_units = tk.Scale(self, from_=800, to=2048, showvalue=False, orient=HORIZONTAL,
                                  command=self.ShowTheValue3)
        self.rnn_units.place(relx=.3, rely=.4)
        self.rnn_units.set(1024)
        self.value_label3 = ttk.Label(self, text="0")
        self.value_label3.place(relx=.6, rely=.4)

        label5 = ttk.Label(self,
                           text='Epoch', font=('Helvetica', 10))

        label5.place(relx=.1, rely=.5)

        self.epoch = tk.Scale(self, from_=1, to=100, showvalue=False, orient=HORIZONTAL,
                              command=self.ShowTheValue4)
        self.epoch.place(relx=.3, rely=.5)
        self.epoch.set(10)
        self.value_label4 = ttk.Label(self, text="0")
        self.value_label4.place(relx=.6, rely=.5)

        self.defaultvalues = tk.Button(self, text="Set values to default", )
        self.defaultvalues.place(relx=.3, rely=.6)
        self.defaultvalues.config(command=self.resetvalue)
        async_loop = asyncio.get_event_loop()
        self.train_button = tk.Button(self, text="Train model", font=('Helvetica', 12), command= lambda:self.do_tasks(async_loop))
        self.train_button.pack(side='bottom', fill='both')



    def ShowTheValue(self, args):
        self.value_label["text"] = str(self.batch_size.get())

    def ShowTheValue2(self, args):
        self.value_label2["text"] = str(self.embedding_dim.get())

    def ShowTheValue3(self, args):
        self.value_label3["text"] = str(self.rnn_units.get())

    def ShowTheValue4(self, args):
        self.value_label4["text"] = str(self.epoch.get())

    def resetvalue(self):
        self.batch_size.set(64)
        self.rnn_units.set(1024)
        self.embedding_dim.set(256)
        self.epoch.set(10)



    def show(self):


        self.datasplit()
        self.sethyperparam()
        self.modelassemble()
        logger.warning("data is ready to train")
        logger.info("adjust your hyperparameter")

        self.tkraise()



    def datasplit(self):
        if self.loadedstringcheck !=  config.getPreprocessData():
            self.target_tensor, self.input_tensor, self.targ_lang, self.inp_lang = preproc_data.load_dataset(
                config.getPreprocessData(), 30000)
            self.max_length_targ, self.max_length_inp = self.target_tensor.shape[1], self.input_tensor.shape[1]

            self.input_tensor_train, self.input_tensor_val, self.target_tensor_train, self.target_tensor_val = train_test_split(
                self.input_tensor,
                self.target_tensor,
                test_size=0.2)



            self.loadedstringcheck = config.getPreprocessData()
            logger.info("input_tensor_train : {}".format(len(self.input_tensor_train)))
            logger.info("target_tensor_train : {}".format(len(self.target_tensor_train)))
            logger.info("input_tensor_val : {}".format(len(self.input_tensor_val)))
            logger.info("target_tensor_val : {}".format(len(self.target_tensor_val)))



    def sethyperparam(self ):


            self.BUFFER_SIZE = len( self.input_tensor_train)
            self.BATCH_SIZE = self.batch_size.get()
            self.steps_per_epoch = len( self.input_tensor_train) // self.BATCH_SIZE
            self.Embedding_dim = self.embedding_dim.get()
            self.units = self.rnn_units.get()
            self.EPOCHS = self.epoch.get()
            self.vocab_inp_size = len( self.inp_lang.word_index) + 1
            self.vocab_tar_size = len( self.targ_lang.word_index) + 1
            self.dataset = tf.data.Dataset.from_tensor_slices(
                ( self.input_tensor_train,  self.target_tensor_train)).shuffle(self.BUFFER_SIZE)
            self.dataset2 = self.dataset.batch(self.BATCH_SIZE, drop_remainder=True)

            #logger.info(len(self.input_tensor_train), len(self.target_tensor_train), len(self.input_tensor_val), len(self.target_tensor_val))



    def modelassemble(self):

        self.modelname = self.page1.getcombobox2()
        print(self.modelname)
        print(self.vocab_inp_size, self.Embedding_dim, self.units,self.BATCH_SIZE)

        self.page_encoder = self.models['encoder'].Encoder(self.vocab_inp_size,
                                                      self.Embedding_dim,
                                                      self.units,
                                                      self.BATCH_SIZE)
        self.decoder_page = self.models['decoders'][self.modelname].Decoder(self.vocab_tar_size,
                                                                       self.Embedding_dim,
                                                                       self.units,
                                                                       self.BATCH_SIZE)

        self.checkpoint_dir = './chekpoints/' + self.page1.getcombobox() + '/' + self.page1.getcombobox2()+ '/'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "chekpoints")
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                              encoder=self.page_encoder,
                                              decoder=self.decoder_page)


        '''
        example_input_batch, example_target_batch = next(iter(self.dataset2))
        example_input_batch.shape,  example_target_batch.shape
        # sample input
        sample_hidden = self.page_encoder.init_hidden_state()
        sample_output, sample_hidden = self.page_encoder(example_input_batch, sample_hidden)
        print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
        print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


        if self.modelname == 'BASIC':
            sample_decoder_output, _  = self.decoder_page(tf.random.uniform((self.BATCH_SIZE, 1)),
                                    sample_hidden)
            print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

        if self.modelname == 'BAHDANAU':
            sample_decoder_output, _, _ = self.decoder_page(tf.random.uniform((self.BATCH_SIZE, 1)),
                                                  sample_hidden, sample_output)

            print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

        if self.modelname == 'LUONG':
            sample_decoder_output, _, _ = self.decoder_page(tf.random.uniform((self.BATCH_SIZE, 1)),
                                                  sample_hidden, sample_output)

            print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))'''

    def _asyncio_thread(self,async_loop):
        async_loop.run_until_complete(self.train())

    def do_tasks(self,async_loop):
        """ Button-Event-Handler starting the asyncio part. """
        threading.Thread(target=self._asyncio_thread, args=(async_loop,)).start()

    def evalutemode(self):
            print(self.modelname)

    @tf.function
    def train_step(self,inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            '''
                here we getting encoder output ((64, 20, 1024) , (64, 1024)).
                by doing this enc_output[1:] we get last state(64,1024).
            '''
            enc_output, enc_hidden = self.page_encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self.targ_lang.word_index['<start>']] * self.BATCH_SIZE, 1)

            ''' 
                Teacher forcing - feeding the target as the next input
                i.e. first we passing encoder last state to decoder initial_state 
                     and as input to the first time stamp we are passing <start> tag from every batch.
                     out of first time stamp is 64, 1, 4483.this will go under argmax and find loss with next word of sentence(label).
                     after that on next time stamp first word is input and second word is label.
            '''
            for t in range(1, targ.shape[1]):  # 12
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder_page(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = self.page_encoder.trainable_variables +self.decoder_page.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss






    async def train(self):
        # lead check point
        logger.warning('training started')
        self.sethyperparam()

        self.modelassemble()

        a = tf.train.latest_checkpoint(self.checkpoint_dir, latest_filename=None)

        #if a is not None:
        self.checkpoint.restore(a)



        for epoch in range(self.EPOCHS):
            start = time.time()

            '''on starting of every epoch iteration ecoder initial state is always zero '''
            enc_hidden = self.page_encoder.init_hidden_state()

            total_loss = 0

            for (batch, (inp, targ)) in enumerate(self.dataset2.take(self.steps_per_epoch)):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
                    logger.info('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                       batch,
                                                                       batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            logger.debug('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.steps_per_epoch))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.steps_per_epoch))

            logger.debug('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            logger.warning('training finished')

    '''this is same as train but we know we dont need back propagation in evaluate.
       but we are checking. if we reached at <end> tag?
    '''
