import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        # embedding_dim = 256
        # vocab_size = 8562
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz #64
        self.enc_units = enc_units #1024
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        x = self.embedding(x) # 64, 20, 256 (batch , max length of sentence ,embedding_dim  )
        output, state_h  = self.gru(x, initial_state=hidden) #  output = 64, 20, 1024 # state_h(last state) = 64, 1024
        return output , state_h

    def init_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)) # 64x1024


        #return (tf.zeros([self.batch_sz, self.lstm_size]),
         #   tf.zeros([self.batch_sz, self.lstm_size]))

