import tensorflow as tf
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        # vocab_size=4483
        #embedding_dim = 256
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz #64
        self.dec_units = dec_units #1024
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def __call__(self,x, hidden, enc_output):
        x = self.embedding(x) #64, 20, 256 (batch , max length of sentence ,embedding_dim  )
        output, state_h  = self.gru(x,hidden )  #output = 64, 20, 1024 # state_h (last state) = 64, 1024
        x = self.fc(output) # 64, 1, 4483
        x = tf.squeeze(  x, axis=1 )
        _=0
        return x, state_h , _