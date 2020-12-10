import tensorflow as tf
class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)

    def __call__(self, decoder_output, encoder_output):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # decoder_output shape: (batch_size, 1, rnn_size)
        # score will have shape: (batch_size, 1, max_len)

        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True) # score = 64,1,11
        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)# alignment = 64,1,11

        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, encoder_output)# context = 64, 1, 1024

        return context, alignment



class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()

    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention = LuongAttention(self.dec_units)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.wc = tf.keras.layers.Dense(self.dec_units, activation='tanh')
    self.ws = tf.keras.layers.Dense(vocab_size)

    # used for attention


  def call(self, x, hidden, enc_output):
    # x (64, 1)
    # hidden (64, 1024)
    # enc_output (64, 11, 1024)
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)

    x = self.embedding(x)
    output, state = self.gru(x,hidden) # output = 64, 1, 1024  state = 64, 1024

    context, alignment = self.attention(output, enc_output) # context= 64, 1, 1024, alignment= 64,1,11

    # x shape after concatenation == (batch_size, 1, rnn_size + rnn_size)
    x = tf.concat([tf.squeeze(context, 1), tf.squeeze(output, 1)], 1) # x = (64, 2048)

    x = self.wc(x)# x = (64, 1024)

    # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
    x = self.ws(x) # x = (64, 9414)

    return x, state, alignment
