import tensorflow as tf
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    # values = (64, 20, 1024)
    # query = (64, 1024)
    query_with_time_axis = tf.expand_dims(query, 1) # (64, 1, 1024)
    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))#(64, 20, 1)


    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values #(64, 20, 1024)
    context_vector = tf.reduce_sum(context_vector, axis=1)#(64, 1024)
    return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        #x(64, 1)
        #hidden(64, 1024)
        #enc_output(64, 20, 1024)
        context_vector, attention_weights = self.attention(hidden, enc_output) # context_vector = 64, 1024, attention_weights = 64, 20, 1
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x) #64,1,256

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1) # (64, 1, 1280)


        # passing the concatenated vector to the GRU
        output, state = self.gru(x) #output=64, 1, 1024, state=64, 1024
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2])) # (64, 1024)

        # output shape == (batch_size, vocab)
        x = self.fc(output)#(64, 4483)

        return x, state, attention_weights
