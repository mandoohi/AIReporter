import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3,5),
    tf.keras.layers.SimpleRNN(3)
])
model.summary()

model.save('../models/rnn.h5')