import tensorflow as tf
import numpy as np

modelo = tf.keras.models.load_model('../models/embedding.h5')
modelt = tf.keras.models.load_model('../models/rnn.h5')
modelth = tf.keras.models.load_model('../models/dense.h5')
modelf = tf.keras.models.load_model('../models/softmax.h5')
predicto = modelo.predict([[0, 1, 2]])
predictt = modelt.predict([[0,1,2]])
predictth = modelth.predict([[0,1,2]])
predictf = modelf.predict([[0,1,2]])
print(predicto)
print(predictt)
print(predictth)
print(predictf)