import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('../models/softmax.h5')

predict = model.predict([[0, 1, 2]])

index_word = ['가','나','다','라','마','비']
index = np.argmax(predict[0])