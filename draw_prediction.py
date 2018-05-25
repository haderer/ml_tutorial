import keras
import numpy as np
import matplotlib.pyplot as pl

from utils import rnn_minibatch_sequencer

model = keras.models.load_model('here.h5')

value = np.array([[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]])

result = np.array([])

predicted = model.predict(value)
print(predicted)
#predicted = model.predict(predicted)
#predicted = model.predict(predicted)
#predicted = model.predict(predicted)



