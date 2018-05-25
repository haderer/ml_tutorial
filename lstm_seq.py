# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as pl
from utils import rnn_minibatch_sequencer, series_to_supervised

# In[29]:


SEQ_LEN = 10
BATCH = 1
EPOCH = 100

# In[30]:

time_points = np.linspace(start=0, stop=1, num=1000, dtype=np.float32)
time_points = np.reshape(time_points, (-1, 1))

# In[31]:


# data_seq_generator = rnn_minibatch_sequencer(raw_data=time_points, batch_size=BATCH, sequence_size=SEQ_LEN,
#                                             nb_epochs=EPOCH)

data_seq_generator = series_to_supervised(time_points, n_in=SEQ_LEN, n_out=1).values
print(data_seq_generator)
x = data_seq_generator[:, 0:SEQ_LEN].reshape(-1, SEQ_LEN, 1)
x = x[0:len(x) - (len(x) % BATCH)]

y = data_seq_generator[:, SEQ_LEN]
y = y[0:len(y) - (len(y) % BATCH)]

print(x.shape)
print(y.shape)

# In[40]:

from keras import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(1, batch_input_shape=(BATCH, SEQ_LEN, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['cosine_proximity', 'mae'])
#model.summary()

# In[41]:

for i in range(100):
    model.fit(x, y, epochs=1, batch_size=BATCH, verbose=1, shuffle=False)
    model.reset_states()
    if i % 5:
        model.save('here.h5')


