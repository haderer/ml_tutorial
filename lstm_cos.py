import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from utils import rnn_minibatch_sequencer, series_to_supervised

SEQ_LEN = 3
BATCH = 3
EPOCH = 100000
STEP = 3

time_points = np.linspace(start=0, stop=20, num=20, dtype=np.int8)
time_points = list(time_points)

# print(pd.DataFrame(list(rnn_minibatch_sequencer(
#    raw_data=time_steps,
#    batch_size=BATCH,
#    sequence_size=SEQ_LEN,
#    nb_epochs=EPOCH))))

dataX = []
dataY = []

for i in range(0, len(time_points) - SEQ_LEN, 1):
    seq_in = time_points[i:i + SEQ_LEN]
    seq_out = time_points[i + SEQ_LEN]
    dataX.append([char for char in seq_in])
    dataY.append(seq_out)

x_1 = pd.DataFrame(rnn_minibatch_sequencer(raw_data=time_points, batch_size=BATCH, sequence_size=SEQ_LEN,
                                           nb_epochs=EPOCH))
x_2 = series_to_supervised(time_points, n_in=3, n_out=3)
print(x_1)
print(x_2)
