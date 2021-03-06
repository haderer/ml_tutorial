import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from utils import rnn_minibatch_sequencer, series_to_supervised

SEQ_LEN = 3
BATCH = 3
EPOCH = 100
STEP = 3

time_points = np.linspace(start=0, stop=20, num=20, dtype=np.int8)

print('coucou')
x = rnn_minibatch_sequencer(raw_data=time_points, batch_size=BATCH, sequence_size=SEQ_LEN,
                                           nb_epochs=EPOCH)

print(x)
print('DONE')
