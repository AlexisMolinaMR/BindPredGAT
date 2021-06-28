import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import Huber, MeanSquaredError
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from spektral.models import GeneralGNN
from spektral.data import DisjointLoader, BatchLoader
from spektral.layers import ECCConv, GlobalSumPool, GeneralConv, GCNConv, EdgeConv

class GraphNeuralNetwork(Model):
    '''
    '''

    def __init__(self, n_hidden):
        super().__init__()

        self.graph_conv = GeneralConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dense = Dense(1, 'linear')

    def call(self, inputs):

        out = self.graph_conv(inputs)
        out = self.pool(out)
        out = self.dense(out)

        return out
