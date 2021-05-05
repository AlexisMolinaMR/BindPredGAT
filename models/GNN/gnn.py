import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import Huber, MeanSquaredError
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from spektral.models import GeneralGNN
from spektral.data import DisjointLoader, BatchLoader
from spektral.layers import ECCConv, GlobalSumPool, GeneralConv, GCNConv

class GraphNeuralNetwork():
    '''
    '''
    def __init__(self, learning_rate, batch_size, epochs, train_loader, test_loader, data, **kwargs):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loader_tr = train_loader
        self.loader_te = test_loader
        self.data = data

        super().__init__(**kwargs)

    def train_GNN(self):
        '''
        '''

        def GNN():
            '''
            '''

            # Build model
            N_in = Input(shape=(None, self.data.n_node_features))
            A_in = Input(shape=(None, None))
        #    E_in = Input(shape=(None, 1))

            X_1 = GCNConv(32, activation="relu")([N_in, A_in])
        #    X_2 = GCNConv(31, activation="relu")([X_1, A_in])
            X_3 = GlobalSumPool()([X_1])

            output = Dense(self.data.n_labels)(X_3)

            model = Model(inputs=[N_in, A_in], outputs=output)
            optimizer = Adam(lr=self.learning_rate)
            model.compile(optimizer=optimizer, loss="mae")

            model.summary()

            return model, optimizer

        model, optimizer = GNN()

        split = int(0.8 * len(self.data))
        data_tr, data_te = self.data[:split], self.data[split:]

        loader_tr = BatchLoader(data_tr, batch_size=self.batch_size)

        model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=self.epochs)

    def evaluate(self):
        '''
        '''

        model.evaluate(self.loader_te.load(), steps=self.loader_te.steps_per_epoch)
