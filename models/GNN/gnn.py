import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import Huber, MeanSquaredError
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam, Nadam

from spektral.models import GeneralGNN


class GraphNeuralNetwork(data):
    '''
    '''
    def __init__(self, F, S, n_out):

        self.F = data.n_node_features  # Dimension of node features
        self.S = data.n_edge_features  # Dimension of edge features
        self.n_out = data.n_labels  # Dimension of the target

    def GNN():
        '''
        '''

        X_in = Input(shape=(F,), name="X_in")
        A_in = Input(shape=(None,), sparse=True, name="A_in")
        E_in = Input(shape=(S,), name="E_in")
        I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

        X_1 = ECCConv(32, activation="relu")([X_in, A_in, E_in])
        X_2 = ECCConv(32, activation="relu")([X_1, A_in, E_in])
        X_3 = GlobalSumPool()([X_2, I_in])
        output = Dense(n_out)(X_3)

        # Build model
        model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
        opt = Adam(lr=learning_rate)
        loss_fn = MeanSquaredError()


    def train_step(inputs, target):
        '''
        '''

        with tf.GradientTape() as tape:

            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions)
            loss += sum(model.losses)

            mse = mean_squared_error(target, predictions)
            mae = mean_absolute_error(target, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        return loss, mse, mae

    def train_GNN():
        '''
        '''

        print("\nFitting model\n")
        current_batch = 0
        model_loss = 0

        for batch in loader_train:
            outs = train_step(*batch)

            model_loss += outs[0]
            current_batch += 1

            mae = outs[2]

            if current_batch == loader_tr.steps_per_epoch:

                print("Loss: {}\t|\t MAE: {}".format(model_loss / loader_tr.steps_per_epoch, mae))
                model_loss = 0
                current_batch = 0

    def evaluate():
        '''
        '''

        print("\nTesting model\n")
        model_loss = 0

        for batch in loader_val:

            inputs, target = batch
            predictions = model(inputs, training=False)
            model_loss += loss_fn(target, predictions)

        model_loss /= loader_te.steps_per_epoch
        mae = mean_absolute_error(target, predictions)

        print("Done. Test loss: {}\t|\t MAE: {}".format(model_loss, mae))
