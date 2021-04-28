import numpy as np


def data_split(data, train_size, val_size, seed):
    '''
    '''

    train, val, test = np.split(data.sample(frac=1, random_state=seed), [int(train_size*len(data)), int(val_size*len(data))])

    return train, val, test
