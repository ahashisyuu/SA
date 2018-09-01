import os
import pickle as pkl
import pandas as pd
from keras.callbacks import Callback

from evaluate import categorical_prf


def load_embedding_matrix(filepath):
    with open(filepath, 'rb') as fr:
        embedding_matrix = pkl.load(fr)
    return embedding_matrix


def load_data(filepath):
    with open(filepath, 'rb') as fr:
        data = pkl.load(fr)
    return data


def check_models(path):
    # open __init__.py file for get all 'import'
    with open(os.path.join(path, '__init__.py'), 'rb') as fr:
        models_list = fr.readlines()
    models_list = [string.strip() for string in models_list]
    list_dir = os.listdir(path)
    with open(os.path.join(path, '__init__.py'), 'a') as fa:
        for name in list_dir:
            if '__' not in name and 'Example' not in name:
                tag = False
                line = 'from models.' + name[:-3] + ' import *'

                for li in models_list:
                    if li.startswith(line.encode(encoding='utf-8')) is True:
                        tag = True
                        break

                if tag is False:
                    fa.write(line)
                    fa.write('\n')


def check_layers(path):
    # open __init__.py file for get all 'import'
    with open(os.path.join(path, '__init__.py'), 'rb') as fr:
        layers_list = fr.readlines()
    layers_list = [string.strip() for string in layers_list]
    list_dir = os.listdir(path)
    with open(os.path.join(path, '__init__.py'), 'a') as fa:
        for name in list_dir:
            if '__' not in name:
                tag = False
                line = 'from layers.' + name[:-3] + ' import *'

                for li in layers_list:
                    if li.startswith(line.encode(encoding='utf-8')) is True:
                        tag = True
                        break

                if tag is False:
                    fa.write(line)
                    fa.write('\n')


def learning_rate(epoch, lr):
    return lr


class PRFAcc(Callback):
    def __init__(self, batch_size=128, validation_data=None, verbose=1):
        assert validation_data is not None
        super(PRFAcc, self).__init__()
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        assert logs is None
        y_pred = self.model.predict(self.validation_data[0], batch_size=self.batch_size, verbose=self.verbose)
        y_true = self.validation_data[1][0]

        group_prf, group_f = categorical_prf(y_true, y_pred)






