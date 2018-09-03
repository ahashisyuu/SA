import os
import pickle as pkl
import sys

import pandas as pd
from keras import Model
from keras.callbacks import ModelCheckpoint
from evaluate import categorical_prf


def load_embedding_matrix(filepath):
    with open(filepath, 'rb') as fr:
        embedding_matrix = pkl.load(fr)
    return embedding_matrix


def load_data(filepath):
    with open(filepath, 'rb') as fr:
        data = pkl.load(fr)
    return data


def save_results(results, args):
    old_path = '../rawData'
    filename = os.path.join(old_path, 'sentiment_analysis_testa.csv')
    testa = pd.read_csv(filename)
    testa[args.arrangement] = results.argmax(axis=-1) - 3
    testa.to_csv(filename)


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


class PRFAcc(ModelCheckpoint):
    def __init__(self, filepath, monitor="val_loss", batch_size=128, validation_data=None, **kwargs):
        assert validation_data is not None
        super(PRFAcc, self).__init__(filepath, monitor=monitor, **kwargs)
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.verbose = None
        self.epochs = None

    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        assert logs is None

        self.model.predict_loss = True
        y_pred, val_loss = self.model.predict(self.validation_data[0], batch_size=self.batch_size, verbose=self.verbose)
        y_true = self.validation_data[1][0]

        group_prf, group_f, acc = categorical_prf(y_true, y_pred)

        info = '\nEPOCH {0} 的PRF值：\n'.format(epoch)
        sys.stdout.write(info)
        for i, prf in enumerate(group_prf):
            info = '{0}: ({1:.4f>8}, {2:.4f>8}, {3:.4f>8})\n'.format(i, *prf)
            sys.stdout.write(info)
        info = 'total_f1_score: {0:.4f>8}, acc: {1:.4f>8}\n'.format(group_f, acc)
        sys.stdout.write(info)
        sys.stdout.flush()

        logs = {'val_acc': acc, 'fmeasure': group_f, 'val_loss': val_loss}
        super(PRFAcc, self).on_epoch_end(epoch, logs)


class ModelForLoss(Model):
    def __init__(self, *args, predict_loss=False, **kwargs):
        super(ModelForLoss, self).__init__(*args, **kwargs)
        self.predict_loss = predict_loss

    def _make_predict_function(self):
        if self.predict_loss:
            self.outputs.append(self.total_loss)
        super(ModelForLoss, self)._make_predict_function()








