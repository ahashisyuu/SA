import pandas as pd
import os


def load_rawData(data_path):
    trainingset = None
    validationset = None
    testa = None
    listdir = os.listdir(data_path)
    for name in listdir:
        if 'trainingset.csv' in name:
            trainingset = pd.read_csv(os.path.join(data_path, name))
        if 'validationset.csv' in name:
            validationset = pd.read_csv(os.path.join(data_path, name))
        if 'testa.csv' in name:
            testa = pd.read_csv(os.path.join(data_path, name))
            testa.to_csv()
    return trainingset, validationset, testa



