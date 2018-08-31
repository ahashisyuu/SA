import os
import pickle as pkl
import pandas as pd


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
    print(models_list)
    list_dir = os.listdir(path)
    with open(os.path.join(path, '__init__.py'), 'a') as fa:
        for name in list_dir:
            if '__' not in name:
                tag = False
                line = 'from models.' + name[:-3] + ' import *'

                for li in models_list:
                    if li.startswith(line.encode(encoding='utf-8')) is True:
                        tag = True
                        break

                if tag is False:
                    fa.write(line)
                    fa.write('\n')




