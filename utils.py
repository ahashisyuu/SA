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



