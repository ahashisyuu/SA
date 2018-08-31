import pickle as pkl
import pandas as pd


def load_vocab(filepath):
    with open(filepath, 'rb') as fr:
        vocab = pkl.load(fr)
    return vocab


def load_data(filepath):
    with open(filepath, 'rb') as fr:
        data = pkl.load(fr)
    return data



