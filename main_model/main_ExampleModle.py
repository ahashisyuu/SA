from utils import load_vocab, load_data
from models.ExampleModle import ExampleModel


def train(config):
    vocab = load_vocab(config.vocab_path)
    trainingset, validationset, _ = load_data(config.data_path)







