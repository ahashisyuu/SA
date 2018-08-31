from utils import load_vocab, load_data
from models.ExampleModel import ExampleModel


def train(config):
    embedding_matrix = load_embedding_matrix(config.matrix_path)
    trainingset, validationset, _ = load_data(config.data_path)
    model = ExampleModel(embedding_matrix=embedding_matrix,
                         max_len=config.max_len,
                         category_num=)






