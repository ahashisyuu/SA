from utils import load_embedding_matrix, load_data
from models.ExampleModel import ExampleModel


def train(config):
    embedding_matrix = load_embedding_matrix(config.matrix_path)
    trainingset, validationset, _ = load_data(config.data_path)
    model = ExampleModel(embedding_matrix=embedding_matrix,
                         max_len=config.max_len,
                         category_num=config.category_num,
                         dropout=config.dropout,
                         optimizer=config.optimizer,
                         loss=config.loss,
                         metrics=config.metrics)
    model.train_model(trainingset[0], trainingset[1],
                      batch_size=config.batch_size,
                      epochs=config.epoch,
                      verbose=config.verbose,
                      validation_data=validationset)


def predict(config, load_best_model=True, model_path=None):
    embedding_matrix = load_embedding_matrix(config.matrix_path)
    _, _, testa = load_data(config.data_path)
    model = ExampleModel(embedding_matrix=embedding_matrix,
                         max_len=config.max_len,
                         category_num=config.category_num,
                         dropout=config.dropout,
                         optimizer=config.optimizer,
                         loss=config.loss,
                         metrics=config.metrics)

    if model_path is not None:
        model.load_weights(model_path)
    elif load_best_model is True:
        path = '../'








