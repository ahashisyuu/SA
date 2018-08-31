import os

from utils import load_embedding_matrix, load_data


class MainModel:
    def __init__(self, cls, config):
        self.cls = cls
        self.config = config

    def train(self):
        embedding_matrix = load_embedding_matrix(self.config.matrix_path)
        trainingset, validationset, _ = load_data(self.config.data_path)
        model = self.cls(embedding_matrix=embedding_matrix,
                         max_len=self.config.max_len,
                         category_num=self.config.category_num,
                         dropout=self.config.dropout,
                         optimizer=self.config.optimizer,
                         loss=self.config.loss,
                         metrics=self.config.metrics)
        model.train_model(trainingset[0], trainingset[1],
                          batch_size=self.config.batch_size,
                          epochs=self.config.epoch,
                          verbose=self.config.verbose,
                          validation_data=validationset)

    def predict(self, load_best_model=True, model_path=None):
        embedding_matrix = load_embedding_matrix(self.config.matrix_path)
        _, _, testa = load_data(self.config.data_path)
        model = self.cls(embedding_matrix=embedding_matrix,
                         max_len=self.config.max_len,
                         category_num=self.config.category_num,
                         dropout=self.config.dropout,
                         optimizer=self.config.optimizer,
                         loss=self.config.loss,
                         metrics=self.config.metrics)

        if model_path is not None:
            model.load_weights(model_path)
        elif load_best_model is True:
            path = '../models/model_' + model.name
            if os.path.exists(path) is False:
                raise NotADirectoryError('不存在该文件夹')
            list_dir = os.listdir(path)

            min_val_loss = 1
            final_name = None
            for name in list_dir:
                num = float(name.split('_')[2][7:])
                if min_val_loss > num:
                    min_val_loss = num
                    final_name = name

            model_path = os.path.join(path, final_name)
            model.load_weights(model_path)
        else:
            raise ValueError('需要指定模型路径，或者有已经训练好的模型')

        output_array = model.predict(test_data=testa,
                                     batch_size=self.config.batch_size,
                                     verbose=self.config.verbose)

        return output_array






