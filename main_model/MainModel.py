import os

from utils import load_embedding_matrix, load_data


class MainModel:
    def __init__(self, cls, config):
        self.cls = cls
        self.config = config

    def train(self):
        embedding_matrix = load_embedding_matrix(self.config.matrix_path)
        char_embedding_matrix = load_embedding_matrix(self.config.char_matrix_path)
        trainingset, validationset, _ = load_data(self.config.data_path)

        model = self.cls(embedding_matrix=embedding_matrix,
                         char_embedding_matrix=char_embedding_matrix,
                         max_len=self.config.max_len,
                         max_char_len=self.config.max_char_len,
                         category_num=self.config.category_num,
                         dropout=self.config.dropout,
                         optimizer=self.config.optimizer,
                         loss=self.config.loss,
                         metrics=self.config.metrics,
                         need_char_level=self.config.need_char_level,
                         need_summary=self.config.need_summary)

        model.train_model(trainingset[0], trainingset[1],
                          batch_size=self.config.batch_size,
                          epochs=self.config.epochs,
                          verbose=self.config.verbose,
                          validation_data=validationset,
                          load_model_name=self.config.last_model)

    def predict(self, load_best_model=True, model_path=None):
        embedding_matrix = load_embedding_matrix(self.config.matrix_path)
        char_embedding_matrix = load_embedding_matrix(self.config.char_matrix_path)
        _, _, testa = load_data(self.config.data_path)

        model = self.cls(embedding_matrix=embedding_matrix,
                         char_embedding_matrix=char_embedding_matrix,
                         max_len=self.config.max_len,
                         max_char_len=self.config.max_char_len,
                         category_num=self.config.category_num,
                         dropout=self.config.dropout,
                         optimizer=self.config.optimizer,
                         loss=self.config.loss,
                         metrics=self.config.metrics,
                         need_char_level=self.config.need_char_level,
                         need_summary=self.config.need_summary)

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

    def evaluate(self, load_best_model=True, model_path=None):
        output_array = self.predict(load_best_model=load_best_model, model_path=model_path)







