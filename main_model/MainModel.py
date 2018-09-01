import os
import tensorflow as tf
from keras.backend import tensorflow_backend as KTF
from evaluate import categorical_prf
from utils import load_embedding_matrix, load_data


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=sess_config)

KTF.set_session(sess)


class MainModel:
    def __init__(self, cls, config):
        self.cls = cls
        self.config = config
        self.arrangement_index = load_data(self.config.map_path).get(self.config.arrangement, 0)

    def train(self):
        embedding_matrix = load_embedding_matrix(self.config.matrix_path)
        if self.config.need_char_level:
            char_embedding_matrix = load_embedding_matrix(self.config.char_matrix_path)
        else:
            char_embedding_matrix = None
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

        validationset = [validationset[0], validationset[1][self.arrangement_index]]
        model.train_model(trainingset[0], trainingset[1][self.arrangement_index],
                          batch_size=self.config.batch_size,
                          valid_batch_size=self.config.valid_batch_size,
                          epochs=self.config.epochs,
                          verbose=self.config.verbose,
                          validation_data=validationset,
                          monitor=self.config.monitor,
                          load_model_name=self.config.model_name)

    def predict(self, load_best_model=True, evaluate=False):
        embedding_matrix = load_embedding_matrix(self.config.matrix_path)
        char_embedding_matrix = load_embedding_matrix(self.config.char_matrix_path)
        _, validationset, testa = load_data(self.config.data_path)
        validationset = [validationset[0], validationset[1][self.arrangement_index]]

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

        if self.config.model_name is not None:
            model.load_weights(self.config.model_name)
        elif load_best_model is True:
            path = '../models/save_model_' + model.name
            final_name = find_best_model(path, monitor=self.config.monitor)
            print('模型全名为：%s' % final_name)
            model_path = os.path.join(path, final_name)
            model.load_weights(model_path)
        else:
            raise ValueError('需要指定模型路径，或者有已经训练好的模型')

        test_data = testa[0] if evaluate is False else validationset[0]
        model.predict_loss = False
        output_array = model.predict(test_data=test_data,
                                     pre_batch_size=self.config.pre_batch_size,
                                     verbose=self.config.verbose)
        if evaluate:
            return output_array, validationset[1]
        return output_array

    def evaluate(self, load_best_model=True):
        y_pred, y_true = self.predict(load_best_model=load_best_model, evaluate=True)
        group_prf, group_f, acc = categorical_prf(y_true, y_pred)

        info = 'PRF值：'
        print(info)
        for i, prf in enumerate(group_prf):
            info = '{0}: ({1:.4f>8}, {2:.4f>8}, {3:.4f>8})\n'.format(i, *prf)
            print(info)
        info = 'total_f1_score: {0:.4f>8}, acc: {1:.4f>8}\n'.format(group_f, acc)
        print(info)


def find_best_model(path, monitor='val_loss'):
    if os.path.exists(path) is False:
        raise NotADirectoryError('不存在该文件夹')
    list_dir = os.listdir(path)

    final_name = None
    if 'loss' in monitor:
        min_val_loss = 1
        for name in list_dir:
            num = float(name.split('_')[2][7:])
            if min_val_loss > num:
                min_val_loss = num
                final_name = name

    else:  # f值为准
        max_f1 = 0
        for name in list_dir:
            num = float(name.split('_')[-2][8:])
            if max_f1 > num:
                max_f1 = num
                final_name = name

    return final_name








