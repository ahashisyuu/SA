import os

from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.engine.training import Model
from keras import backend as K
from keras.layers import Input, Dense, GRU, Reshape, Embedding, concatenate

from utils import learning_rate


class ExampleModel:
    def __init__(self, embedding_matrix, char_embedding_matrix, max_len, max_char_len, category_num=4,
                 dropout=0.2, optimizer='RMSprop',
                 loss='categorical_crossentropy', metrics=None,
                 need_char_level=False, need_summary=False,
                 **kwargs):
        self.embedding_matrix = embedding_matrix            # 词嵌入矩阵
        self.char_embedding_matrix = char_embedding_matrix  # 字嵌入矩阵
        self.max_len = max_len                              # 最大文档长度
        self.max_char_len = max_char_len                    # 词最多包含多少字
        self.category_num = category_num                    # 总的类别数量
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics                              # 评价方法，必需是列表
        self.need_char_level = need_char_level              # 是否需要中文字级
        self.need_summary = need_summary                    # 是否需要summary

        # =====  一些必要的层初始化  =====
        self.document = None
        self.embedded_doc = None
        self.doc = None
        self.model = None
        self.doc_char = None
        self.embedded_doc_char = None
        self.processed_char = None
        # =============================

        self.creat_input()
        self.embedding_vector()
        self.output = self.build_model(self.doc)  # 模型的运算主体，所有运算全部定义在这个函数下
        self.complie_model()

    def creat_input(self):
        self.document = Input(shape=[self.max_len, ], dtype='int32')
        if self.need_char_level:
            self.doc_char = Input(shape=[self.max_len, self.max_char_len], dtype='int32')
        # self.feature = Input(shape=[self.max_len,], dtype='float32')

    def embedding_vector(self):
        self.embedded_doc = Embedding(input_dim=self.embedding_matrix.shape[0],
                                      output_dim=self.embedding_matrix.shape[1],
                                      mask_zero=True,
                                      weights=self.embedding_matrix)
        if self.need_char_level:
            self.embedded_doc_char = Embedding(input_dim=self.char_embedding_matrix.shape[0],
                                               output_dim=self.char_embedding_matrix.shape[1],
                                               mask_zero=True,
                                               weights=self.embedding_matrix)
            self.processed_char = self.process_char(self.embedded_doc_char)
            self.doc = concatenate([self.embedded_doc, self.processed_char], axis=-1)
        else:
            self.doc = self.embedded_doc

    def process_char(self, embedded_doc_char):
        """
            实现字级嵌入的处理工作，以得到用于与词级嵌入拼接的输出processed_char
            输入：embedded_doc_char
            输出：processed_char

        """
        # return processed_char
        if self.need_char_level:
            raise NotImplementedError

    def build_model(self, doc):
        """
            模型运算主体
            输入：doc: 词向量序列，形状为(Batch_size, max_len, dimensions)
            输出：output: 类别预测，为经过softmax运算得到的概率值，形状为(Batch_size, category_num)

        """
        # return output
        raise NotImplementedError

    def complie_model(self):
        if self.need_char_level:
            self.model = Model(inputs=[self.document, self.doc_char], outputs=[self.output])
        else:
            self.model = Model(inputs=[self.document], outputs=[self.output])
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                           metrics=self.metrics)
        if self.need_summary:
            self.model.summary()

    def train_model(self, train_data, train_label,
                    batch_size=64, epochs=30, verbose=1,
                    validation_data=None, callbacks=None,
                    load_model_name=None):
        if os.path.exists('./model_' + self.model.name) is False:
            os.mkdir('./model_' + self.model.name)

        if load_model_name is not None:
            filepath = os.path.join('./model_' + self.model.name, load_model_name)
            self.model.load_weight(filepath)

        if callbacks is None:
            save_path = './model_' + self.model.name \
                        + '/epoch{epoch}_loss{loss}_valloss{val_loss}_f1{val_f1_score}.model'
            callbacks = [ModelCheckpoint(filepath=save_path, save_weights_only=True),
                         TensorBoard(write_graph=True, histogram_freq=0),
                         LearningRateScheduler(schedule=learning_rate)]

        self.model.fit(train_data, train_label,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=validation_data,
                       callbacks=callbacks)

    def predict(self, test_data, batch_size=128, verbose=1):
        return self.model.predict(test_data, batch_size=batch_size, verbose=verbose)

    def load_weights(self, filepath):
        self.model.load_weight(filepath)

    @property
    def name(self):
        return self.model.name



















