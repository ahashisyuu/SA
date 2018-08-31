import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine.training import Model
from keras import backend as K
from keras.layers import Input, Dense, GRU, Reshape, Embedding


class ExampleModel:
    def __init__(self, embedding_matrix, max_len, category_num = 4,
                 dropout=0.2, optimizer='RMSprop',
                 loss='categorical_crossentropy', metrics=['acc'],
                 **kwargs):
        self.embedding_matrix = embedding_matrix  # 嵌入矩阵
        self.max_len = max_len  # 最大文档长度
        self.category_num = category_num  # 总的类别数量
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics  # 评价方法，必需是列表

        # =====  一些必要的层初始化  =====
        self.document = None
        self.embedded_doc = None
        self.output = None
        self.model = None
        # =============================

        self.creat_input()
        self.embedding_vector()
        self.modeling()  # 模型的运算主体，所有运算全部定义在这个函数下
        self.complie_model()

    def creat_input(self):
        self.document = Input(shape=[self.max_len, ], dtype='int32')
        # self.feature = Input(shape=[self.max_len,], dtype='float32')

    def embedding_vector(self):
        self.embedded_doc = Embedding(input_dim=self.embedding_matrix.shape[0],
                                      output_dim=self.embedding_matrix.shape[1],
                                      mask_zero=True,
                                      weights=self.embedding_matrix)

    def modeling(self):
        """
        模型运算主体
        输入：self.embedded_sentence: 已经嵌入的词向量序列，形状为(Batch_size, max_len, demensions)
        输出：self.output: 类别预测，为已经过softmax运算得到的概率值，形状为(Batch_size, category_num)
        """

        # self.output = Dense(self.category_num, activation='softmax')(xxx)
        pass

    def complie_model(self):
        self.model = Model(inputs=[self.document], outputs=[self.output])
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                           metrics=self.metrics)

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
            save_path = './model_' + self.model.name
            callbacks = [ModelCheckpoint(filepath=save_path, save_weights_only=True),
                         TensorBoard(write_graph=True, histogram_freq=0)]

        self.model.fit(train_data, train_label,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=validation_data,
                       callbacks=callbacks)

    def predict(self, test_data, batch_size=128, verbose=1):
        return self.model.predict(test_data, batch_size=batch_size, verbose=verbose)


















