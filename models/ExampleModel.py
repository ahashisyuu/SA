import os
import keras.backend as K
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.engine import Model
from keras.layers import Input, Dense, GRU, Reshape, Embedding, concatenate

from utils import learning_rate, PRFAcc


class ExampleModel:
    def __init__(self, embedding_matrix, char_embedding_matrix, max_len, max_char_len, category_num=4,
                 dropout=0.2, optimizer='RMSprop', arrangement_index=0,
                 loss='categorical_crossentropy', metrics=None,
                 need_char_level=False, need_summary=False, vector_trainable=False,
                 **kwargs):
        self.embedding_matrix = embedding_matrix            # 词嵌入矩阵
        self.char_embedding_matrix = char_embedding_matrix  # 字嵌入矩阵
        self.max_len = max_len                              # 最大文档长度
        self.max_char_len = max_char_len                    # 词最多包含多少字
        self.class_num = category_num                    # 总的类别数量
        self.dropout = dropout
        self.optimizer = optimizer
        self.arrangement_index = arrangement_index
        self.loss = loss
        self.metrics = metrics                              # 评价方法，必需是列表
        self.need_char_level = need_char_level              # 是否需要中文字级
        self.need_summary = need_summary                    # 是否需要summary
        self.trainable = vector_trainable                   # 词向量是否可训练

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
                                      weights=[self.embedding_matrix],
                                      trainable=self.trainable)(self.document)
        if self.need_char_level:
            self.embedded_doc_char = Embedding(input_dim=self.char_embedding_matrix.shape[0],
                                               output_dim=self.char_embedding_matrix.shape[1],
                                               mask_zero=True,
                                               weights=[self.embedding_matrix],
                                               trainable=self.trainable)(self.doc_char)
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
            输出：output: 类别预测，为经过softmax运算得到的概率值，形状为(Batch_size, class_num)

        """
        # return output
        raise NotImplementedError

    def complie_model(self):
        output_shape = K.int_shape(self.output)
        assert len(output_shape) == 2 and output_shape[1] == self.class_num, \
            'output的形状必需是（B, class_num），但得到的是：（{0}， {1}）'.format(*output_shape)
        if self.need_char_level:
            self.model = Model(inputs=[self.document, self.doc_char],
                               outputs=[self.output])
        else:
            self.model = Model(inputs=[self.document], outputs=[self.output])
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                           metrics=self.metrics)

        if self.need_summary:
            self.model.summary()

    def train_model(self, train_data, train_label, arrangement,
                    batch_size=64, valid_batch_size=128, epochs=30, verbose=1,
                    validation_data=None, callbacks=None, monitor='val_loss',
                    load_model_name=None):
        path_root = './models/save_model_' + self.__class__.__name__
        if os.path.exists(path_root) is False:
            os.mkdir(path_root)
        path = os.path.join(path_root, arrangement)
        if os.path.exists(path) is False:
            os.mkdir(path)

        if load_model_name is not None:
            filepath = os.path.join(path, load_model_name)
            self.model.load_weight(filepath)

        if callbacks is None:
            save_path = path \
                        + '/epoch{epoch}_loss{loss:.4f}_valloss{val_loss:.4f}' \
                          '_fmeasure{fmeasure:.4f}_valacc{val_acc:.4f}.model'
            callbacks = [TensorBoard(write_graph=True, histogram_freq=0),
                         LearningRateScheduler(schedule=learning_rate),
                         PRFAcc(filepath=save_path, monitor=monitor,
                                batch_size=valid_batch_size,
                                arrangement_index=self.arrangement_index,
                                validation_data=validation_data)]

        self.model.fit(train_data, train_label,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=None,
                       callbacks=callbacks)

    def predict(self, test_data, pre_batch_size=128, verbose=1):
        return self.model.predict(test_data, batch_size=pre_batch_size, verbose=verbose)

    def load_weights(self, filepath):
        self.model.load_weight(filepath)

    @property
    def name(self):
        return self.__class__.__name__




















