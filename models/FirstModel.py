from models.ExampleModel import ExampleModel
from keras.layers import *
from layers import *


class FirstModel(ExampleModel):
    def build_model(self, doc):
        """

        :param doc: (B, max_len, dim)
        :return: self.output = 整个模型预测的概率，形状应为(B, class_num),
            class_num可由self.class_num得到
        """
        self.output = doc



