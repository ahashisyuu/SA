from models.ExampleModel import ExampleModel
from keras.layers import *
from layers import *


class NormalAttentionModel(ExampleModel):
    def build_model(self, doc):
        """

        :param doc: (B, max_len, dim)
        :return: output， 整个模型预测的概率，形状应为(B, class_num),
            class_num可由self.class_num得到
        """
        gru_units = 256   # 暂时未将该参数加入默认值内，有需要再加入
        act_func = 'tanh'

        encoding_doc = Bidirectional(GRU(gru_units, activation=act_func,
                                         return_sequences=True))(doc)
        encoding_doc, forward_s, backward_s = Bidirectional(GRU(gru_units, activation=act_func,
                                                                return_sequences=True, return_state=True))(encoding_doc)   # (B, 2*dim)
        s = concatenate([forward_s, backward_s])

        vector = Attention(activation='tanh')([encoding_doc, s])
        output = Dense(units=self.class_num, activation='softmax')(vector)
        return output
