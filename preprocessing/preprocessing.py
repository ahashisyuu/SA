import jieba
import pickle as pkl
import pandas as pd
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences

from preprocessing.word_count import word_count, get_vec_dic, process_dic, ConvertToENG


def load_rawData(data_path):
    trainingset = None
    validationset = None
    testa = None
    listdir = os.listdir(data_path)
    for name in listdir:
        if 'trainingset.csv' in name:
            trainingset = pd.read_csv(os.path.join(data_path, name))
        if 'validationset.csv' in name:
            validationset = pd.read_csv(os.path.join(data_path, name))
        if 'testa.csv' in name:
            testa = pd.read_csv(os.path.join(data_path, name))
    return trainingset, validationset, testa


def one_hot(number):
    li = [0, 0, 0, 0]
    li[number] = 1
    return li


def one_hot_series(series):
    return series.apply(one_hot)


def tokenize(content, filters='！!“”"#$%&（）()*+,，-。、./:：；;‘’《》……·<=>?@[\\]^_`{|}~\t\n'):
    return [token for token in jieba.cut(content[1:-1]) if token not in filters]


def tokenizor():
    old_path = './rawData'
    new_path = './data'

    if not os.path.exists(old_path):
        os.mkdir(old_path)
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    trainingset, validationset, testa = load_rawData(old_path)

    # 对有标签的数据进行处理
    trainingset.iloc[:, 2:] = trainingset.iloc[:, 2:].apply(lambda x: x + 2)
    validationset.iloc[:, 2:] = validationset.iloc[:, 2:].apply(lambda x: x + 2)

    trainingset.iloc[:, 2:] = trainingset.iloc[:, 2:].apply(one_hot_series)
    validationset.iloc[:, 2:] = validationset.iloc[:, 2:].apply(one_hot_series)

    trainingset['content'] = trainingset['content'].apply(tokenize)
    validationset['content'] = validationset['content'].apply(tokenize)
    testa['content'] = testa['content'].apply(tokenize)

    # trainingset.to_csv(os.path.join(new_path, 'trainingset.csv'), index=False)
    # validationset.to_csv(os.path.join(new_path, 'validationset.csv'), index=False)
    # testa.to_csv(os.path.join(new_path, 'testa.csv'), index=False)
    return trainingset, validationset, testa


def make_input_list(data_frame, max_len=314):
    """

    :param data_frame: 为pandas的Series对象
    :return: numpy对象列表
    """
    input_list = [text for text in data_frame.values]
    input_list = pad_sequences(input_list, maxlen=max_len, padding='post', truncating='post')
    return [input_list]


def make_output(data_frame):
    return np.asarray([label for label in data_frame.values])


def make_output_list(data_frame, arrangement_map):
    output_list = []
    for key, value in arrangement_map.items():
        output_list.append(make_output(data_frame[key]))
    return output_list


def preprocessing(args):
    """
        整合成如下纯列表形式：[input_list, output_list],
        input_list为输入列表，每一项为numpy矩阵的输入；
        output_list为标记列表，每一项为单个层次对应的类别标记numpy矩阵, test数据output_list为空列表。
        为了方便分类器选择，要生成一个如{‘层次名’: 序号}的字典，方便训练

    """
    print('-----  开始分词  -----')
    trainingset, validationset, testa = tokenizor()   # pandas
    print('-----  分词结束  -----')
    print('---------------------')

    # get arrangement_map
    print('start getting arrangement_map!!!\n')
    keys_list = trainingset.keys().tolist()[2:]
    arrangement_map = {key: i for i, key in enumerate(keys_list)}
    map_name = os.path.join('./data', 'arrangement_map.pkl')
    keys_list_name = os.path.join('./data', 'keys_list.txt')
    with open(map_name, 'wb') as fw, open(keys_list_name, 'w') as fw2:
        pkl.dump(arrangement_map, fw)

        for key in keys_list:
            fw2.write(key + '\n')

    count_dic = {}
    trainingset['content'].apply(word_count, args=(count_dic,))
    validationset['content'].apply(word_count, args=(count_dic,))

    # get vector dict
    print('start getting vector dict\n')
    word_vector_name = os.path.join('./rawData', 'sgns.merge.word')
    with open(word_vector_name, 'r', encoding='utf-8') as fr:
        vec_dic = get_vec_dic(fr)

    # get word2index
    print('start getting word2index!!!\n')
    word2index, drop_word = process_dic(dic_count=count_dic, vec_dic=vec_dic)
    word2index_name = os.path.join('./data', 'word2index.pkl')
    with open(word2index_name, 'wb') as fw:
        pkl.dump([word2index, drop_word], fw)

    # get embedding matrix
    print('start getting embedding matrix\n')
    embedding_matrix = np.random.randn(len(word2index) + 1, 300)
    for key, index in word2index.items():
        if key == '<unk>':
            continue
        embedding_matrix[index] = vec_dic[key]
    embedding_matrix[0] = np.zeros((300,))

    embedding_matrix_name = os.path.join('./data', 'embedding_matrix.pkl')
    with open(embedding_matrix_name, 'wb') as fw:
        pkl.dump(embedding_matrix, fw)

    # replace data
    print('start translating word to index!!!\n')
    trainingset['content'] = trainingset['content'].apply(ConvertToENG, args=(word2index, drop_word))
    validationset['content'] = validationset['content'].apply(ConvertToENG, args=(word2index, drop_word))
    testa['content'] = testa['content'].apply(ConvertToENG, args=(word2index, drop_word))

    # make arrangement input
    print('start making arrangement input\n')
    keys = []
    for key in keys_list:
        key_ = []
        for word in key.split('_'):
            if word in word2index:
                key_.append(word2index[word])
            else:
                key_.append(word2index['<unk>'])

        while len(key_) < 5:
            key_.append(0)
        keys.append(key_)

    train_keys = []
    val_keys = []
    testa_keys = []
    for key in keys:
        train_keys.append(np.asarray([key]*len(trainingset['content'])))
        val_keys.append(np.asarray([key]*len(validationset['content'])))
        testa_keys.append(np.asarray([key]*len(testa['content'])))

    print('start saving overall data\n')
    trainingset_data = [make_input_list(trainingset['content'], max_len=args.max_len) + train_keys,
                        make_output_list(trainingset.iloc[:, 2:], arrangement_map)]
    validationset_data = [make_input_list(validationset['content'], max_len=args.max_len) + val_keys,
                          make_output_list(validationset.iloc[:, 2:], arrangement_map)]
    testa_data = [make_input_list(testa['content'], max_len=args.max_len) + testa_keys, []]

    with open(os.path.join('./data', 'dataset.pkl'), 'wb') as fw:
        pkl.dump([trainingset_data, validationset_data, testa_data], fw)

    print('saving data successfully')

    return None

