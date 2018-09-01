import jieba
import pandas as pd
import os


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
            testa.to_csv()
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
    old_path = '../rawData'
    new_path = '../data'

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

    trainingset.to_csv(os.path.join(new_path, 'trainingset.csv'))
    validationset.to_csv(os.path.join(new_path, 'validationset.csv'))
    testa.to_csv(os.path.join(new_path, 'testa.csv'))


def preprocessing(args):
    return None

