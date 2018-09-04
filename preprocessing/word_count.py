import numpy as np


def word_count(doc_list, dic_count):
    # 计算词频
    # 如果字典里有该单词则加1，否则添加入字典

    for str in doc_list:
        if str in dic_count.keys():
            dic_count[str] = dic_count[str] + 1
        else:
            dic_count[str] = 1

    return dic_count


def get_vec_dic(vec_list):
    # 获得词向量字典
    vec_dic = {}
    count = 1
    for line in vec_list:
        if count == 1:
            count += 1
            continue
        values = line.split()
        vec_dic[values[0]] = np.asarray(values[1:], dtype='float32')
    return vec_dic


def process_dic(dic_count, vec_dic, max_cnt=1500000, min_cnt=0):
    # 按照词频从高到低排列，并表示unknown字符
    count_list = sorted(dic_count.items(), key=lambda x: x[1], reverse=True)
    count_dic_sorted = {}
    split_word = []
    for line in count_list:
        if line[1] >= max_cnt or line[1] <= min_cnt:
            split_word.append(line[0])
            continue
        count_dic_sorted[line[0]] = line[1]

    new_dic = {}
    sort_num = 1
    for key, value in count_dic_sorted.items():
        if key in vec_dic:
            new_dic[key] = sort_num
            sort_num += 1
    new_dic['<unk>'] = sort_num

    return new_dic, split_word


def ConvertToENG(doc_list, count_dic_sorted, drop_word):
    # 将文章中的单词转为字典序，返回列表
    doc_num_list = []
    unknown_num = len(count_dic_sorted)
    for word in doc_list:
        if word in count_dic_sorted:
            doc_num_list.append(count_dic_sorted.get(word))
        elif word in drop_word:
            continue
        else:
            doc_num_list.append(unknown_num)

    return doc_num_list

