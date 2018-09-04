# 初始化unknown字符序号
unknown_num = 0

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
    for line in vec_list:
        vec_dic[line[0]] = line[1:]
    return vec_dic

def process_dic(dic_count, vec_dic):
    # 按照词频从高到低排列，并表示unknown字符
    count_list = sorted(dic_count.iteritems(), key=lambda x:x[1], reverse=True)
    count_dic_sorted = {}
    for line in count_list:
        count_dic_sorted[line[0]] = line[1]
    sort_num = 0

    for key, value in count_dic_sorted.items():
        if value != 1 and value != 2:
            count_dic_sorted[key] = sort_num
            sort_num += 1
    unknown_num = sort_num + 1

    for key, value in count_dic_sorted.items():
        if value == 1 or value == 2 or vec_dic.has_key(key) == False:
            count_dic_sorted[key] = unknown_num

    return count_dic_sorted

def ConvertToENG(doc_list, count_dic_sorted):
    # 将文章中的单词转为字典序，返回列表
    doc_num_list = []

    for word in doc_list:
        if count_dic_sorted.has_key(word):
            doc_num_list.append(count_dic_sorted.get(word))
        else:
            doc_num_list.append(unknown_num)
    doc_num_list.append("\n")

    return doc_num_list

