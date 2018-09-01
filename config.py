class Config:

    matrix_path = './data/embedding_matrix.pkl'            # 词嵌入矩阵的完整路径
    char_matrix_path = './data/char_embedding_matrix.pkl'  # 字嵌入矩阵的路径
    data_path = './data/dataset.pkl'                       # 数据集文件的完整路径

    max_len = 200                                          # 文本最大长度
    max_char_len = 5                                       # 词的最大字数
    category_num = 4                                       # 单个层次的类别数
    lr = 0.001
    dropout = 0.2
    optimizer = 'RMSprop'
    loss = 'categorical_crossentropy'
    metrics = ['acc']
    monitor = 'val_loss'

    need_char_level = False                                # 是否启用字级向量
    need_summary = False                                   # 是否打印summary信息

    batch_size = 64
    epochs = 50
    verbose = 1
    model_name = None                                      # 用于预测，评估或者训练的模型名称












