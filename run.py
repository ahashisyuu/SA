"""从main_model选择模型进行train，test，evaluate以及可视化等工作"""
import os

import main_model
import argparse
from main_model.MainModel import MainModel
from config import Config
from preprocessing.preprocessing import preprocessing
from utils import check_models, check_layers, save_results, load_data

config = Config()

# 每次运行前检查models和layers模块的完整性
check_models('./models')
check_layers('./layers')


def parse_args():
    parser = argparse.ArgumentParser('Fine-grained Sentimental Analysis')
    parser.add_argument("--mode", default='train',
                        choices=['train', 'prepare', 'predict', 'evaluate'])
    parser.add_argument("--model", type=str, default='FirstModel', help='class name of model')
    parser.add_argument('--load_best_model', action='store_true')

    # 路径相关
    parser.add_argument('--matrix_path', default=os.path.abspath(config.matrix_path))
    parser.add_argument('--char_matrix_path', default=os.path.abspath(config.char_matrix_path))
    parser.add_argument('--data_path', default=os.path.abspath(config.data_path))
    parser.add_argument('--map_path', default=os.path.abspath(config.map_path))

    parser.add_argument('--arrangement', type=str, default=config.arrangement)
    parser.add_argument('--max_len', type=int, default=config.max_len)
    parser.add_argument('--max_char_len', type=int, default=config.max_char_len)
    parser.add_argument('--category_num', type=int, default=config.category_num)
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument('--dropout', type=float, default=config.dropout)
    parser.add_argument('--optimizer', type=str, default=config.optimizer)
    parser.add_argument('--loss', type=str, default=config.loss)
    parser.add_argument('--metrics', type=list, default=config.metrics)
    parser.add_argument('--monitor', type=str, default=config.monitor)

    parser.add_argument('--need_char_level', type=bool, default=config.need_char_level)
    parser.add_argument('--need_summary', type=bool, default=config.need_summary)
    parser.add_argument('--vector_trainable', type=bool, default=config.vector_trainable)

    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--valid_batch_size', type=int, default=config.valid_batch_size)
    parser.add_argument('--epochs', type=int, default=config.epochs)
    parser.add_argument('--verbose', type=int, default=config.verbose)
    parser.add_argument('--model_name', type=str, default=config.model_name)

    return parser.parse_args()


def run(args):
    if args.mode == 'prepare':
        preprocessing(args)
    else:
        cls = main_model.get(args.model)
        if cls is None:
            return
        model = MainModel(cls=cls, config=args)
        print('--------------------------------------------------------------')
        print('        本次对应的层次为: %s' % args.arrangement)
        print('--------------------------------------------------------------')
        if args.mode == 'train':
            model.train()
        elif args.mode == 'predict':
            results = model.predict(load_best_model=args.load_best_model)
            save_results(results, args)
        else:
            model.evaluate(load_best_model=args.load_best_model)


if __name__ == '__main__':
    run(parse_args())

