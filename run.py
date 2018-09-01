"""从main_model选择模型进行train，test，evaluate以及可视化等工作"""

import main_model
from config import Config
from utils import check_models, check_layers

# 每次运行前检查models和layers模块的完整性
check_models('./models')
check_layers('./layers')


cls = main_model.get('ExampleModel')




