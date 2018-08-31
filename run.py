"""从main_model选择模型进行train，test，evaluate以及可视化等工作"""

import main_model
from utils import check_models


# 每次运行前检查models模块的完整性
check_models('./models')








cls = main_model.get('ExampleModel')




