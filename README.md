# SA
## 1.框架结构
### 关于数据
（1）**rawData**文件夹存的是官网下的原生语料；  
（2）**data**文件夹是已经处理好的数据文件；  
（3）**preprocessing**文件夹包含数据预处理的一些文件；  
（3）要得到处理好的数据可以直接运行**preprocessing**文件夹内的preprocessing.py文件,或者在当前命令行执行以下指令：  

    python run.py --mode prepare   

### 关于模型文件
继承**models**文件夹内的**ExampleModel**类，重写**build_model**函数即可；  
该类文件应当像如下导入（**models**文件夹内的**FirstModel**有完整导入格式，可参考）：  

    from models.ExampleModel import ExampleModel
    from keras.layers import *
    from layers import *  
    
## 2.模型训练
训练哪个模型要先记住你的模型的类的名称，模型必需是一个类，其模板在**models**文件夹内，新模型也应放在该文件夹内，假设你的模型类名为class_name，在当前命令行运行以下命令即可训练模型：  

    python run.py --mode train --model class_name

注意，默认超参数可以在config.py里改，也可在命令行直接指定，比如要改学习率：  

    python run.py --mode train --model class_name --lr 0.001

**mode**有四种，分别为prepare, train, predict, evaluate, 默认是train  
另外想要指定模型继续训练或者预测结果，可以把模型文件名作为参数传给命令行：  

    python run.py --mode train --model class_name --model_name filaname   



