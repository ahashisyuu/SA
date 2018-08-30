# SA
## 1.框架结构
###关于数据
（1）**rawData**文件夹存的是官网下的原生语料；  
（2）**data**文件夹是已经处理好的数据文件；  
（3）**preprocessing**文件夹包含数据预处理的一些文件；  
（3）要得到处理好的数据可以直接运行**preprocessing**文件夹内的preprocessing.py文件，也可以在当前命令行运行以下命令：  

    python run.py --prepro
## 2.模型训练
训练哪个模型要先记住你的模型的类的名称，模型必需是一个类，其模板在**models**文件夹内，新模型也应放在该文件夹内，假设你的模型类名为modle_name，在当前命令行运行以下命令即可训练模型：  

    python run.py --train --model_name



