# WSDM2019 第二名
WSDM Cup 2019 Fake News Classification
1)	模型准备
提前下载Google的3个bert预训练模型于data/bert/路径下：
chinese_L-12_H-768_A-12
uncased_L-24_H-1024_A-16
uncased_L-12_H-768_A-12
2)	数据集划分
数据集划分和数据增强。直接运行脚本:
sh 0_prepare.sh
3)	第一层模型训练和预测。直接运行脚本:
sh 1_first_level_model_train_and_predict.sh
4)	第二层和第三层模型的训练和预测。直接运行脚本:
sh 2_1_second_level_and_third_level_model.sh
5)	 结果位置
result/result.csv
6)	运行时长
第一层的25个基于bert的模型训练时间较长，保守估计平均每个模型需要5个小时左右。第二三层模型运行时间大概需要一两个小时。
