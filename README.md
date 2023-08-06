# CSHP
一种新的小样本图像分类任务PEFT方法（基于CLIP，可扩展为其他）
# 项目依赖
CoOp：https://github.com/KaiyangZhou/CoOp
dassl:https://github.com/KaiyangZhou/Dassl.pytorch
# 训练及测试方法
1、下载dassl库
2、复制CoOp项目，将本项目manualprompt文件夹放于根目录下、cshp.py文件放于trainer目录下、cshp文件夹放于scripts目录下


训练：sh scripts/cshp/train.sh 
测试：sh scripts/cshp/train.sh 
