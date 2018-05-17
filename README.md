# 程序简介
一个基于深度学习的二十个场景分类项目。  
场景类型参看image_scene_training/categories.csv。


# 系统要求
* Ubutun16.04
* 所以依赖的第三方程序和库及其版本信息见requirements.txt
* 运行在cuda6.0版本驱动
* 主要深度学习库tensorflow 1.4，keras 2.1.3
# 部署和运行方法说明

## 目录情况如下  
/image_scene_training  
&nbsp;&nbsp;&nbsp;/data  
&nbsp;&nbsp;&nbsp;/train  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/0  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/...  
&nbsp;&nbsp;&nbsp;/validation  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/0  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/...  
&nbsp;&nbsp;&nbsp;/categories.csv  
&nbsp;&nbsp;&nbsp;/list.csv  
&nbsp;&nbsp;&nbsp;/pic-license.csv  
/image_scene_test_b_0515  
&nbsp;&nbsp;&nbsp;/data  
&nbsp;&nbsp;&nbsp;/categories.csv  
&nbsp;&nbsp;&nbsp;/list.csv  
&nbsp;&nbsp;&nbsp;/pic-license.csv  
/__init__.py  
/train.py  
/train_continue.py  
/preprocess.py  
/predict.py  
/validation_top3.py  
/validation_top3_fuse.py  
/test_write_csv.py  
/一些参数设定.txt  
/ending.csv  
/model.h5  
/model_new.h5  
/model_new2.h5    
/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5  
/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5  
/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5  
  
## 运行方法详解
### 1. 运行preprocess.py将对/image_scene_training/data中的图片进行处理，生成训练集和验证集图片在/image_scene_training/train和/image_scene_training/validation目录下所对应的类别文件夹0到19，对于某类图片数量少则根据此类原有图片进行上下或者左右反转生成一定数量新图片。  
### 2. 运行train.py进行训练，注意对于不同模型的训练，model=inception()需将函数inception（）改成对应网络的函数,具体参数的设定参看&nbsp;&nbsp;一些参数设定.txt。
### 3. 运行train_continue.py 将学习器改为SGD，降低lr继续训练。
### 4. 运行predict.py 进行预测并生成ending.csv
### 5. h5及图片文件夹，文件下载地址为：https://pan.baidu.com/s/14Eu0PIRQJi3MX4ncaOnkDQ 密码rzil




  


