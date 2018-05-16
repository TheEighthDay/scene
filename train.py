#coding:utf-8
import os
import glob
# import matplotlib.pyplot as plt

from keras import __version__
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input,Concatenate,Dropout,GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt
# get_nb_files('F:/文件/scene_classification/image_scene_training/train')

IM_WIDTH, IM_HEIGHT = 299, 299 #InceptionV3指定的图片尺寸
FC_SIZE = 1024                # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 400  # 冻结层的数量


train_dir = 'image_scene_training/train'  # 训练集数据
val_dir = 'image_scene_training/validation' # 验证集数据
nb_classes= 20
nb_epoch = 10
batch_size = 48

nb_train_samples = get_nb_files(train_dir)      # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)       #验证集样本个数
nb_epoch = int(nb_epoch)                # epoch数量
batch_size = int(batch_size)

print("训练样本个数"+str(nb_train_samples)) 
print("分类数"+str(nb_classes))
print("验证集样本个数"+str(nb_val_samples))




#　图片生成器
train_datagen =  ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True
)
test_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True
)

# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(IM_WIDTH, IM_HEIGHT),batch_size=batch_size,class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(val_dir,target_size=(IM_WIDTH, IM_HEIGHT),batch_size=batch_size,class_mode='categorical')

def add_new_last_layer(base_model, nb_classes):
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
	x = Dropout(0.3)(x)
	predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
	model = Model(input=base_model.input, output=predictions)
	return model


def setup_to_finetune(model):
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    return model

# 冻上base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_transfer_learn(model):
    """Freeze all layers and compile the model"""
    for layer in model.layers:
        layer.trainable = False
    print('layers='+str(len(model.layers)))
    return model

# 定义网络框架
def inception():
    base_model = InceptionV3(weights=None, include_top=False) # 预先要下载no_top模型
    print("loading inceptionv3")
    base_model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")  
    print("done")
    model = setup_to_finetune(base_model)
    model = add_new_last_layer(model, nb_classes)              # 从基本no_top模型上添加新层
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def inception_resnet():
    base_model = InceptionResNetV2(weights=None, include_top=False) # 预先要下载no_top模型
    print("loading inception_resnet_v2")
    base_model.load_weights("inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5")  
    print("done")
    model = setup_to_finetune(base_model)
    model = add_new_last_layer(model,nb_classes)              # 从基本no_top模型上添加新层
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# def fuse():效果不好
#     input_layer=Input(shape=(299,299,3)) 

#     inceptionv3 = InceptionV3(weights=None, include_top=False,input_shape=(299,299,3)) # 预先要下载no_top模型
#     inceptionresnetv2= InceptionResNetV2(weights=None, include_top=False,input_shape=(299,299,3)) # 预先要下载no_top模型
#     freeze_A=300
#     freeze_B=700

#     for layer in inceptionv3.layers[:freeze_A]:
#         layer.trainable = False
#     for layer in inceptionv3.layers[freeze_A:]:
#         layer.trainable = True

#     for layer in inceptionresnetv2.layers[:freeze_B]:
#         layer.trainable = False
#     for layer in inceptionresnetv2.layers[freeze_B:]:
#         layer.trainable = True

#     print("loading inceptionv3")
#     inceptionv3.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")  
#     print("done") 
#     print("loading inception_resnet_v2")
#     inceptionresnetv2.load_weights("inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5")  
#     print("done")

#     inceptionv3=inceptionv3(input_layer)
#     inceptionresnetv2=inceptionresnetv2(input_layer)

#     top1_model=GlobalMaxPooling2D(data_format='channels_last')(inceptionv3)  
#     top2_model=GlobalMaxPooling2D(data_format='channels_last')(inceptionresnetv2)

#     t=Concatenate(axis=1)([top1_model,top2_model])
#     top_model=Dense(1024,activation="relu")(t)  
#     top_model=Dropout(rate=0.5)(top_model)  
#     top_model=Dense(20,activation="softmax")(top_model)
#     model=Model(input=input_layer, output=top_model)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model











model=inception_resnet()

model_checkpoint = ModelCheckpoint('model_new.h5', monitor='val_loss',verbose=1, save_best_only=True)
# 模式一训练
history_tl = model.fit_generator(
train_generator,
nb_epoch=nb_epoch,
steps_per_epoch=200,
validation_data=validation_generator,
nb_val_samples=20,
class_weight='auto',
shuffle=True,
callbacks=[model_checkpoint]
)

# def plot_training(history):
#   acc = history.history['acc']
#   val_acc = history.history['val_acc']
#   loss = history.history['loss']
#   val_loss = history.history['val_loss']
#   epochs = range(len(acc))
#   plt.plot(epochs, acc, 'r.')
#   plt.plot(epochs, val_acc, 'r')
#   plt.title('Training and validation accuracy')
#   plt.figure()
#   plt.plot(epochs, loss, 'r.')
#   plt.plot(epochs, val_loss, 'r-')
#   plt.title('Training and validation loss')
#   plt.savefig("examples.jpg") 

# # 训练的acc_loss图
# plot_training(history_tl)