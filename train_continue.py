#coding:utf-8
import os
import glob


from keras import __version__
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
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


IM_WIDTH, IM_HEIGHT = 299, 299 #InceptionV3指定的图片尺寸
FC_SIZE = 1024                # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 450  # 冻结层的数量


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


learning_rate = 1e-4
decay_rate = learning_rate / nb_epoch
momentum = 0.8


model=load_model("model_new2.h5")
optimizer=SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('model_new2.h5', monitor='val_loss',verbose=1, save_best_only=True)

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

