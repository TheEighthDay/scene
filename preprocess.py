#coding:utf-8

import numpy as np
import tensorflow as tf
import cv2
import csv, random
from os import path
from tqdm import tqdm

NUM_TRAIN=15611
NUM_VALIDATION=1000
FILENAME_V="validation_224.tfrecords"
FILENAME="train_224.tfrecords"

def read_csvlist(listfile):  #读取csv文件
    print("reading  "+listfile)
    csv_reader = csv.reader(open(listfile))
    is_first_line = True
    results = []
    for row in csv_reader:
        # Skip the head line.
        if (is_first_line):
            is_first_line = False
            continue
        results.append(row)
    print("done")
    return results
#[['2b2b344f-1c85-11e8-aaf5-00163e025669', '0'], ['1dcd5677-1c83-11e8-aaf2-00163e025669', '0']]
def split_dir(csvlist):
    summ=(len(csvlist))
    print(summ)
    listt=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(summ):
        listt[int(csvlist[i][1])].append(csvlist[i][0])
    
    for i in range(20):  #乱序
        random.shuffle(listt[i])

    for i in range(20):
        print(listt[i])

    for table in range(20):
        if len(listt[table])>700:
            for x in tqdm(listt[table]):
                img=cv2.imread('image_scene_training/data/'+ str(x) + '.jpg')
                cv2.imwrite("image_scene_training/train/"+str(table)+"/"+str(x) + '.jpg',img)
        else:
            for x in tqdm(listt[table]):
                img=cv2.imread('image_scene_training/data/'+ str(x) + '.jpg')
                cv2.imwrite("image_scene_training/train/"+str(table)+"/"+str(x) + '.jpg',img)
            need=700-len(listt[table])
            gen_picture(need,listt[table],table)


def gen_picture(need,filepath,table):
    for i in range(need):
        img=cv2.imread('image_scene_training/data/'+filepath[np.random.randint(len(filepath))]+'.jpg')
        img=cv2.flip(img, np.random.randint(3))
        cv2.imwrite("image_scene_training/train/"+str(table)+"/"+str(i) + 'add.jpg',img)




    # for i in tqdm(range(len(csvlist)-1000)):
    #     img=cv2.imread('image_scene_training/data/'+ csvlist[i][0] + '.jpg')
    #     cv2.imwrite("image_scene_training/train/"+csvlist[i][1]+"/"+csvlist[i][0] + '.jpg',img)
    # for i in range(len(csvlist)-1000,len(csvlist)):
    #     img=cv2.imread('image_scene_training/data/'+ csvlist[i][0] + '.jpg')
    #     cv2.imwrite("image_scene_training/validation/"+csvlist[i][1]+"/"+csvlist[i][0] + '.jpg',img)

def one_hot(labels):  #onehot编码
    sess = tf.Session()
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels],1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 20]), 1, 0)
    return sess.run(onehot_labels)

def load_train_data(csv_results): #将csv读取信息转化为图片地址和label
    random.shuffle(csv_results)  #乱序
    train_filepaths=[]
    train_labels=[]
    validation_filepaths=[]
    validation_labels=[]

    print('Reading train data_info...')
    for i in range(NUM_TRAIN):
        file_id, category_id = csv_results[i]
        file_path = 'image_scene_training/data/'+ file_id + '.jpg'
        train_filepaths.append(file_path)
        train_labels.append(int(category_id))
    print('Done')
    print('Reading validation data_info...')
    for i in range(NUM_TRAIN,NUM_TRAIN+NUM_VALIDATION):
        file_id, category_id = csv_results[i]
        file_path = 'image_scene_training/data/'+ file_id + '.jpg'
        validation_filepaths.append(file_path)
        validation_labels.append(int(category_id))
    print('Done')

    return (train_filepaths,train_labels,validation_filepaths,validation_labels)

def gen_tfrecords(filepaths,labels,tffilename):  #生成tfrecords文件
    writer = tf.python_io.TFRecordWriter(tffilename)
    print('Creating '+tffilename)
    labels=one_hot(labels)
    for i in tqdm(range(len(filepaths))):
        image=cv2.imread(filepaths[i])
        image=cv2.resize(image,(299,299))
        # print(type(image[0][0][0]),type(labels[i][0]))
        image_raw = image.tostring()
        label_raw = labels[i].tostring()
        # 将一个样例转化为Example Protocol Buffer，并将所有信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        # 将一个Example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()    
    print("done")

def is_exist():  #判断是否有tfrecords文件，没有则生成
    if not path.exists(FILENAME) or not path.exists(FILENAME_V):
        print('File does not exist.')
        csvlist=read_csvlist("image_scene_training/list.csv")
        train_filepaths,train_labels,validation_filepaths,validation_labels=load_train_data(csvlist)
        gen_tfrecords(train_filepaths,train_labels,FILENAME)
        gen_tfrecords(validation_filepaths,validation_labels,FILENAME_V)

def distort_color(image, color_ordering=0):  #图片增强
    if color_ordering == 0:  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)#亮度  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)#饱和度  
        image = tf.image.random_hue(image, max_delta=0.2)#色相  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)#对比度  
    if color_ordering == 1:  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  
        image = tf.image.random_hue(image, max_delta=0.2)  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  
    if color_ordering == 2:  
        image = tf.image.random_hue(image, max_delta=0.2)  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  
    if color_ordering == 3:  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  
        image = tf.image.random_hue(image, max_delta=0.2)  
    return image 

def generate_batch(filename, batch_size=1): #取tfrecords产出batch
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表，
    filename_queue = tf.train.string_input_producer([filename])
    # 从文件中读出一个样例，也可以使用read_up_to函数一次性读出多个样例
    _, serialized_example = reader.read(filename_queue)
    # 解析读入的一个样例，如果需要解析多个样例，可以使用parse_example函数
    features = tf.parse_single_example(
        serialized_example,
        {
            # TensorFlow提供两种不同的属性解析方法。一种方法是tf.FixedLenFeature,
            # 这种方法解析的结果为一个Tensor。另一种方法是tf.VarLenFeature，这种方法
            # 得到的解析结果为SparseTensor，用于处理稀疏数据。这里解析数据的格式需要和
            # 上面程序写入数据的格式一致。
            'label_raw': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([],tf.string)

        }
    )

    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.reshape(tf.decode_raw(features['image_raw'], tf.uint8), [224,224,3])   #修改flaot和标准化
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = distort_color(image,np.random.randint(4))  #亮度饱和度色相对比度 随机排列变换
    image = tf.image.random_flip_left_right(image)    #左右反转
    image = tf.image.random_flip_up_down(image)   #上下反转
    
    
    label = tf.reshape(tf.decode_raw(features['label_raw'], tf.int32), [20])

    sess = tf.Session()
    # 启动多线程处理输入数据
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 每次运行可以读取TFRecord文件中的一个样例
    while 1:
        xs, ys = [], []
        for i in range(batch_size):
            # 读取一组数据
            x, y = sess.run([image, label])
            xs.append(x)
            ys.append(y)
            
        # 转换为数组，归一化
        xs= (np.array(xs, dtype=np.float32))/255. #标准化
        ys=np.array(ys)
        yield (xs, ys)

def generate_val_batch(filename, batch_size=1): #取tfrecords产出batch
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表，
    filename_queue = tf.train.string_input_producer([filename])
    # 从文件中读出一个样例，也可以使用read_up_to函数一次性读出多个样例
    _, serialized_example = reader.read(filename_queue)
    # 解析读入的一个样例，如果需要解析多个样例，可以使用parse_example函数
    features = tf.parse_single_example(
        serialized_example,
        {
            # TensorFlow提供两种不同的属性解析方法。一种方法是tf.FixedLenFeature,
            # 这种方法解析的结果为一个Tensor。另一种方法是tf.VarLenFeature，这种方法
            # 得到的解析结果为SparseTensor，用于处理稀疏数据。这里解析数据的格式需要和
            # 上面程序写入数据的格式一致。
            'label_raw': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([],tf.string)

        }
    )

    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.reshape(tf.decode_raw(features['image_raw'], tf.uint8), [224,224,3])   #修改flaot和标准化
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    label = tf.reshape(tf.decode_raw(features['label_raw'], tf.int32), [20])

    sess = tf.Session()
    # 启动多线程处理输入数据
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 每次运行可以读取TFRecord文件中的一个样例
    while 1:
        xs, ys = [], []
        for i in range(batch_size):
            # 读取一组数据
            x, y = sess.run([image, label])
            xs.append(x)
            ys.append(y)   
        # 转换为数组，归一化
        xs= (np.array(xs, dtype=np.float32))/255. #标准化
        ys=np.array(ys)
        yield (xs, ys)

if __name__ == '__main__':
    csvlist=read_csvlist("image_scene_training/list.csv")
    split_dir(csvlist)


# i=0
# for xs,ys in generate_batch(FILENAME_V):
#     i+=1
#     # print(ys,xs)
#     print(np.argmax(ys[0]))
#     cv2.imshow("rwr"+str(i),xs[0])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     if(i>5):
#     	break

#
# image=cv2.imread("image_scene_training/data/000de15c-165a-11e8-aaec-00163e025669.jpg")
# image=image_resize(image)
# cv2.imwrite("2.jpg",image)
# csvlist=read_csvlist("image_scene_training/list.csv")
# train_filepaths,train_labels,validation_filepaths,validation_labels=load_train_data(csvlist)
# print(one_hot(validation_labels)[0])