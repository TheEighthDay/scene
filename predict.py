from preprocess import read_csvlist
import numpy as np
import tensorflow as tf
import cv2
from keras.models import  load_model
from tqdm import tqdm

def load_test_data(csv_results): #将csv读取信息转化为图片地址
    test_filepaths=[]

    print('Reading test data_info...')
    for i in range(len(csv_results)):
        file_id= csv_results[i]
        file_path = 'image_scene_testing/data/'+ file_id + '.jpg'  #猜测文件名image_scene_testing
        test_filepaths.append(file_path)
    print('Done')
  
    return test_filepaths

def gen_test_tfrecords(filepaths,tffilename):  #生成tfrecords文件
    writer = tf.python_io.TFRecordWriter(tffilename)
    print('Creating '+tffilename)

    for i in tqdm(range(len(filepaths))):
        image=cv2.imread(filepaths[i])
        image=cv2.resize(image,(224,224))
        image_raw = image.tostring()
        # 将一个样例转化为Example Protocol Buffer，并将所有信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        # 将一个Example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()   

    print("done")

def generate_test_batch(filename, batch_size=1): #取tfrecords产出batch
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
            'image_raw': tf.FixedLenFeature([],tf.string)

        }
    )
    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.reshape(tf.decode_raw(features['image_raw'], tf.uint8), [224,224,3])

    sess = tf.Session()
    # 启动多线程处理输入数据
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 每次运行可以读取TFRecord文件中的一个样例
    while 1:
        xs=[]
        for i in range(batch_size):
            # 读取一组数据
            x = sess.run([image])
            xs.append(x)
            
        # 转换为数组，归一化
        xs= (np.array(xs, dtype=np.float32))/255. #标准化
        yield xs

if __name__ == '__main__':

	filepaths="test.tfrecords"
	csv_results=read_csvlist("image_scene_testing/list.csv")  #文件名为猜测，还没放数据
	test_filepaths=load_test_data(csv_results)
	gen_test_tfrecords(test_filepaths,filepaths)

	model=load_model("model.h5")
	end1=model.predict_generator(generate_test_batch(filepaths),steps=100)  
	print(end1)
