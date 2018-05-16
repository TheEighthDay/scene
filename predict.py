from preprocess import read_csvlist
import numpy as np
import cv2
from keras.models import  load_model
import csv

def write_csvlist(listfile, data, header = []):
    with open(listfile, 'w',newline='') as fh:
        csv_writer = csv.writer(fh)
        if (len(header)):
            csv_writer.writerow(header)
        for row in data:
            csv_writer.writerow(row)

def load_test_data(csv_results): #将csv读取信息转化为图片地址
    test_filepaths=[]

    print('Reading test data_info...')
    for i in range(len(csv_results)):
        file_id= csv_results[i]
        file_path = 'image_scene_test_a_v2/a/data/'+ file_id[0] + '.jpg'  #猜测文件名image_scene_testing
        test_filepaths.append(file_path)
    print('Done')
  
    return test_filepaths

def predict_top3(test_filepaths):
    print("loading..")
    model =load_model("model.h5")
    print("done")

    summ=len(test_filepaths)
    batch=int(summ/100)
    # print("batch="+str(batch))

    ending=[]
    for i in range(100):
        batch_img=[]
        print(batch*i)
        for j in range(batch*i,batch*(i+1)):
            img=cv2.imread(test_filepaths[j])
            img=cv2.resize(img,(299,299))
            img=img/255.
            batch_img.append(img)
        print(np.array(batch_img))
        predict=model.predict(np.array(batch_img))
        print(predict)

        for i in range(batch):
            top3 = np.argsort(predict[i])[-3:]
            ending.append(top3)

    last=[]
    for k in range(batch*100,summ):
        img=cv2.imread(test_filepaths[k])
        img=cv2.resize(img,(299,299))
        img=img/255.
        # cv2.imshow("kankan",img)
        # cv2.waitKey(0)
        last.append(img)
    predict=model.predict(np.array(last))
    for i in range(len(last)):
        top3 = np.argsort(predict[i])[-3:]
        ending.append(top3)

    print("ending_top3_number="+str(len(ending)))
    print(ending[:10])
    return ending

        

if __name__ == '__main__':

    csv_results=read_csvlist("image_scene_test_a_v2/a/list.csv")  #文件名为猜测，还没放数据
    # print(csv_results)
    test_filepaths=load_test_data(csv_results)
    print("test_number="+str(len(test_filepaths)))
    top3_ending=predict_top3(test_filepaths)
    
    ending=[]
    for i in range(len(top3_ending)):
        ending.append([csv_results[i][0],str(top3_ending[i][0]),str(top3_ending[i][1]),str(top3_ending[i][2])])
    write_csvlist("testending.csv",ending,header=['FILE_ID','CATEGORY_ID0','CATEGORY_ID1','CATEGORY_ID2'])
    