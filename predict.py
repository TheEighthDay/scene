#coding:utf-8
from preprocess import read_csvlist
import numpy as np
import cv2
from keras.applications.inception_v3 import preprocess_input
from keras.models import  load_model
import csv
duiying=[0,1,10,11,12,13,14,15,16,17,18,19,2,3,4,5,6,7,8,9]
def write_csvlist(listfile, data, header = []):
    with open(listfile, 'w') as fh:
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
        file_path = 'image_scene_test_b_0515/data/'+ file_id[0] + '.jpg'  
        test_filepaths.append(file_path)
    print('Done')
  
    return test_filepaths
def find_top3(L,firstL):
    ending=[]
    print(L)
    sort=np.bincount(np.array(L))
    tem=np.argsort(sort)[-3:]
    if sort[tem[2]]!=1:
        ending.append(tem[2])
    if sort[tem[1]]!=1:
        ending.append(tem[1])
    if sort[tem[0]]!=1:
        ending.append(tem[0])
    if len(ending)<3:
        need=3-len(ending)
        for i in range(need):
            for j in range(3):
                if firstL[j] not in ending:
                    ending.append(firstL[j])
                    break;
            
    print(ending)
    for i in range(3):
        ending[i]=duiying[ending[i]]
    return ending

def predict_top3(test_filepaths):
    print("loading..")
    model1 =load_model("model.h5")
    print("done")
    print("loading..")
    model2 =load_model("model_new.h5")
    print("done")
    print("loading..")
    model3 =load_model("model_new2.h5")
    print("done")

    summ=len(test_filepaths)
    batch=int(summ/100)

    ending=[]
    for i in range(100):
        batch_img=[]
        print(batch*i)
        for j in range(batch*i,batch*(i+1)):
            img=cv2.imread(test_filepaths[j])
            img=cv2.resize(img,(299,299))
            batch_img.append(img)
        predict1=model1.predict(preprocess_input(np.array(batch_img,dtype=np.float32)))
        predict2=model2.predict(preprocess_input(np.array(batch_img,dtype=np.float32)))
        predict3=model3.predict(preprocess_input(np.array(batch_img,dtype=np.float32)))

        for w in range(batch):
            top3_model1 = np.argsort(predict1[w])[-3:]
            top3_model2 = np.argsort(predict2[w])[-3:]
            top3_model3 = np.argsort(predict3[w])[-3:]

            top3=find_top3([top3_model1[0],top3_model1[1],top3_model1[2],top3_model2[0],top3_model2[1],top3_model2[2],top3_model3[0],top3_model3[1],top3_model3[2]],[top3_model1[0],top3_model1[1],top3_model1[2]])
            ending.append(top3)

    last=[]
    for k in range(batch*100,summ):
        img=cv2.imread(test_filepaths[k])
        img=cv2.resize(img,(299,299))
        last.append(img)

    predict_last1=model1.predict(preprocess_input(np.array(last,dtype=np.float32)))
    predict_last2=model2.predict(preprocess_input(np.array(last,dtype=np.float32)))
    predict_last3=model3.predict(preprocess_input(np.array(last,dtype=np.float32)))    

    for q in range(len(last)):
        top3_model1 = np.argsort(predict_last1[q])[-3:]
        top3_model2 = np.argsort(predict_last2[q])[-3:]
        top3_model3 = np.argsort(predict_last3[q])[-3:]
        top3=find_top3([top3_model1[0],top3_model1[1],top3_model1[2],top3_model2[0],top3_model2[1],top3_model2[2],top3_model3[0],top3_model3[1],top3_model3[2]],[top3_model1[0],top3_model1[1],top3_model1[2]])
        ending.append(top3)

    print("ending_top3_number="+str(len(ending)))

    return ending

        

if __name__ == '__main__':

    csv_results=read_csvlist("image_scene_test_b_0515/list.csv")  #文件名为猜测，还没放数据
    test_filepaths=load_test_data(csv_results)
    print("test_number="+str(len(test_filepaths)))
    top3_ending=predict_top3(test_filepaths)
    
    ending=[]
    for i in range(len(top3_ending)):
        ending.append([csv_results[i][0],str(top3_ending[i][0]),str(top3_ending[i][1]),str(top3_ending[i][2])])
    write_csvlist("ending.csv",ending,header=['FILE_ID','CATEGORY_ID0','CATEGORY_ID1','CATEGORY_ID2'])


    