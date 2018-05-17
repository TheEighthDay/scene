#coding:utf-8
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from glob import glob


print("loading model1...")
model1=load_model('model.h5')
print("done")
print("model alyers="+str(len(model1.layers)))

print("loading model2...")
model2=load_model('model_new.h5')
print("done")
print("model alyers="+str(len(model2.layers)))

print("loading model3...")
model3=load_model('model_new2.h5')
print("done")
print("model alyers="+str(len(model3.layers)))

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
    return ending




# duiying=[0,1,10,11,12,13,14,15,16,17,18,19,2,3,4,5,6,7,8,9]

vary= [0,1,12,13,14,15,16,17,18,19,2,3,4,5,6,7,8,9,10,11]
summ=0
correct=0
for i in range (20):
    print(i)
    allpaths=glob('image_scene_training/validation/'+str(i)+'/*.jpg')
    s=len(allpaths)
    summ=summ+s
    imgarrays=[]
    for x in allpaths:
        img=cv2.imread(x)
        img=cv2.resize(img,(299,299))
        imgarrays.append(img)
    pre1=model1.predict(preprocess_input(np.array(imgarrays,dtype=np.float32)))
    pre2=model2.predict(preprocess_input(np.array(imgarrays,dtype=np.float32)))
    pre3=model3.predict(preprocess_input(np.array(imgarrays,dtype=np.float32)))
    all_top3_model1=[]
    all_top3=[]
    for w in range(s):
        top3_model1 = np.argsort(pre1[w])[-3:]
        top3_model2 = np.argsort(pre2[w])[-3:]
        top3_model3 = np.argsort(pre3[w])[-3:]
        all_top3.append([top3_model1[0],top3_model1[1],top3_model1[2],top3_model2[0],top3_model2[1],top3_model2[2],top3_model3[0],top3_model3[1],top3_model3[2]])
        all_top3_model1.append([top3_model1[0],top3_model1[1],top3_model1[2]])
    for z in range(s):
        all_top3[z]=find_top3(all_top3[z],all_top3_model1[z])
    for j in range(s):
        if all_top3[j][0]==vary[i] or all_top3[j][1]==vary[i] or all_top3[j][2]==vary[i]:
          correct=correct+1
    
print("正确率："+str(correct/float(summ)))










