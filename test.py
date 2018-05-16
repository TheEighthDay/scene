#coding:utf-8
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from glob import glob


print("loading model...")
model=load_model('model.h5')
print("done")
print("model alyers="+str(len(model.layers)))

# allpaths=glob('image_scene_training/validation/'+'3'+'/*.jpg')
# imgarrays=[]
# for x in allpaths:
#     img=cv2.imread(x)
#     img=cv2.resize(img,(299,299))
#     imgarrays.append(img)
# pre=model.predict(preprocess_input(np.array(imgarrays,dtype=np.float32)))
# pre_ending=np.argmax(pre,axis=1)
# print(pre_ending)

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
    pre=model.predict(preprocess_input(np.array(imgarrays,dtype=np.float32)))
    all_top3=[]
    for w in range(s):
        top3 = np.argsort(pre[w])[-3:]
        all_top3.append(top3)

    print(all_top3)
    for j in range(s):
        if all_top3[j][0]==vary[i] or all_top3[j][1]==vary[i] or all_top3[j][2]==vary[i]:
          correct=correct+1
print("正确率："+str(correct/float(summ)))










