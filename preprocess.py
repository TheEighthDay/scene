#coding:utf-8

import numpy as np
import cv2
import csv, random
from os import path
from tqdm import tqdm



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




if __name__ == '__main__':
    csvlist=read_csvlist("image_scene_training/list.csv")
    split_dir(csvlist)
