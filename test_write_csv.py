#coding:utf-8
import csv

import numpy as np

def write_csvlist(listfile, data, header = []):
    with open(listfile, 'w') as fh:
        csv_writer = csv.writer(fh)
        if (len(header)):
            csv_writer.writerow(header)
        for row in data:
            csv_writer.writerow(row)

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



csv_results=[['asdad'],['bbbbbbb'],['asdada']]
top3_ending=[[1,2,3],[12,2,3],[1,23,4]]





ending=[]
for i in range(len(top3_ending)):
    ending.append([csv_results[i][0],str(top3_ending[i][0]),str(top3_ending[i][1]),str(top3_ending[i][2])])
write_csvlist("ending.csv",ending,header=['FILE_ID','CATEGORY_ID0','CATEGORY_ID1','CATEGORY_ID2'])
print(read_csvlist("ending.csv"))