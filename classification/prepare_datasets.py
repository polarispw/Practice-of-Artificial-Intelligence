import pandas as pd
import cv2
from tqdm.contrib import tzip

data = pd.read_csv("E:/git repositories/Fundus-Imaging-Diagnosis/原始数据集/眼底分级/Mess1_annotation_train.csv")
source = "E:/train/"
destination = "datasets/"

name = data.iloc[:, 1]
grade = data.iloc[:, 4]
z = tzip(name, grade)
for x, y in z:
    x = str(x)
    y = str(y)
    path = source+x+".png"
    try:
        img = cv2.imread(path)
        if y=="0":
            cv2.imwrite(destination+"0/"+x+".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if y=="1":
            cv2.imwrite(destination+"1/"+x+".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if y=="2":
            cv2.imwrite(destination+"2/"+x+".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if y=="3" or y=="4":
            cv2.imwrite(destination+"3/"+x+".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    except:
        print(x)