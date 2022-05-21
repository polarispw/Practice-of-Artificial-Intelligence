import pandas as pd
import cv2
from tqdm.contrib import tzip

data = pd.read_csv("trainLabels_cropped.csv")
source = "resized_train_cropped/resized_train_cropped/"
destination = "datasets_from_kaggle/"

name = data.iloc[:, 2]
grade = data.iloc[:, 3]
z = tzip(name, grade)
for x, y in z:
    x = str(x)
    y = str(y)
    path = source+x+".jpeg"
    try:
        img = cv2.imread(path)
        if y=="0":
            cv2.imwrite(destination+"0/"+x+".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if y=="1":
            cv2.imwrite(destination+"1/"+x+".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if y=="2":
            cv2.imwrite(destination+"2/"+x+".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if y=="3":
            cv2.imwrite(destination+"3/"+x+".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if y=="4":
            cv2.imwrite(destination+"4/"+x+".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    except:
        print(x)