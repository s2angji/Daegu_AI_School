import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

data_path = "./DATASET/TRAIN"
data_dir = os.listdir(data_path)

print("data_path: ", data_dir)

for folder in data_dir:
    if folder not in ".DS_Store":

        if folder == "R":
            file_list_R = glob.glob(os.path.join(data_path, folder, "*.jpg"))

        elif folder == "O":
            file_list_O = glob.glob(os.path.join(data_path, folder, "*.jpg"))


r_data_size = len(file_list_R)
o_data_size = len(file_list_O)

print("r_data_size: ", r_data_size, "o_data_size: ", o_data_size)

r_indices = list(range(r_data_size))
o_indices = list(range(o_data_size))


r_data_split_number = 0.04
o_data_split_number = 0.032

r_split = int(np.floor(r_data_split_number * r_data_size))
o_split = int(np.floor(o_data_split_number * o_data_size))
print(r_split)
print(o_split)

r_data_indices, o_data_indices = r_indices[:r_split+1], o_indices[:o_split+1]

r_data = []
for i in r_data_indices:
    path = file_list_R[i]
    r_data.append(path)

o_data = []
for i in o_data_indices:
    path = file_list_O[i]
    o_data.append(path)

all_data = r_data + o_data
x_train, x_valid = train_test_split(
    all_data, test_size=0.2, shuffle=False, random_state=777)

print("x_train size >> ", len(x_train))  # train data
print("y_train size >> ", len(x_valid))  # val data
