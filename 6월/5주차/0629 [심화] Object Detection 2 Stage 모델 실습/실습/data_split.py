import os
import random
import numpy as np
import shutil


"""폴더구성"""
os.makedirs("./test_images", exist_ok=True)
os.makedirs("./test_annotations", exist_ok=True)

print(len(os.listdir("./annotations")))
print(len(os.listdir("./images")))

random.seed(7777)
idx = random.sample(range(853), 170)

for img in np.array(sorted(os.listdir("./images")))[idx]:
    print("img info ", img)
    shutil.move("./images/" + img, "./test_images/"+img)

for anno in np.array(sorted(os.listdir("./annotations")))[idx]:
    print("annotation info ", anno)
    shutil.move("./annotations/" + anno, "./test_annotations/" + anno)

print("info file size \n")
print(len(os.listdir("./images")))
print(len(os.listdir("./annotations")))
print(len(os.listdir("./test_images")))
print(len(os.listdir("./test_annotations")))
