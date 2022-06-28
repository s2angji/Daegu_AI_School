import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mping
import matplotlib.patches as patches
from bs4 import BeautifulSoup
from natsort import natsort

"""pip install natsort"""
"""주어진 이미지에 바운딩 박스를 시각화 해서 올바르게 레이블링 되어있는지 확인작업 !!"""
img_list = natsort.natsorted(glob.glob("./images/*.png"))
label_list = natsort.natsorted(glob.glob("./annotations/*.xml"))
# print(len(img_list), len(label_list))

# print(img_list[:10])
# print("\n")
# print(label_list[:10])

"""바운딩 박스 시각화 위한 함수 정의"""


def generate_box(obj):
    """
    <xmin>79</xmin>
    <ymin>105</ymin>
    <xmax>109</xmax>
    <ymax>142</ymax>
    """
    print("test", obj)
    xmin = float(obj.find("xmin").text)
    ymin = float(obj.find('ymin').text)
    xmax = float(obj.find('xmax').text)
    ymax = float(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    # <name>with_mask</name>
    """label info -> mask_weared_incorrect -> 2, with_mask -> 1, without_mask -> 0 """

    print("label ", obj)
    if obj.find("name").text == 'with_mask':
        return 1
    elif obj.find("name").text == 'mask_weared_incorrect':
        return 2

    return 0


def generate_target(file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all("object")
        print(objects)
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return target


"""box 정보 가져오기"""
# print(img_list.index('./images/maksssksksss15.png'))
# bbox = generate_target(label_list[15])
# print(bbox)


def plot_image(img_path, annotation):

    # img = mping.imread(img_path)

    # 텐서 이미지 -> 이미지 화 처리
    print(type(img_path))
    img = img_path.permute(1, 2, 0)

    print("test type 처리후 : ", type(img))

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for idx in range(len(annotation['boxes'])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 0:
            rect = patches.Rectangle(
                (xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r', facecolor='none'
            )
        elif annotation['labels'][idx] == 1:
            rect = patches.Rectangle(
                (xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='g', facecolor='none'
            )
        else:
            rect = patches.Rectangle(
                (xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='b', facecolor='none'
            )
        ax.add_patch(rect)

    plt.show()


# bbox = generate_target(label_list[100])
# plot_image(img_list[100], bbox)
