import os
import json
import cv2
import numpy as np

json_path = "json_data/instances_polygon.json"

with open(json_path, "r") as f:
    coco_info = json.load(f)

assert len(coco_info) > 0, "파일 읽기 실패"

# 카테고리 정보 수집
categories = dict()
for category in coco_info['categories']:
    categories[category["id"]] = category["name"]

# print("categories info >> ", categories)

# annotaiton 정보
ann_info = dict()
for annotation in coco_info['annotations']:
    # print("annotation >> ", annotation)
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    category_id = annotation["category_id"]
    segmentation = annotation["segmentation"]

    if image_id not in ann_info:
        ann_info[image_id] = {
            "boxes": [bbox], "segmentation": [segmentation],
            "categories": [category_id]
        }
    else:
        ann_info[image_id]["boxes"].append(bbox)
        ann_info[image_id]["segmentation"].append(segmentation)
        ann_info[image_id]["categories"].append(categories[category_id])

for image_info in coco_info["images"]:
    filename = image_info["file_name"]
    height = image_info["height"]
    width = image_info["width"]
    img_id = image_info["id"]

    file_path = os.path.join("0525_image_data", filename)
    # ./0525_image_data/image.jpeg
    # image read
    img = cv2.imread(file_path)

    # 예외처리 문법
    try:
        annotation = ann_info[img_id]
    except KeyError:
        continue

    for bbox, segmentation, category in zip(annotation['boxes'], annotation['segmentation'], annotation['categories']):

        x1, y1, w, h = bbox

        for seg in segmentation:
            poly = np.array(seg, np.int32).reshape((int(len(seg)/2), 2))
            print(poly)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            org_img = img.copy()

            text_img = cv2.putText(img, categories[category],
                                   (int(x1), int(y1) - 10), font, fontScale, color, thickness, cv2.LINE_AA)
            poly_img = cv2.polylines(text_img, [poly], True, (255, 0, 0), 2)
            cv2.imshow("test", poly_img)
            cv2.waitKey(0)
