from Question_1 import MyCustomDatasetImage
import csv

dataset = MyCustomDatasetImage(data_path=".\\data\\", json_path=".\\anno\\raccoon_annotations.coco.json")
# dict로 CSV 파일 쓰기
with open("bbox.csv", 'w') as file:
    header = ['file_name', 'box_x', 'box_y', 'box_w', 'box_h']
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows([{header[0]:dataset.image_info[i]['file_name'],
                       header[1]:dataset.ann_info[i]['boxes'][0],
                       header[2]:dataset.ann_info[i]['boxes'][1],
                       header[3]:dataset.ann_info[i]['boxes'][2],
                       header[4]:dataset.ann_info[i]['boxes'][3]}
                      for i in range(len(dataset))])
