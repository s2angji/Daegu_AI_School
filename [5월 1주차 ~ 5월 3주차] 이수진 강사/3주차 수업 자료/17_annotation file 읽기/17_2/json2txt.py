import os
import json

save_root = r'C:\Users\user\ai_school\220516\steel_masking\anno'
json_path = r'C:\Users\user\ai_school\220516\steel_masking\annotation.json'

with open(json_path, 'r') as j:
    json_data = json.load(j)

for filename in json_data.keys():
    filename_txt = filename.split('.')[0] + '.txt'
    save_path = os.path.join(save_root, filename_txt)

    f = open(save_path, 'w')

    json_image = json_data[filename]
    width = json_image['width']
    height = json_image['height']
    annos = json_image['anno']
    for anno in annos:
        xmin, ymin, xmax, ymax = anno

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin

        x_center = x_center / width
        bbox_w = bbox_w / width

        y_center = y_center / height
        bbox_h = bbox_h / height

        write = f'0 {x_center} {y_center} {bbox_w} {bbox_h}\n'
        f.write(write)
    f.close()