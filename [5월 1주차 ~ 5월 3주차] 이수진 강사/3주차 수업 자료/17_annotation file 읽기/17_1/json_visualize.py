import os
import json
import cv2
import numpy as np

image_root = r'C:\Users\user\ai_school\220516_20\17_automotive_engine\automotive_engine\image'
json_root = r'C:\Users\user\ai_school\220516_20\17_automotive_engine\automotive_engine\json'

for filename in os.listdir(image_root):
    image_path = os.path.join(image_root, filename)
    image = cv2.imread(image_path)

    filename_json = filename.split('.')[0] + '.json'
    json_path = os.path.join(json_root, filename_json)

    with open(json_path, 'r') as j:
        json_data = json.load(j)

    annos = json_data['shapes']
    for anno in annos:
        points = anno['points'] # [ [x1, y1], [x2, y2], ... ]
        points = np.array(points, np.int)

        image = cv2.polylines(image, [points], True, (255, 255, 0), 3)

    cv2.imshow('visual', image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()