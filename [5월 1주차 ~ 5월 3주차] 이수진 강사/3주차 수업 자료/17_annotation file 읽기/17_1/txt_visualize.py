import os
import cv2

image_root = r'C:\Users\user\ai_school\220516_20\17_automotive_engine\automotive_engine\image'
txt_root = r'C:\Users\user\ai_school\220516_20\17_automotive_engine\automotive_engine\txt'

for filename in os.listdir(image_root):
    image_path = os.path.join(image_root, filename)
    image = cv2.imread(image_path)
    height, width, channel = image.shape

    filename_txt = filename.split('.')[0] + '.txt'
    txt_path = os.path.join(txt_root, filename_txt)

    with open(txt_path, 'r') as f:
        while True:
            line = f.readline()[:-2]
            if not line:
                break
            line = line.split(' ')
            x_center, y_center, w, h = line[1:]
            x_center = float(x_center)
            y_center = float(y_center)
            w = float(w)
            h = float(h)

            x_center = x_center * width
            y_center = y_center * height
            w = w * width
            h = h * height

            xmin = int(x_center - w/2)
            ymin = int(y_center - h/2)
            xmax = int(x_center + w/2)
            ymax = int(y_center + h/2)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 3)

    cv2.imshow('visualize', image)
    if cv2.waitKey(0) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        exit()
