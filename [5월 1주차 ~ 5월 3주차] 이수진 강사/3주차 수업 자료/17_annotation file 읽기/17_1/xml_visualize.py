import xml.etree.ElementTree as ET
import os
import cv2

image_root = r'C:\Users\user\ai_school\220516_20\17_automotive_engine\automotive_engine\image'
xml_root = r'C:\Users\user\ai_school\220516_20\17_automotive_engine\automotive_engine\xml'

for filename in os.listdir(image_root):
    image_path = os.path.join(image_root, filename)
    image = cv2.imread(image_path)

    filename_xml = filename.split('.')[0] + '.xml'
    xml_path = os.path.join(xml_root, filename_xml)

    annotation = ET.parse(xml_path)
    object_nodes = annotation.findall('object')
    for object_node in object_nodes:
        bnd_node = object_node.find('bnd_box')
        xmin = int(bnd_node.find('xmin').text)
        xmax = int(bnd_node.find('xmax').text)
        ymin = int(bnd_node.find('ymin').text)
        ymax = int(bnd_node.find('ymax').text)

        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 3)

    cv2.imshow('visual', image)
    if cv2.waitKey(0) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        exit()