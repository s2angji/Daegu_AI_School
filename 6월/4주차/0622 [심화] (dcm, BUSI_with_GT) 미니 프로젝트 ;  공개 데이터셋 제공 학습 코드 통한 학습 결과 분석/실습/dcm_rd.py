import cv2
import os
import pydicom

input_data_path = "./dcm_data/"
out_image_save_path = "./dcm_data_png/"
os.makedirs(out_image_save_path, exist_ok=True)

dcm_file_list = [f for f in os.listdir(input_data_path)]

for f in dcm_file_list:

    # f -> ./dcm_data/ + 0a9fd225-a33a-47de-849e-156933b21296.dcm
    file_path = os.path.join(input_data_path, f)
    # ./dcm_data/0a5a6574-d94d-441f-afe4-115ba66b322e.dcm
    ds = pydicom.read_file(file_path)
    # print(ds['PatientID'])
    img = ds.pixel_array  # get image array
    # print(img.shape)
    # print(img)

    """image save"""
    cv2.imwrite(out_image_save_path + f.replace('dcm', 'png'), img)
