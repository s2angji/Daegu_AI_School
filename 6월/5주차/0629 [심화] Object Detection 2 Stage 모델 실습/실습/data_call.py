from zipfile import ZipFile
import gdown
import argparse

"""pip install gdown"""
"""사용법 : python3 data_call.py --data FaceMaskDetection"""

file_destinations = {
    'FaceMaskDetection': 'Face Mask Detection.zip', }
file_id_dic = {
    'FaceMaskDetection': '1pJtohTc9NGNRzHj5IsySR39JIRPfkgD3'
}


def download_file_from_google_drive(id_, destination):
    url = f"https://drive.google.com/uc?id={id_}"
    output = destination
    gdown.download(url, output, quiet=True)
    print(f"{output} download complete")


parser = argparse.ArgumentParser(
    description='data loader ... '
)
parser.add_argument('--data', type=str, help='key for selecting data..!!')

args = parser.parse_args()


download_file_from_google_drive(
    id_=file_id_dic[args.data], destination=file_destinations[args.data]
)

"""압축 풀기"""
test_file_name = "./Face Mask Detection.zip"

with ZipFile(test_file_name, 'r') as zip:
    zip.printdir()
    zip.extractall()

# download_file_from_google_drive(
#     id=file_id_dic["FaceMaskDetection"],
#     destination=file_destinations["FaceMaskDetection"]
# )
