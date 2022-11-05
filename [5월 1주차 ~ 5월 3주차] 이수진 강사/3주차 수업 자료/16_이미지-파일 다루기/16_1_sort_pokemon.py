import pandas as pd
import os
import shutil

image_root = r'C:\Users\user\ai_school\220516\pocketmon\image'
csv_path = r'C:\Users\user\ai_school\220516\pocketmon\pokemon.csv'

save_root = r'C:\Users\user\ai_school\220516\pocketmon\sorted'
os.makedirs(save_root, exist_ok=True)

pokemon_df = pd.read_csv(csv_path)
type_list = pokemon_df['Type1'].unique()

# 속성에 따른 하위 폴더 만들기
for t in type_list:
    dir_path = os.path.join(save_root, t)
    os.makedirs(dir_path, exist_ok=True)

# 이미지 폴더 안에서 포켓몬 파일/이름 읽기
filenames = os.listdir(image_root)
for filename in filenames:
    # 읽은 포켓몬 이름을, csv 파일이랑 매칭해서 type1 값 찾기
    name = filename.split('.')[0]
    type_ = pokemon_df[pokemon_df['Name'] == name]['Type1'].values[0]

    # 만들어 놓은 하위 폴더에 이미지 파일 옮기기
    current_path = os.path.join(image_root, filename)
    move_path = os.path.join(save_root, type_, filename)

    shutil.move(current_path, move_path)
