import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv import Config
from mmdet.datasets.builder import DATASETS
import numpy as np
import matplotlib.pyplot as plt
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.apis import set_random_seed
import json
import os

# Dynamic RCNN 옵션 불러오기 및 출력
config_file = '../Train/mmdetection/configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_file)

@DATASETS.register_module(force=True)
class SmokeDataset(CocoDataset):
    CLASSES = ('Smoke','smoke')


cfg.dataset_type = "SmokeDataset"
# 데이터가 있는 폴더명 제 기준 데이터 폴더명은 dataset
cfg.data_root = "./dataset"
# coco_detection.py 파일에서 data dict에 train, val, test에 대한 위치 경로 수정
cfg.data.train.type = "SmokeDataset"
cfg.data.train.ann_file = "./dataset/train/_annotations.coco.json"
cfg.data.train.img_prefix = "./dataset/train/"

cfg.data.val.type = "SmokeDataset"
cfg.data.val.ann_file = "./dataset/valid/_annotations.coco.json"
cfg.data.val.img_prefix = "./dataset/valid/"

cfg.data.test.type = "SmokeDataset"
cfg.data.test.ann_file = "./dataset/test/_annotations.coco.json"
cfg.data.test.img_prefix = "./dataset/test/"


cfg.load_from = "./dynamic_rcnn_r50_fpn_1x-62a3f276.pth"

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정
cfg.work_dir = "./work_dir/0701"


# 학습율 변경 환경 파라미터 설정.
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# CocoDataset의 경우 metric을 bbox로 설정해야 함.(mAP아님. bbox로 설정하면 mAP를 iou threshold를 0.5 ~ 0.95까지 변경하면서 측정)
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 12
# Epochs 설정
cfg.runner.max_epochs = 10
cfg.seed = 0
cfg.gpu_ids = range(1)
set_random_seed(0, deterministic=False)

# model loader
checkpoint_file = './work_dir/0701/학습한 모델.pth'

model = init_detector(cfg, checkpoint_file, device='cuda:0')

# 테스트 할 이미지 * 만약 테스트 폴더에 있는 이미지를 전부 처리하고 싶으면 이미지 가져오는 코드 작성하여 진행하시면됩니다.
img = './sample.png'

results = inference_detector(model, img)
show_result_pyplot(model, img, results)

# 쓰레시홀더를 줘서 50% 이하는 제거 하고 50% 이상 인 박스만 표기
# 박스 값을 확인 하고 싶은 경우
score_threshold = 0.5
for number, result in enumerate(results):
    if len(result) == 0:
        continue
    category_id = number + 1
    result_filtered = result[np.where(result[:, 4] > score_threshold)]
    if len(result_filtered) == 0:
        continue
    for i in range(len(result_filtered)):
        x_min = int(result_filtered[i, 0])
        y_min = int(result_filtered[i, 1])
        x_max = int(result_filtered[i, 2])
        y_max = int(result_filtered[i, 3])
        print(x_min, y_min, x_max, y_max)
