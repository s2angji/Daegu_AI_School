from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import set_random_seed

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv.runner import get_dist_info, init_dist
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes
import torch


# 데이터셋 정의 부분 
# 정의 되어잇는 위치 경로 
# ./mmdet/datasets/coco.py
# coco.py 데이터셋 부분에서 클래스 이름 변경
@DATASETS.register_module(force=True)
class SmokeDataset(CocoDataset):
    CLASSES = ('Smoke','smoke')

# config 파일 호출 
# Dynamic RCNN 모델 사용 
# 모델 위치 ./configs/dynamic_rcnn
# 모델 config 호출 
config_file = "./configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_file)

print("cfg info ", cfg)

# 러닝 레이트 수정 기본 lr 경우 GPU 8개 가지고 진행하여서 싱글 GPU 경우 0.0025 기본으로 하면됩니다
# 단 학습 데이터 양에 따라서 기본 0.0025해보시고 판단하에 조정 하시면됩니다. 
# lr 기본 설정 확인 방법 : dynamic_rcnn_r50_fpn_1x_coco.py 보면 첫번째 줄보면 
# _base_ = "../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py" 있는 위치를 가셔서
# faster_rcnn_r50_fpn_1x_coco.py 열어보시면 아래와 같이 존재 
"""
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
요렇게 존재 합니다 여기서 보면 lr 경우는 schedule/schedule_1x.py 안에 있을겁니다 

schedule/schedule_1x.py 열어보면 다음과 같습니다. 아래의 내용에서 수정이 필요한경우 
수정하면됩니다. 
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# 스케줄에 보시면 스탭이 8, 11 되어있습니다 이 이유는 GPU 8대로 학습을 해서 GPU 1대로 
변경된다면 8 * 12 = 96 그러면 정상적으로 GPU 하나로 했을경우는 96번 학습해야합니다. 

기본적으로는 36, 56, 76 정도로 해보시고 loss 값을 보시고 조정 하시면됩니다. 
"""
# 다음과 같이 수정하면 러닝 레이트 가 변경되어 반영이 됩니다.
cfg.optimizer.lr = 0.0025

# 기존에 있던 coco dataset에 대한 환경 파라미터 수정 필요합니다. 
# 위에 42번째 줄을 참고하여 _base_ 리스트 안에 보면 ../_base_/datasets/coco_detection.py
# 있습니다 이부분에 내용을 coco에서 우리가 학습하고자 하는 데이터 형식으로 변경 필요 합니다. 
# coco_detection.py 들어가보시면 data dict 가 존재합니다. 이부분을 수정해야합니다. 
"""
필요시에 이미지 사이즈 조절은 
dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),

필요한 사이즈로 변경하시면됩니다. 투스테이션 경우는 여러 크기의 이미지 학습이 가능합니다. 
컴퓨팅 환경을 고려하여 세팅하시면됩니다. 
"""
# 데이터 대한 환경 파라미터 수정
# 24번 줄에 정해준 클래스 명을 넣어주면됩니다.  
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

# 클래스 개수 설정 
# 위에 42번째 줄에 참고하여 '../_base_/models/faster_rcnn_r50_fpn.py' 확인
# 코드 들어가보면 투스테이지 이기때문에 roid_head -> bbox_head dict 안에 num_classes 있습니다.
# 여기서 저부분을 변경하겠습니다. 
# 클래스 변경 설정 부분
cfg.model.roi_head.bbox_head.num_classes = 2

# pretrained 모델 로드
# 파일을 다운로드 해야합니다. 확인 방법은 
# configs/dynamic_rcnn/README.md 파일 체크 Results and Models 부분에 다운로드
# 존재합니다 거기서 다운로드 하시고 mmdetection 폴더에 넣어주세요 !! 

cfg.load_from = "./dynamic_rcnn_r50_fpn_1x-62a3f276.pth"

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정
cfg.work_dir = "./work_dir/0701"

# 학습율 변경 환경 파라미터 설정 
cfg.lr_config.warmup = None

# 몇번 마다 중간 평가와 세이브 파일 만들것인가 설정 
cfg.log_config.interval = 10 

# CocoDataset의 경우 metric을 bbox로 설정해야 함.(mAP아님. bbox로 설정하면 
# mAP를 iou threshold를 0.5 ~ 0.95까지 변경하면서 측정)
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 6
cfg.checkpoint_config.interval = 6

# Epochs 설정
# 8*12 -> 96
cfg.runner.max_epochs = 96
cfg.seed = 0

# GPU 설정
cfg.data.samples_per_gpu = 6
cfg.data.workers_per_gpu = 2
cfg.gpu_ids = range(1)
cfg.device = 'cuda'

# 시드 고정 
set_random_seed(0, deterministic=False)

datasets = [build_dataset(cfg.data.train)]
print(datasets[0])
# datasets[0].__dict__ 로 모든 self variables의 key와 value값을 볼 수 있음.
datasets[0].__dict__.keys()

model = build_detector(cfg.model, train_cfg=cfg.get(
    'train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)

# epochs는 config의 runner 파라미터로 지정됨. 학습 진행
train_detector(model, datasets, cfg, distributed=False, validate=True)






