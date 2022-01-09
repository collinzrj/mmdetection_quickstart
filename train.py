from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from generate_config import generate_config
from preprocess import generate_all_annotations
import sys


def train_mask_rcnn():
    cfg = generate_config()
    print(cfg.pretty_text)
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == '__main__':
    generate_all_annotations()
    train_mask_rcnn()
