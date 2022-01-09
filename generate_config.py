from mmcv import Config
from mmdet.apis import set_random_seed
import time
import os

ann_dir = os.environ['ANN_PATH']
img_prefix = os.environ['IMG_PATH']
checkpoint = os.environ['MODEL_PATH']

def generate_config():
    cfg = Config.fromfile(checkpoint)
    cfg.dataset_type = 'CocoDataset'
    cfg.classes = ('cell',)

    for head in cfg.model.roi_head.bbox_head:
        head.num_classes = 1
    cfg.model.roi_head.mask_head.num_classes=1

    # cfg.data.test.ann_file = ann_dir + 'train.json'
    # cfg.data.test.img_prefix = img_prefix
    # cfg.data.test.classes = cfg.classes
    cfg.data.train.ann_file = ann_dir + f'train.json'
    cfg.data.train.img_prefix = img_prefix
    cfg.data.train.classes = cfg.classes
    cfg.data.val.ann_file = ann_dir + f'val.json'
    cfg.data.val.img_prefix = img_prefix
    cfg.data.val.classes = cfg.classes
    cfg.load_from = checkpoint

    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 2

    cfg.evaluation.metric = 'segm'
    cfg.evaluation.interval = 1

    cfg.checkpoint_config.interval = 1
    cfg.runner.max_epochs = 12
    cfg.log_config.interval = 20

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    t = time.localtime()
    current_time = time.strftime("%y-%m-%d--%H:%M:%S", t)
    cfg.work_dir = './work_dir/' + current_time

    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=False, with_mask=True),
        # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
    ]
    return cfg

if __name__ == '__main__':
    print(generate_config().pretty_text)