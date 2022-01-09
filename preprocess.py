import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from os.path import exists

# TODO: prepare a train_df
train_df = pd.read_csv("foo.csv")

def generate_annotations(rows, filename, ann_limit):
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0
    count= 0
    for id, group in tqdm(rows):
        # skip if too many annotations
        if len(group['annotation']) > ann_limit:
            continue
        annotation_count = 0
        for mask in group['annotation']:
            # TODO generate bbox and polygons
            bbox = []
            polygons = []
            annotation = dict(
                image_id=image_id,
                id=annotation_id,
                category_id=0,
                bbox=bbox,
                segmentation=polygons,
                area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1]),
                iscrowd=0)
            annotations.append(annotation)
            annotation_id += 1
            annotation_count += 1
        image = dict(
            id=image_id,
            file_name=id+'.png',
            height=520,
            width=704)
        images.append(image)
        image_id += 1
    # TODO fill categories
    categories = []
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)
    with open(filename, 'w') as f:
        json.dump(coco_format_json, f)

def generate_all_annotations(ann_limit):
    rows = list(train_df.groupby('id'))
    divide = int(len(rows) * 0.7)
    train_set = rows[:divide]
    val_set = rows[divide:]
    train_path = 'data/annotations/train.json'
    val_path = 'data/annotations/val.json'
    if not exists(train_path):
        generate_annotations(train_set, train_path, ann_limit)
    else:
        print("train annotation exists")
    if not exists(val_path):
        generate_annotations(val_set, val_path, ann_limit)
    else:
        print("validation annotation exists")

if __name__ == '__main__':
    generate_all_annotations(2)