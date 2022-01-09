import json
from os.path import exists

def generate_annotations(filename):
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0
    for _ in range(10):
        annotation = dict(
            image_id=image_id,
            id=annotation_id,
            category_id=0,
            # minx, miny, maxx, maxy
            bbox=[0, 0, 100, 100],
            # x1, y1, x2, y2, x3, y3...
            segmentation=[[0, 0, 0, 100, 100, 100, 100, 0]],
            # calculate area
            area=10000,
            iscrowd=0)
        annotations.append(annotation)
        annotation_id += 1
        image = dict(
            id=image_id,
            file_name='basic.jpg',
            # update based on real height and width
            height=400,
            width=400)
        images.append(image)
        image_id += 1
    categories = [{"id": 0, "name": "cat"}]
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)
    with open(filename, 'w') as f:
        json.dump(coco_format_json, f)

def generate_all_annotations():
    train_path = './data/annotations/train.json'
    val_path = './data/annotations/val.json'
    if not exists(train_path):
        generate_annotations(train_path)
    else:
        print("train annotation exists")
        generate_annotations(train_path)
    if not exists(val_path):
        generate_annotations(val_path)
    else:
        print("validation annotation exists")
        generate_annotations(val_path)

if __name__ == '__main__':
    generate_all_annotations()