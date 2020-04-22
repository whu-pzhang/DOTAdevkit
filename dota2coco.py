# Convert splited DOTA dataset into COCO format

from pathlib import Path
import argparse
import json

from PIL import Image
from tqdm import tqdm
from shapely import geometry

wordname_15 = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
]

wordname_16 = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter', 'container-crane'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='cropped DOTA dataset path')
    parser.add_argument('out_json', type=str, help='output json file')
    parser.add_argument('-c',
                        '--class_names',
                        nargs='*',
                        default=wordname_15,
                        help='class names list you want to convert')

    args = parser.parse_args()
    return args


def DOTA2COCO(srcpath, dstfile=None, class_names=None):

    txt_path = Path(srcpath) / 'labelTxt'
    img_path = Path(srcpath) / 'images'

    file_ids = [f.stem for f in txt_path.glob('*.txt')]

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []

    instance_cnt = 1
    image_id = 1

    for fid in tqdm(file_ids):
        img_fp = img_path / f'{fid}.png'
        txt_fp = txt_path / f'{fid}.txt'

        img = Image.open(img_fp)
        width, height = img.size

        data_dict['images'].append(
            dict(
                file_name=img_fp.name,
                id=image_id,
                width=width,
                height=height,
            ))

        # annotations
        objs = parse_dota_anno(txt_fp, select_classes=class_names)
        for obj in objs:
            xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                max(obj['poly'][0::2]), max(obj['poly'][1::2])
            width, height = xmax - xmin, ymax - ymin

            ann = {
                'area': obj['area'],
                'category_id': class_names.index(obj['name']) + 1,
                'segmentation': [obj['poly']],
                'iscrowd': 0,
                'bbox': (xmin, ymin, width, height),
                'image_id': image_id,
                'id': instance_cnt
            }
            data_dict['annotations'].append(ann)

            instance_cnt += 1
        image_id += 1

    #
    for idx, name in enumerate(class_names):
        single_cat = {'id': idx + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    with open(dstfile, 'w') as fout:
        json.dump(data_dict, fout, indent=2)


def parse_dota_anno(ann_file, select_classes=None):
    objects = []
    with ann_file.open(mode='r') as fp:
        for line in fp:
            instance_dict = {}
            splitlines = line.strip().split(' ')
            if (len(splitlines) < 9):
                continue

            if (len(splitlines) >= 9):
                class_name = splitlines[8]
                if select_classes is not None:
                    if class_name not in select_classes:
                        continue
                    else:
                        instance_dict['name'] = class_name
                else:
                    instance_dict['name'] = class_name

            if (len(splitlines) == 9):
                instance_dict['difficult'] = '0'
            elif (len(splitlines) >= 10):
                instance_dict['difficult'] = splitlines[9]

            xy = [float(i) for i in splitlines[:8]]
            # calc area
            gtpoly = geometry.Polygon([
                (xy[0], xy[1]),
                (xy[2], xy[3]),
                (xy[4], xy[5]),
                (xy[6], xy[7]),
            ])
            instance_dict['area'] = gtpoly.area

            # convert poly to flantten format
            instance_dict['poly'] = [int(i) for i in xy]

            objects.append(instance_dict)

    return objects


if __name__ == "__main__":
    args = parse_args()
    assert set(args.class_names).issubset(set(wordname_15))
    print(args)

    DOTA2COCO(args.root, args.out_json, args.class_names)

    # # train
    # DOTA2COCO(
    #     r'/home/pzhang/data/DOTA/dota1.0_split/train800/',
    #     r'/home/pzhang/data/DOTA/dota1.0_split/DOTA_small-vehicle_train800.json',
    #     class_names=('small-vehicle', ),
    # )
    # # val
    # DOTA2COCO(
    #     r'/home/pzhang/data/DOTA/dota1.0_split/val800/',
    #     r'/home/pzhang/data/DOTA/dota1.0_split/DOTA_small-vehicle_val800.json',
    #     class_names=('small-vehicle', ),
    # )
