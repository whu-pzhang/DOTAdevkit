# Convert UCAS-AOD dataset to COCO format
from pathlib import Path
import argparse
import json

from PIL import Image
from tqdm import tqdm
from shapely import geometry

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        type=str,
                        help='root path of UCAS-AOD dataset')
    parser.add_argument('--class_name',
                        type=str,
                        help='class name',
                        default='car')
    parser.add_argument('--val_ratio',
                        type=float,
                        default=0.2,
                        help='validation set ratio')

    return parser.parse_args()


def AOD2COCO(txt_list, dstfile=None, class_name='car'):
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []

    instance_cnt = 1
    image_id = 1

    for txt_fp in tqdm(txt_list):
        img_fp = txt_fp.with_suffix('.png')
        img = Image.open(img_fp)
        width, height = img.size

        data_dict['images'].append(
            dict(file_name=img_fp.name,
                 id=image_id,
                 width=width,
                 height=height))

        # annotations
        objs = parse_aod_anno(txt_fp)
        for obj in objs:
            xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                max(obj['poly'][0::2]), max(obj['poly'][1::2])
            width, height = xmax - xmin, ymax - ymin

            data_dict['annotations'].append({
                'area': obj['area'],
                'category_id': 1,  # only CAR or Plane,
                'segmentation': [obj['poly']],
                'iscrowd': 0,
                'bbox': (xmin, ymin, width, height),
                'image_id': image_id,
                'id': instance_cnt
            })
            instance_cnt += 1

        image_id += 1

    #

    data_dict['categories'] = [{
        'id': 1,
        'name': class_name,
        'supercategory': class_name
    }]

    with open(dstfile, 'w') as fout:
        json.dump(data_dict, fout, indent=2)


def parse_aod_anno(ann_file):
    objects = []
    with ann_file.open(mode='r') as fp:
        for line in fp:
            rec = [float(i) for i in line.strip().split('\t')]

            instance_dict = {
                'theta': rec[8],
                'x': rec[9],
                'y': rec[10],
                'width': rec[11],
                'height': rec[12]
            }

            # calc area
            gtpoly = geometry.Polygon([
                (rec[0], rec[1]),
                (rec[2], rec[3]),
                (rec[4], rec[5]),
                (rec[6], rec[7]),
            ])
            instance_dict['area'] = gtpoly.area

            # convert poly to flantten format
            instance_dict['poly'] = [int(i) for i in rec[:8]]

            objects.append(instance_dict)

    return objects


if __name__ == "__main__":
    args = parse_args()
    class_name = args.class_name.upper()  # CAR or PLANE
    root = Path(args.root) / class_name
    txt_list = list(root.glob('*.txt'))

    train_list, val_list = train_test_split(txt_list,
                                            test_size=args.val_ratio,
                                            random_state=42)

    AOD2COCO(
        train_list,
        dstfile='/home/pzhang/data/UCAS_AOD/AOD_vehicle_train.json',
    )
    AOD2COCO(
        val_list,
        dstfile='/home/pzhang/data/UCAS_AOD/AOD_vehicle_val.json',
    )
