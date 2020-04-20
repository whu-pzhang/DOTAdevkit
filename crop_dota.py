import os
from pathlib import Path
import argparse

from dotatools import ImgSplit_multi_process
from dotatools import SplitOnlyImage_multi_process

wordname_15 = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
]


def parse_args():
    parser = argparse.ArgumentParser(description='prepare dota1')
    parser.add_argument('--srcpath', default='/home/dingjian/project/dota')
    parser.add_argument('--dstpath',
                        default=r'/home/dingjian/workfs/dota1-split-1024',
                        help='prepare data')
    parser.add_argument('--subsize',
                        type=int,
                        default=1024,
                        help='patch size of sub-images')
    parser.add_argument('--gap',
                        type=int,
                        default=512,
                        help='overlap between two patches')
    args = parser.parse_args()

    return args


def prepare(srcpath, dstpath, subsize, gap):
    """
    :param srcpath: train, val, test
          train --> trainval1024, val --> trainval1024, test --> test1024
    :return:
    """
    dstpath = Path(dstpath)
    train_dir = dstpath / f'train{subsize}'
    val_dir = dstpath / f'val{subsize}'
    test_dir = dstpath / f'test{subsize}/images'  # test_dir only has images directory

    # train_dir.mkdir(parents=True, exist_ok=True)
    # val_dir.mkdir(parents=True, exist_ok=True)
    # test_dir.mkdir(parents=True, exist_ok=True)

    split_kwargs = {
        'subsize': subsize,
        'gap': gap,
        'num_process': 8,
    }

    split_train = ImgSplit_multi_process.splitbase(
        os.path.join(srcpath, 'train'), train_dir, **split_kwargs)
    split_train.splitdata(1)

    split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
                                                 val_dir, **split_kwargs)
    split_val.splitdata(1)

    split_test = SplitOnlyImage_multi_process.splitbase(
        os.path.join(srcpath, 'test', 'images'), test_dir, **split_kwargs)
    split_test.splitdata(1)


if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    subsize = args.subsize
    gap = args.gap
    prepare(srcpath, dstpath, subsize, gap)
