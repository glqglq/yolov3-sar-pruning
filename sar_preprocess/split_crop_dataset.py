# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

# 将原始的SARShip图像划分为训练集和测试集，对训练集和测试集分别使用不同的参数进行切分

import os
import argparse
import logging
from utils import parse_file_info, crop_image_list

logger = logging.getLogger(__name__)

class_list = ['ship']


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("rootdir", help="The root path of SARShip dataset")
    parser.add_argument("savedir", help="The path to save the cropped dataset")
    parser.add_argument("--fileinfo", help='The path to the fileinfo', default='fileinfo.txt')
    parser.add_argument('--width', help="The width of the cropped image", default=500, type=int)
    parser.add_argument('--height', help="The height of the cropped image", default=500, type=int)
    parser.add_argument('--train_stride', help="The stride between two crop images, in other word, overlap = image_h "
                                               "- stride", default=400, type=int)
    parser.add_argument('--valid_stride', help="The stride or overlap between two crop images", default=400, type=int)

    return parser.parse_args()

def same_tarin_valid_with_paper():
    all = set(i for i in range(1, 32))
    valid_set = {5, 6, 8, 10, 12, 22, 24, 29, 30, 31}
    train_set = all - valid_set

    train_list = ["SARShip-1.0-{}".format(x) for x in train_set]
    valid_list = ["SARShip-1.0-{}".format(x) for x in valid_set]

    return train_list, valid_list


def get_train_valid(images, same_with_paper=True):
    if same_with_paper:
        return same_tarin_valid_with_paper()
    else:
        return images[:21], images[21:]




def main():
    args = parser_args()

    root_dir_train = os.path.join(args.savedir, "train")
    label_save_dir_train = os.path.join(args.savedir, "train", "labeltxt")
    images_save_dir_train = os.path.join(args.savedir, "train", "images")

    root_dir_valid = os.path.join(args.savedir, "valid")
    label_save_dir_valid = os.path.join(args.savedir, "valid", "labeltxt")
    images_save_dir_valid = os.path.join(args.savedir, "valid", "images")

    dirs = [label_save_dir_train, label_save_dir_valid, images_save_dir_train, images_save_dir_valid]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    images = parse_file_info(args.fileinfo)['all']


    train_images, valid_images = get_train_valid(images)
    # print(train_images, valid_images)

    crop_image_list(args.rootdir, train_images,
                    height=args.height,
                    width=args.width,
                    stride=args.train_stride,
                    class_list=class_list,
                    save_dir=root_dir_train)

    crop_image_list(args.rootdir, valid_images,
                    height=args.height,
                    width=args.width,
                    stride=args.valid_stride,
                    class_list=class_list,
                    save_dir=root_dir_valid)


if __name__ == '__main__':
    main()
