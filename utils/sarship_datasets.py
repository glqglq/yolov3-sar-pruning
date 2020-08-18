# coding=utf-8

import os
import copy
import cv2
import numpy as np
import torch
import os.path as osp
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from utils.utils import xyxy2xywh


class SarShipDataset(Dataset):
    CLASSES = ('ship', )

    def __init__(self, data_dir, is_train, img_size=416, batch_size=16, cache = True):
        self.min_size = None
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.cache = cache
        self.cat2label_dict = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.names = SarShipDataset.CLASSES
        if is_train:
            self.img_prefix = osp.join(self.data_dir, 'Train/JPEGImages/')
            self.label_prefix = osp.join(self.data_dir, 'Train/Annotations/')
            self.output_dir = osp.join(self.data_dir, 'Train/OutPut/')
        else:
            self.img_prefix = osp.join(self.data_dir, 'Valid/JPEGImages/')
            self.label_prefix = osp.join(self.data_dir, 'Valid/Annotations/')
            self.output_dir = osp.join(self.data_dir, 'Valid/OutPut/')
        assert osp.exists(self.img_prefix)
        assert osp.exists(self.label_prefix)
        # os.rmdir(self.output_dir)
        if(not osp.exists(self.output_dir)):
            os.mkdir(self.output_dir)
        self.xml_file_names = os.listdir(self.label_prefix)
        self.length = len(self.xml_file_names)
        self.batch_count = np.floor(np.arange(len(self.xml_file_names)) / self.batch_size).astype(np.int)[-1] + 1

        if cache:
            self.img_ann = []
            for idx in range(len(self.xml_file_names)):
                img = self.load_image(idx)
                img2 = copy.deepcopy(img) * 255.0
                ann_dict = self.get_ann_info(idx)
                for bbox in ann_dict['bboxes']:
                    img2 = cv2.rectangle(img2, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0))
                cv2.imwrite(osp.join(self.output_dir, ann_dict['img_filename']), img2)
                self.img_ann.append([torch.from_numpy(img).permute([2, 0, 1]), ann_dict['normalized_bboxes'],
                                     osp.join(self.img_prefix, ann_dict['img_filename']), (ann_dict['height'], ann_dict['width'])])



    def __getitem__(self, idx):
        if(self.cache):
            return self.img_ann[idx]
        else:
            img = self.load_image(idx)
            ann_dict = self.get_ann_info(idx)
            return ([torch.from_numpy(img).permute([2, 0, 1]), ann_dict['normalized_bboxes'],
                                     osp.join(self.img_prefix, ann_dict['img_filename']), (ann_dict['height'], ann_dict['width'])])


    def __len__(self):
        return self.length


    def get_ann_info(self, idx):
        xml_path = osp.join(self.label_prefix, self.xml_file_names[idx])
        img_id, xml_suffix = os.path.splitext(self.xml_file_names[idx])
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.cat2label_dict[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            points = []
            for i in range(4):
                x = int(bnd_box.find("x{}".format(i)).text)
                y = int(bnd_box.find("y{}".format(i)).text)
                points.append([x, y])
            x, y, w, h = cv2.boundingRect(np.array(points))
            if(hasattr(self, 'r')):
                bbox = [item * self.r for item in [x, y, x + w, y + h]]  # for resize
            ignore = False
            if self.min_size:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)

        normalized_bboxes = []
        for bbox in bboxes:
            normalized_bboxes.append(np.array([0.0, 0.0] + list(map(lambda x: x * 1.0 / self.img_size, self.xyxy2xywh(bbox)))))
        normalized_bboxes = np.array(normalized_bboxes)

        ann = dict(
            id=img_id, xml_filename=self.xml_file_names[idx], img_filename = '%s.%s' % (img_id, 'png'), width=width, height=height,
            bboxes = bboxes.astype(np.float32),
            normalized_bboxes = torch.from_numpy(normalized_bboxes.astype(np.float32)),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann


    def load_image(self, idx):
        # loads 1 image from dataset

        img_path = osp.join(self.img_prefix, self.xml_file_names[idx].replace('xml', 'png'))
        img = cv2.imread(img_path)  # grayscale  , flags = cv2.IMREAD_GRAYSCALE
        assert img is not None, 'Image Not Found ' + img_path
        self.r = self.img_size / max(img.shape)  # size ratio
        if self.r < 1:
            h, w, _ = img.shape
            img = cv2.resize(img, (int(w * self.r), int(h * self.r)))  # _LINEAR fastest

        # Augment colorspace
        # if self.augment:
        #     augment_hsv(img, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])

        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img

    @staticmethod
    def load_image_from_path(img_path, img_size):
        # loads 1 image from dataset

        img = cv2.imread(img_path)  # grayscale  , flags = cv2.IMREAD_GRAYSCALE
        assert img is not None, 'Image Not Found ' + img_path
        r = img_size / max(img.shape)  # size ratio
        if r < 1:
            h, w, _ = img.shape
            img = cv2.resize(img, (int(w * r), int(h * r)))  # _LINEAR fastest

        # Augment colorspace
        # if self.augment:
        #     augment_hsv(img, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])

        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img

    @staticmethod
    def collate_fn(batch):
        img, labels, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(labels):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(labels, 0), path, hw

    def xyxy2xywh(self, x):
        # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[0] = (x[0] + x[2]) / 2
        y[1] = (x[1] + x[3]) / 2
        y[2] = x[2] - x[0]
        y[3] = x[3] - x[1]
        return y

if __name__ == '__main__':
    SarShipDataset('/home/luckygong/luckygong/yolov3-channel-and-layer-pruning/data/SARShip_pre_hepeng/', is_train = True, img_size=416, batch_size=16, cache = True)