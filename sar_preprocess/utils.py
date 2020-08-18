# Copyright 2020 by He Peng (sdythp@gmail.com).
# All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
import os
import copy
import numpy as np
import cv2
from xml.dom.minidom import Document
import logging

logger = logging.getLogger(__name__)


def parse_file_info(file_info_path):
    """

    :param file_info_path: file info path
    :return: dict:
                all: all the images
                offshore: the offshore images
                inshore: the inshore images
                3: 3m sar resolution
                1: 1m sar resolution
    """
    assert os.path.exists(file_info_path), "File info not exists"
    with open(file_info_path, 'r', encoding='utf-8') as f:
        # Skip first comment line
        ret = defaultdict(list)
        for line in f:
            if line.startswith('#'):
                continue
            fileinfo = line.strip().split(' ')
            filename = fileinfo[0]
            shore = fileinfo[3]
            resolution = fileinfo[4]
            ret['all'].append(filename)
            ret[shore].append(filename)
            ret[resolution].append(filename)
    return ret


def save_to_xml(save_path, im_height, im_width, objects_axis, label_name):
    im_depth = 0
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('VOC2007')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_name = doc.createTextNode('000024.jpg')
    filename.appendChild(filename_name)
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('The VOC2007 Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)

    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)

    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('322409915'))
    source.appendChild(flickrid)

    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('knautia'))
    owner.appendChild(flickrid_o)

    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('gum'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(im_width)))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(im_height)))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(im_depth)))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(object_num):
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(
            label_name[int(objects_axis[i][-1])]))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        x0 = doc.createElement('x0')
        x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
        bndbox.appendChild(x0)
        y0 = doc.createElement('y0')
        y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
        bndbox.appendChild(y0)

        x1 = doc.createElement('x1')
        x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
        bndbox.appendChild(x1)
        y1 = doc.createElement('y1')
        y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
        bndbox.appendChild(y1)

        x2 = doc.createElement('x2')
        x2.appendChild(doc.createTextNode(str((objects_axis[i][4]))))
        bndbox.appendChild(x2)
        y2 = doc.createElement('y2')
        y2.appendChild(doc.createTextNode(str((objects_axis[i][5]))))
        bndbox.appendChild(y2)

        x3 = doc.createElement('x3')
        x3.appendChild(doc.createTextNode(str((objects_axis[i][6]))))
        bndbox.appendChild(x3)
        y3 = doc.createElement('y3')
        y3.appendChild(doc.createTextNode(str((objects_axis[i][7]))))
        bndbox.appendChild(y3)

    f = open(save_path, 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


def clip_image(origin_filename, image, height, width, stride, boxes_all, class_list, save_dir):
    if len(boxes_all) > 0:
        shape = image.shape
        for start_h in range(0, shape[0], stride):
            for start_w in range(0, shape[1], stride):
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[0]:
                    start_h_new = shape[0] - height
                if start_w + width > shape[1]:
                    start_w_new = shape[1] - width
                top_left_row = max(start_h_new, 0)
                top_left_col = max(start_w_new, 0)
                bottom_right_row = min(start_h + height, shape[0])
                bottom_right_col = min(start_w + width, shape[1])

                subImage = image[top_left_row:bottom_right_row,
                           top_left_col: bottom_right_col]

                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col
                box[:, 4] = boxes[:, 4] - top_left_col
                box[:, 6] = boxes[:, 6] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                box[:, 5] = boxes[:, 5] - top_left_row
                box[:, 7] = boxes[:, 7] - top_left_row
                box[:, 8] = boxes[:, 8]
                center_y = 0.25 * (box[:, 1] + box[:, 3] + box[:, 5] + box[:, 7])
                center_x = 0.25 * (box[:, 0] + box[:, 2] + box[:, 4] + box[:, 6])

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[
                                           0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)

                if len(idx) > 0:
                    xml = os.path.join(save_dir, 'labeltxt', "%s_%04d_%04d.xml" % (
                        origin_filename, top_left_row, top_left_col))
                    save_to_xml(
                        xml, subImage.shape[0], subImage.shape[1], box[idx, :], class_list)
                    if subImage.shape[0] > 5 and subImage.shape[1] > 5:
                        img = os.path.join(save_dir, 'images', "%s_%04d_%04d.png" % (
                            origin_filename, top_left_row, top_left_col))
                        am = np.amax(subImage)
                        subImage = subImage * 255 / am
                        cv2.imwrite(img, subImage)


def format_label(txt_list, class_list):
    format_data = []
    for i in txt_list:
        box_label = i.strip().split(",")
        if len(i) < 10:
            continue
        format_data.append(
            [int(xy) for xy in box_label[:8]] +
            [class_list.index(box_label[8])]
            # {'x0': int(i.split(' ')[0]),
            # 'x1': int(i.split(' ')[2]),
            # 'x2': int(i.split(' ')[4]),
            # 'x3': int(i.split(' ')[6]),
            # 'y1': int(i.split(' ')[1]),
            # 'y2': int(i.split(' ')[3]),
            # 'y3': int(i.split(' ')[5]),
            # 'y4': int(i.split(' ')[7]),
            # 'class': class_list.index(i.split(' ')[8]) if i.split(' ')[8] in class_list else 0,
            # 'difficulty': int(i.split(' ')[9])}
        )
        if box_label[8] not in class_list:
            logger.error('warning found a new label :', i.split(' ')[8])
            exit()
    return np.array(format_data)


def crop_image_list(root_dir, image_names, height, width, stride, class_list, save_dir):
    for idx, image_name in enumerate(image_names):
        img_path = os.path.join(root_dir, image_name, "{0}.png".format(image_name))
        assert os.path.isfile(img_path), "{} not exists!".format(img_path)
        logger.info('=> Processing {}'.format(image_name))

        txt_path = os.path.join(root_dir, image_name, "SARShip-1.mbrect")
        txt_data = open(txt_path, 'r').readlines()
        box = format_label(txt_data, class_list)

        # img_data = plt.imread(img_path)
        # BGR -> RGB
        img_data = cv2.imread(img_path)[:,:,::-1]
        image_name_split = image_name.split("-")
        file_index = image_name_split[0] + image_name_split[2]
        clip_image(file_index, img_data,
                   boxes_all=box,
                   height=height,
                   width=width,
                   stride=stride,
                   class_list=class_list,
                   save_dir=save_dir)
