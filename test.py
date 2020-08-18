# coding=utf-8
import argparse
import json
import copy
import torch
import os
import numpy as np
import torch.nn as nn
import cv2
import time

from thop import profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Darknet, attempt_download, load_darknet_weights, coco80_to_coco91_class, compute_loss
from utils.utils import torch_utils, non_max_suppression, xywh2xyxy, xyxy2xywh, scale_coords, clip_coords, floatn,bbox_iou, ap_per_class, plot_one_box
from utils.sarship_datasets import SarShipDataset


import warnings
# warnings.filterwarnings("ignore")
# torch.set_printoptions(profile="full")

def test(cfg, nc = 1, output_dir = '',
         dataloader = None,
         weights=None,
         batch_size=16,
         img_size=416,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5,
         save_json=False,
         model=None):


    #########
    # model #
    #########
    if not model:
        device = torch_utils.select_device(opt.device)
        verbose = True

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1 and opt.device != 'cpu':
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = False
    model.eval()

    # flops, params = profile(model, inputs=(torch.randn(1, 3, 416, 416).to(device), ))
    # print('flops is %d, param count is %d' %(flops, params))
    # exit(0)


    ##########
    # option #
    ##########
    seen = 0
    # coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    start_time = time.time()

    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)

    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

        targets = targets.to(device)
        imgs = imgs.to(device).float()
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Run model
        inf_out, train_out = model(imgs)  # inference outputs([batch size, 507 + 2028 + 8112, 6])

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)  # bs, [bounding box count, 7]

        # Statistics per image & Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls
        start_time2 = time.time()
        for si, pred in enumerate(output):

            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to pycocotools JSON dictionary
            # if save_json:
            #     # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            #     file_name = Path(paths[si]).stem
            #     # image_id = int(Path(paths[si]).stem.split('_')[-1])
            #     box = pred[:, :4].clone()  # xyxy
            #     scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
            #     box = xyxy2xywh(box)  # xywh
            #     box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            #     for di, d in enumerate(pred):
            #         jdict.append({'file_name': file_name,
            #                       'category_id': coco91class[int(d[6])],
            #                       'bbox': [floatn(x, 3) for x in box[di]],
            #                       'score': floatn(d[4], 5)})

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            img = SarShipDataset.load_image_from_path(paths[si], 416) * 255.0
            # img = copy.deepcopy(imgs[si]).permute([1, 2, 0]).cpu().numpy() * 255.0
            for bbox in pred:
                if(bbox[4] >= conf_thres):
                    x1y1 = (int(bbox[0]), int(bbox[1]))
                    x2y2 = (int(bbox[2]), int(bbox[3]))
                    color = (255, 0, 0)
                    img = cv2.rectangle(img, x1y1, x2y2,  color)
            cv2.imwrite(os.path.join(output_dir, os.path.split(paths[si])[1]), img)


            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
            """
            ([1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             tensor([0.9792, 0.9346, 0.4839, 0.1365, 0.1064, 0.0541, 0.0184, 0.0184, 0.0152, 0.0134]),
             tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), [0.0, 0.0, 0.0, 0.0])
            """
        static_time = time.time() - start_time2
    print('inference time is %f' % (time.time() - start_time - static_time))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy


    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    # if verbose and nc > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         print(pf % (test_dataset.names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    # if save_json and map and len(jdict):
    #     try:
    #         imgIds = [int(Path(x).stem.split('_')[-1]) for x in test_dataset.img_files]
    #         with open('results.json', 'w') as file:
    #             json.dump(jdict, file)
    #
    #         from pycocotools.coco import COCO
    #         from pycocotools.cocoeval import COCOeval
    #
    #         # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #         cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
    #         cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api
    #
    #         cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    #         cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
    #         cocoEval.evaluate()
    #         cocoEval.accumulate()
    #         cocoEval.summarize()
    #         map = cocoEval.stats[1]  # update mAP to pycocotools mAP
    #     except:
    #         print('WARNING: missing dependency pycocotools from requirements.txt. Can not compute official COCO mAP.')

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--nc', type=int, default=1, help='class number')
    parser.add_argument('--data_path', type=str, default='', help='data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    from utils.sarship_datasets import SarShipDataset

    test_dataset = SarShipDataset(data_dir=opt.data_path, is_train=False, img_size=opt.img_size, batch_size=opt.batch_size,
                                  cache=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=opt.batch_size,
                                                  num_workers=min([os.cpu_count(), opt.batch_size, 16]),
                                                  pin_memory=True,
                                                  collate_fn=test_dataset.collate_fn)

    with torch.no_grad():
        test(opt.cfg,
             opt.nc, os.path.join(opt.data_path, 'Valid/PredictOutPut/'),
             test_dataloader,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.iou_thres,
             opt.conf_thres,
             opt.nms_thres,
             opt.save_json)