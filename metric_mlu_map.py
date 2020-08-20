import os
import numpy as np
import torch

from utils.sarship_datasets import SarShipDataset
from utils.utils import clip_coords, bbox_iou, xywh2xyxy
from utils.utils import ap_per_class

class Metric(SarShipDataset):
    def __init__(self, data_label_dir, mlu_pred_dir, iou_thres = 0.5, nc = 1):

        super(Metric, self).__init__(data_label_dir, is_train = False, img_size=416, batch_size=20, cache = True)

        assert os.path.exists(mlu_pred_dir)

        self.iou_thres = iou_thres
        self.nc = nc

        mlu_pred_filenames = [path for path in os.listdir(mlu_pred_dir) if path.endswith('txt')]

        self.name_bboxes_dict = {os.path.splitext(os.path.split(item[2])[1])[0]: item[1] for item in self.img_ann}

        self.cal_map(mlu_pred_dir, self.name_bboxes_dict)

    def cal_map(self, mlu_pred_dir, true_name_bboxes_dict):

        stats = []


        for filename in true_name_bboxes_dict.keys():
            # true label
            tcls = true_name_bboxes_dict[filename][:, 0].tolist()
            tcls_tensor = true_name_bboxes_dict[filename][:, 0]
            # target boxes
            tbox = xywh2xyxy(true_name_bboxes_dict[filename][:, 2:6])
            tbox[:, [0, 2]] *= self.img_size
            tbox[:, [1, 3]] *= self.img_size

            # pred label
            pred_bboxes = []
            with open(os.path.join(mlu_pred_dir, filename + '.txt')) as f:
                for line in f:
                    class_name, conf, x1, y1, x2, y2, _, _ = line.strip().split()
                    conf = float(conf)
                    x1 = float(x1) * self.img_size
                    y1 = float(y1) * self.img_size
                    x2 = float(x2) * self.img_size
                    y2 = float(y2) * self.img_size
                    pred_bboxes.append([x1, y1, x2, y2, conf, 1.0, 0.0])
            pred_bboxes = torch.Tensor(pred_bboxes)
            # clip_coords(pred_bboxes, (self.img_size, self.img_size))

            # print(filename, pred_bboxes)
            # print(tbox)
            # print(tcls_tensor)
            # print(tcls)
            # exit()

            # metric
            correct = [0] * len(pred_bboxes)

            detected = []

            # Search for correct predictions
            for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred_bboxes):

                # Best iou, index between pred and targets
                m = (pcls == tcls_tensor).nonzero().view(-1)
                # print(m)
                iou, bi = bbox_iou(pbox, tbox[m]).max(0)
                # print(iou, bi)

                # If iou > threshold and class is correct mark as correct
                if iou > self.iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                    correct[i] = 1
                    detected.append(m[bi])
            # print(detected)
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred_bboxes[:, 4].cpu(), pred_bboxes[:, 6].cpu(), tcls))

        stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        self.mp, self.mr, self.map, self.mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        self.nt = np.bincount(stats[3].astype(np.int64), minlength=self.nc)  # number of targets per class



if __name__ == '__main__':
    m = Metric('data/SARShip_pre_hepeng/', 'data/pred_416/')
    print(m.mp, m.mr, m.map, m.mf1, m.nt)