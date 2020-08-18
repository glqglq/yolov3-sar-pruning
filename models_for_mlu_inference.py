from __future__ import division, absolute_import

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils import model_zoo
from collections import defaultdict

from utils.parse_config import parse_model_cfg


model_urls = {
    'yolov3' : 'yolov3.pth',
}
#yolov3_url = "https://pjreddie.com/media/files/yolov3.weights"

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layes

    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))
            elif module_def["activation"] == "mish":
                modules.add_module("activation", Mish())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module("_debug_padding_%d" % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            # upsample = nn.functional.interpolate(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":  # nn.Sequential() placeholder for "route" layer
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[layer_i + 1 if layer_i > 0 else layer_i] for layer_i in layers])
            if "groups" in module_def:
                filters = filters // 2
            routs.extend([l if l > 0 else l + i for l in layers])
            modules.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":  # nn.Sequential() placeholder for "shortcut" layer
            filters = output_filters[int(module_def["from"])]
            layer = int(module_def["from"])
            routs.extend([i + layer if layer < 0 else layer])
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "reorg3d":  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
            pass

        elif module_def["type"] == "yolo":
            save_val = i
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]  # anchor mask
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            nG = int(module_def["input_size"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, nG)
            modules.add_module("yolo_%d" % save_val, yolo_layer)

        else:
            print("Warning: Unrecognized Layer Type: " + module_def["type"])
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, hyperparams


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul(torch.tanh(F.softplus(x)))

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim, nG):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

        # add parameter
        nA = self.num_anchors
        self.nG = nG
        self.stride = self.image_dim / nG
        self.grid_x = Parameter(torch.arange(nG).repeat(nG, 1).reshape([1, 1, nG*nG, 1]).float())
        self.grid_y = Parameter(torch.arange(nG).repeat(nG, 1).t().contiguous().reshape([1, 1, nG*nG, 1]).float())
        self.scaled_anchors = Parameter(torch.FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]))
        self.anchor_w = Parameter(self.scaled_anchors[:, 0:1].reshape((1, nA, 1, 1)))
        self.anchor_h = Parameter(self.scaled_anchors[:, 1:2].reshape((1, nA, 1, 1)))

    def forward(self, x):
        nA = self.num_anchors
        nB = x.size(0)
        # print(nB, nA, self.bbox_attrs, self.nG* self.nG)
        # exit(0)
        prediction = x.reshape(nB, nA, self.bbox_attrs, self.nG* self.nG).permute(0, 1, 3, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0:1])  # Center x
        y = torch.sigmoid(prediction[..., 1:2]) # Center y
        w = prediction[..., 2:3]  # Width
        h = prediction[..., 3:4]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4:5])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.


        # Calculate offsets for each grid
        # moved to init

        # Add offset and scale with anchors
        pred_boxes = torch.cat(
                (
                       x + self.grid_x,
                       y + self.grid_y,
                       torch.exp(w) * self.anchor_w,
                       torch.exp(h) * self.anchor_h
                       ), 3
                )

        # If not in training phase return predictions
        output = torch.cat(
            (
                pred_boxes.reshape(nB, -1, 4) * self.stride,
                pred_conf.reshape(nB, -1, 1),
                pred_cls.reshape(nB, -1, self.num_classes),
            ),
            -1,
        )
        # print(output.shape)
        # exit(0)
        return output


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg, img_size=416, conf_thres=0.8, nms_thres=0.4):
        super(Darknet, self).__init__()

        if isinstance(cfg, str):
            self.module_defs = parse_model_cfg(cfg)
        elif isinstance(cfg, list):
            self.module_defs = cfg
        self.module_list, self.hyperparams = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.num_anchors = 0
        self.num_classes = 0
        self.nG = 0
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.maxBoxNum = 15360;
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x, debug = False):
        # print(x.cpu().shape, x.cpu())
        # exit(0)
        self.losses = defaultdict(float)
        layer_outputs = []
        output = []
        anchors = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                    if "groups" in module_def:
                        x = x[:, (x.shape[1]//2):]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layer_i], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layer_i[1]] = F.interpolate(layer_outputs[layer_i[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                anchor_idxs = [int(val) for val in module_def["mask"].split(",")]
                # Extract anchors
                anchor = [int(val) for val in module_def["anchors"].split(",")]
                anchor = [(anchor[j], anchor[j + 1]) for j in range(0, len(anchor), 2)]
                anchor = [anchor[j] for j in anchor_idxs]
                for element1, element2 in anchor:
                    anchors.append(element1)
                    anchors.append(element2)
                self.num_classes = int(module_def["classes"])
                self.nG = int(module_def["input_size"])
                # x = module(x)  # [[bs, 18, 13, 13],[bs, 18, 26, 26],[bs, 18, 52, 52]]  -> [[bs, 507, 6]]
                # print(y.shape)
                # prediction = x.reshape(1, 3, 6, 169).permute(0, 1, 3, 2).contiguous().reshape(1, 507, 6)
                # print(prediction.cpu().shape, prediction.cpu())
                # exit()
                output.append(x)
            else:
                raise Exception()
            layer_outputs.append(x)
        # exit(0)

            # print(module_def["type"], x.shape)
        self.losses["recall"] /= 3
        self.losses["precision"] /= 3


        self.num_anchors = len(anchors)

        # output = [output[0].reshape(), output[1].reshape(), output[2].reshape()]
        # print(x[0].cpu().reshape([1, 3, 6, 13, 13]).transpose())
        # print(x[1].cpu().reshape([1, 3, 6, 26, 26]))
        # print(x[2].cpu().reshape([1, 3, 6, 52, 52]))
        # print(output[0].shape)
        # print(output[1].shape)
        # print(output[2].shape)
        # exit(0)

        detect_out = torch.mlu_yolov3_detection_output(output[0], output[1], output[2],
                          tuple(anchors), self.num_classes, self.num_anchors,
                          self.img_size, self.conf_thres, self.nms_thres, self.maxBoxNum)
        # print(detect_out.cpu().shape, detect_out.cpu())
        # exit(0)

        return detect_out


    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """

    def save_weights(self, path, cutoff=-1):

        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

def yolov3(cfg, pretrained=False, img_size=416, conf_thres=0.8, nms_thres=0.4):
    r"""Yolov3 model architecture from ModelZoo

    Args:
        pretrained (bool): If True, returns a model pre-trained
        img_size: image size
    """

    cwd = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(cwd, 'config', cfg)
    model = Darknet(config_path, img_size, conf_thres, nms_thres).eval()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['yolov3']))

    return model
