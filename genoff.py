# coding=utf-8
from __future__ import division
import torch

import os
import logging
import argparse

#configure logging path
logging.basicConfig(level=logging.INFO,
    format='[genoff.py line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("TestNets")


def get_args():
    parser = argparse.ArgumentParser(description='Generate offline model.')
    parser.add_argument("-cfg_path", dest = 'cfg_path', help = "Yolo Cfg Path", default = "", type = str)
    parser.add_argument("-weights_path", dest='weights_path', help="Yolo Weights Path", default="", type=str)
    parser.add_argument("-parallel", dest = "parallel", help =
                        "Model parallelism of offline model. \n" +
                        "model parallel means one model runs in multicores. ",
                        default = 1,type = int)
    parser.add_argument("-core_number", dest="core_number", help=
                       "Core number of offline model with simple compilation. ",
                        default = 0, type = int)
    parser.add_argument("-offline_model_path", dest = 'offline_model_path', help =
                        "The path for the offline model to be generated",
                        default = "offline", type = str)
    parser.add_argument("-mcore", dest = 'mcore', help =
                        "Specify the offline model run device type.",
                        default = "MLU200", type = str)
    parser.add_argument("-modelzoo", dest = 'modelzoo', type = str,
                        help = "Specify the path to the model weight file.",
                        default = None)
    parser.add_argument("-batch_size", dest = "batch_size", help =
                        "batch size for one inference.",
                        default = 1,type = int)
    return parser.parse_args()

def genoff(cfg_path, weights_path, parallel, offline_model_path, batch_size, core_number):

    in_h = 416
    in_w = 416

    # from models_for_mlu_me import yolov3
    # from models_for_mlu_official import yolov3
    # net = yolov3(cfg=cfg_path, pretrained=False, img_size=in_h, conf_thres=0.001, nms_thres=0.5)
    import torchvision.models as torchvision_models
    # net = torchvision_models.object_detection.Darknet(cfg_path, in_h, conf_thres=0.001, nms_thres=0.5).eval()
    net = torchvision_models.object_detection.yolov3(False, in_h, conf_thres=0.001, nms_thres=0.5).eval()
    net.load_weights(weights_path)

    # Generate offline model
    net_mlu = net.eval().float().mlu()
    net_mlu.set_model_parallelism(parallel)
    net_mlu.set_core_number(core_number)
    example_mlu = torch.randn(batch_size, 3, in_h, in_w, dtype=torch.float).mlu()

    net = torch.jit.trace(net, torch.randn(1, 3, in_h, in_w, dtype=torch.float).mlu(), check_trace=False)
    net(example_mlu)
    net.save(offline_model_path, True)

if __name__ == "__main__":
    # options
    args = get_args()
    cfg_path = args.cfg_path
    parallel = args.parallel
    weights_path = args.weights_path
    core_number = args.core_number
    offline_model_path = args.offline_model_path
    modelzoo = args.modelzoo
    mcore = args.mcore
    batch_size = args.batch_size
    if modelzoo != None:
       os.environ['TORCH_MODEL_ZOO'] = modelzoo
       logger.info("TORCH_MODEL_ZOO: " + modelzoo)
    os.putenv('ATEN_CNML_COREVERSION', mcore);

    #check param
    assert mcore in ['MLU100', 'MLU200'], "The specified device type is not supported."

    #genoff
    logger.info("ATEN_CNML_COREVERSION: " + mcore)
    logger.info(" model parallelism: " + str(parallel))
    genoff(cfg_path, weights_path, parallel, offline_model_path, batch_size, core_number)