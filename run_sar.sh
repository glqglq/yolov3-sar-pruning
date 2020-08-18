WORKDIR=.
GLOBAL_PERCENT=0.6
LAYER_KEEP=0.01
SHORTCUT=16
BATCH_SIZE=20
cd ${WORKDIR}

# conventional train
python3 train.py \
  --cfg cfg/yolov3-spp-hand.cfg \
  --data_path data/SARShip_pre_hepeng/ \
  --weights weights/official_weights/yolov3-spp.weights \
  --batch-size ${BATCH_SIZE} \
  --epochs 100
mkdir weights/before_prune_sar
mv weights/*.pt weights/before_prune_sar/
# convert conventional train file pt to weights
sudo python3 -c "from models import *; convert('cfg/yolov3-spp-hand.cfg', 'weights/before_prune_sar/last.pt')"
mv converted.weights weights/before_prune_sar/

# sparse train
python3 train.py \
  --cfg cfg/yolov3-spp-hand.cfg \
  --data_path data/SARShip_pre_hepeng/ \
  --weights weights/before_prune_sar/converted.weights \
  --batch-size ${BATCH_SIZE} \
  --epochs 300 \
  --s 0.001 \
  -sr \
  --prune 1
mkdir weights/before_prune_sparse_sar
mv weights/*.pt weights/before_prune_sparse_sar/
# convert conventional train file pt to weights
sudo python3 -c "from models import *; convert('cfg/yolov3-spp-hand.cfg', 'weights/before_prune_sparse_sar/last.pt')"
mv converted.weights weights/before_prune_sparse_sar/

# prune
python3 layer_channel_prune.py \
  --cfg cfg/yolov3-spp-hand.cfg \
  --data_path data/SARShip_pre_hepeng/ \
  --weights weights/before_prune_sparse_sar/last.pt \
  --global_percent ${GLOBAL_PERCENT} \
  --layer_keep ${LAYER_KEEP} \
  --shortcuts ${SHORTCUT}
# --------> before map0.897765  param62573334  after prune channels: map0.570068  param40049144  after prune layers: map0.010232  param30247548

# finetune
python3 train.py \
  --cfg cfg/prune_${GLOBAL_PERCENT}_keep_${LAYER_KEEP}_${SHORTCUT}_shortcut_yolov3-spp-hand.cfg \
  --data_path data/SARShip_pre_hepeng/ \
  --weights weights/before_prune_sparse_sar/last.weights \
  --batch-size ${BATCH_SIZE} \
  --epochs 100
mkdir weights/after_finetune_sar
mv weights/*.pt weights/after_finetune_sar/
# convert conventional train file pt to weights
sudo python3 -c "from models import *; convert('cfg/prune_${GLOBAL_PERCENT}_keep_${LAYER_KEEP}_${SHORTCUT}_shortcut_yolov3-spp-hand.cfg', 'weights/after_finetune_sar/last.pt')"
mv converted.weights weights/after_finetune_sar/

# test compressed model on gpu
python3 test.py \
  --cfg cfg/prune_${GLOBAL_PERCENT}_keep_${LAYER_KEEP}_${SHORTCUT}_shortcut_yolov3-spp-hand.cfg \
  --data_path data/SARShip_pre_hepeng/ \
  --weights weights/after_finetune_sar/converted.weights \
  --batch-size 1 \
  --save-json \
  --device 0
# to mlu100
python genoff.py \
  -cfg_path prune_${GLOBAL_PERCENT}_keep_${LAYER_KEEP}_${SHORTCUT}_shortcut_yolov3-spp-hand.cfg \
  -mcore MLU100 \
  -offline_model_path yolov3-compressed \
  -core_number 32 \
  -batch_size 32
# test compressed model on mlu100
/home/Cambricon-MLU100/pytorch/src/pytorch/test/offline_examples/build/yolo_v3/yolov3_offline_multicore \
  -offlinemodel yolov3-compressed.cambricon \
  -images data/SARShip_pre_hepeng/valid.txt \
  -labels data/SARShip_pre_hepeng/all.names \
  -resize 416,416 \
  -outputdir pred \
  -dump 1 \
  -simple_compile 1
# test origin model on gpu
python3 test.py \
  --cfg cfg/yolov3-spp-hand.cfg \
  --data_path data/SARShip_pre_hepeng/ \
  --weights weights/before_prune_sparse_sar/converted.weights \
  --batch-size 1 \
  --save-json \
  --device 0



