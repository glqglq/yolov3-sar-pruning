FROM pytorch/pytorch

WORKDIR /workspace

RUN pip install -r requirements.txt && \
apt-get update && \
apt-get install -y libglib2.0-dev && \
apt-get install -y libsm6 && \
apt-get install -y libxrender1 && \
apt-get install -y libxext6


#RUN python3 test.py \
#--cfg cfg/prune_0.93_keep_0.01_16_shortcut_yolov3-spp-hand.cfg \
#--data data/oxfordhand.data \
#--weights weights/after_finetune/best.weights \
#--batch-size 1 \
#--save-json \
#--device 0,1


# libgthread-2.0.so.0
# libSM.so.6
# libXrender.so.1
# libXext.so.6