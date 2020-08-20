FROM pytorch_env

WORKDIR /workspace

COPY / /workspace/

RUN pip install opencv-python thop tqdm matplotlib