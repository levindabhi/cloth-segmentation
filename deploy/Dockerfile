# pull the latest torchserve image
FROM pytorch/torchserve:latest-cpu

# copy model artifacts, custom handler and other dependencies
COPY ./handler.py /home/model-server/
COPY ./torchscript.pt /home/model-server/

# create torchserve configuration file
USER root
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=cloth_seg \
  --version=1.0 \
  --serialized-file=/home/model-server/torchscript.pt \
  --handler=/home/model-server/handler.py \
  --export-path=/home/model-server/model-store

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "cloth_seg=cloth_seg.mar", \
     "--model-store", \
     "/home/model-server/model-store"]