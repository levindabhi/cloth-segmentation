# Clothes Segmentation using U2NET #

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EhEy3uQh-5oOSagUotVOJAf8m7Vqn0D6?usp=sharing)

This repo contains training code, inference code and pre-trained model for Cloths Parsing from human portrait.</br>
Here clothes are parsed into 3 category: Upper body(red), Lower body(green) and Full body(yellow)

![Sample 000](assets/000.png)
![Sample 024](assets/024.png)
![Sample 018](assets/018.png)

This model works well with any background and almost all poses. For more samples visit [samples.md](samples.md)

# Techinal details

* **U2NET** : This project uses an amazing [U2NET](https://arxiv.org/abs/2005.09007) as a deep learning model. Instead of having 1 channel output from u2net for typical salient object detection task it outputs 4 channels each respresting upper body cloth, lower body cloth, fully body cloth and background. Only categorical cross-entropy loss is used for a given version of the checkpoint.

* **Dataset** : U2net is trained on 45k images [iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) dataset. To reduce complexity, I have clubbed the original 42 categories from dataset labels into 3 categories (upper body, lower body and full body). All images are resized into square `¯\_(ツ)_/¯` 768 x 768 px for training. (This experiment was conducted with 768 px but around 384 px will work fine too if one is retraining on another dataset).

# Training 

- For training this project requires,
<ul>
    <ul>
    <li>&nbsp; PyTorch > 1.3.0</li>
    <li>&nbsp; tensorboardX</li>
    <li>&nbsp; gdown</li>
    </ul>
</ul>

- Download dataset from this [link](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data), extract all items.
- Set path of `train` folder which contains training images and `train.csv` which is label csv file in `options/base_options.py`
- To port original u2net of all layer except last layer please run `python setup_model_weights.py` and it will generate weights after model surgey in `prev_checkpoints` folder.
- You can explore various options in `options/base_options.py` like checkpoint saving folder, logs folder etc.
- For single gpu set `distributed = False` in `options/base_options.py`, for multi gpu set it to `True`.
- For single gpu run `python train.py`
- For multi gpu run <br>
&nbsp;`python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=4 --use_env train.py` <br>
Here command is for single node, 4 gpu. Tested only for single node.
- You can watch loss graphs and samples in tensorboard by running tensorboard command in log folder.


# Testing/Inference
- Download pretrained model from this [link](https://drive.google.com/file/d/1mhF3yqd7R-Uje092eypktNl-RoZNuiCJ/view?usp=sharing)(165 MB) in `trained_checkpoint` folder.
- Put input images in `input_images` folder
- Run `python infer.py` for inference.
- Output will be saved in `output_images`
### OR 
- Inference in colab from here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EhEy3uQh-5oOSagUotVOJAf8m7Vqn0D6?usp=sharing)

# Deploy on Vertex AI with Torchserve
- Download pretrained model of `.pt` format from this [link](https://drive.google.com/file/d/1Ee4igrf5axte9nV1KvcgtEnO7KPKaVm6/view?usp=sharing)(169 MB) in `deploy` folder.
- Build the docker image using the Dockerfile inside the `deploy` folder using command(replace `<project-id>` with the id of your google cloud project):
```
docker build -t gcr.io/<project-id>/pytorch_predict_cloth_seg .
```
- Authenticate with the Google Cloud SDK by running the command `gcloud auth configure-docker`.
- Push your Docker image to the Google Cloud Registry using the command `docker push gcr.io/(project-id)/pytorch_predict_cloth_seg`.
- Create a new custom model on vertex ai using this commad in google cloud sdk(replace `<location>` and `<project-id>`):
```
gcloud ai models upload \
  --container-ports=7080 \
  --container-predict-route="/predictions/cloth_seg" \
  --container-health-route="/ping" \
  --region=<location> \
  --display-name=cloth_seg \
  --container-image-uri=gcr.io/<project-id>/pytorch_predict_cloth_seg
```
- Create an endpoint using this command in google cloud sdk(replace `<location>` and `<project-id>`):
```
gcloud ai endpoints create \
  --project=<project-id> \
  --region=<location> \
  --display-name=cloth_seg
```
- Deploy the model on the above created endpoint using the following command in google cloud sdk(replace `<endpoint-id>`(endpoint id from the endpoint we created in above steps), `<location>`, `<project-id>`, `<model-id>`(model id from the model we created in above steps) and `<machine-type>`):
```
gcloud ai endpoints deploy-model <endpoint-id> \
  --project=<project-id> \
  --region=<location> \
  --model=<model-id> \
  --traffic-split=0=100 \
  --machine-type="<machine-type>" \
  --display-name=cloth_seg
```
- Download the required libraries for inference using `pip install pillow google`.
- Test the deployed model using the `deployed_infer.py` file:
```
python deployed_infer.py --input <path/to/input/image> --output <path/to/output/folder> --project <project-id> --location <location> --project_number <project-number> --endpoint_id <endpoind-id>
```
Example:
```
python deployed_infer.py --input image.jpg --output output/ --project demo-project-374930 --location asia-south1 --project_number 63********42 --endpoint_id 2129************216	
```
To get info about the arguments:
```
python deployed_infer.py --help
```

# Acknowledgements
- U2net model is from original [u2net repo](https://github.com/xuebinqin/U-2-Net). Thanks to Xuebin Qin for amazing repo.
- Complete repo follows structure of [Pix2pixHD repo](https://github.com/NVIDIA/pix2pixHD)

