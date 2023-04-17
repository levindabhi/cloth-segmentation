import os
import io
from PIL import Image
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import base64
import json
import datetime

from ts.torch_handler.base_handler import BaseHandler


class SegmentationHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.ctx = None
        self.initialized = False
        self.net = None
        self.device = None
    
    def initialize(self, ctx):
        # Initialize the handler with the given context
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
        # Load the PyTorch model from the serialized file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        self.net = torch.jit.load(model_pt_path)
        
        # Define the transforms to apply to the input image
        self.transforms_list = []
        self.transforms_list += [transforms.ToTensor()]
        self.transforms_list += [transforms.Lambda(lambda x: normalize_image(x, 0.5, 0.5))]
        self.transform_rgb = transforms.Compose(self.transforms_list)
        self.do_palette = True

        # Define the color palette to use for the output mask
        self.palette = get_palette(4)
        self.initialized = True

    
    def preprocess(self, data):
        # Preprocess the input data by converting the image to a tensor
        inputs = []
        #print(data)
        input_path = data[0].get('body')
        #print(input_path)
        if type(input_path)==bytearray:
            # If the input is in JSON format, extract the image data and decode it
            json_string = input_path.decode()
            data_dict = json.loads(json_string)
            image = Image.open(io.BytesIO(base64.b64decode(data_dict['instances'][0]['instance_key'])))
        else:
            # If the input is already a decoded image, extract the image data
            final_data = input_path['instances'][0]['instance_key']
            #print(final_data)
            image = Image.open(io.BytesIO(base64.b64decode(final_data)))

        # Preprocess image
        image_tensor = self.transform_rgb(image)
        inputs.append(image_tensor)

        return inputs


    def inference(self, data):
        # Predict output mask
        image_tensor = data.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            output_tensor = self.net.forward(image_tensor)
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

        output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
        if self.do_palette:
            output_img.putpalette(self.palette)

        return output_img
    
    def handle(self, data, context):
        # Preprocess input
        inputs = self.preprocess(data)
        # Perform inference
        outputs = []
        for idx, d in enumerate(inputs):
            start_time = time.time()
            output_img = self.inference(d)
            end_time = time.time()
            print(f'Time taken for inference {idx}: {end_time - start_time:.4f} seconds')

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            img_bytes = io.BytesIO()
            output_img.save(img_bytes, format='PNG')
            outputs.append({"body":base64.b64encode(img_bytes.getvalue()).decode('utf-8'), "content_type": "image/png"})

        return outputs
    
def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette

def normalize_image(image_tensor, mean, std):

    assert isinstance(mean, float)
    assert isinstance(std, float)

    if image_tensor.shape[0] == 1:
        normalize = transforms.Normalize([mean], [std])
    elif image_tensor.shape[0] == 3:
        normalize = transforms.Normalize([mean] * 3, [std] * 3)
    elif image_tensor.shape[0] == 18:
        normalize = transforms.Normalize([mean] * 18, [std] * 18)
    else:
        raise ValueError(f"Unsupported tensor shape {image_tensor.shape}")

    return normalize(image_tensor)
