# Import necessary libraries
from networks import U2NET
import torch
import torchvision.transforms as transforms
from collections import OrderedDict
from PIL import Image

# Define the image transformation pipeline
transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [transforms.Lambda(lambda x: normalize_image(x, 0.5, 0.5))] # Normalize the image tensor
transform_rgb = transforms.Compose(transforms_list)

# Load the pre-trained U2NET model
model = U2NET(in_ch=3, out_ch=4)
model_state_dict = torch.load('trained_checkpoint/cloth_segm_u2net_latest.pth', map_location=torch.device("cpu"))
new_state_dict = OrderedDict()
for k, v in model_state_dict.items():
  name = k[7:]
  new_state_dict[name] = v

model.load_state_dict(new_state_dict)

# Load the input image and apply the transformation pipeline
image = Image.open('image.jpg')
image_tensor = transform_rgb(image)

# Define a helper function to normalize the image tensor
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
