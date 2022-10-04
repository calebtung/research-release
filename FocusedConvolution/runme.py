import copy
from torchbench.image_classification import ImageNet
from torchvision.models import vgg16, VGG16_Weights
from enum import Enum
import torchvision.transforms as transforms
import PIL
import torch
from focusedconv import AoIAdjuster, AoISizeEstimator

from utils import focusify_all_conv2d
from torchinfo import summary

TOP_LAYERS_TO_USE = 4
ACTIVATION_BRIGHTNESS_THRESH = 90
DATASET_PATH = "datasets/imagenet/"

# Define the transforms need to convert dataset to expected
# model input
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


mask_holder = {"aoi_mask": None}
model = vgg16(weights=VGG16_Weights.DEFAULT)
top_layers = copy.deepcopy(model.features[0:TOP_LAYERS_TO_USE])

# This automatically converts Conv2d in a model to a focused convolution
focusify_all_conv2d(model, mask_holder)

# Reassemble the model for auto-AoI generation
model.features = torch.nn.Sequential(top_layers, AoIAdjuster(mask_holder, ACTIVATION_BRIGHTNESS_THRESH), model.features[TOP_LAYERS_TO_USE:])

# This model is ready for Focused Conv benchmarking!
summary(model, (1,3,224,224))


print("\nCOLLECTING AOI INFO for activation brightness threshold of", ACTIVATION_BRIGHTNESS_THRESH)
dataset_aois = []

# This module will automatically append estimated AoI percentages to dataset_aois; it does not actually impact focused conv
aoi_measurer = torch.nn.Sequential(top_layers, AoISizeEstimator(mask_holder, ACTIVATION_BRIGHTNESS_THRESH, dataset_aois, (1,1000)))

try:
    ImageNet.benchmark(
        model=aoi_measurer,
        input_transform=input_transform,
        batch_size=1, # Don't touch this
        device="cuda",
        data_root=DATASET_PATH
    )
except:
    pass
print("AVERAGE AOI AREA PERCENTAGE:", sum(dataset_aois)/len(dataset_aois))

print("MEASURING ACCURACY...")
try:
    ImageNet.benchmark(
        model=model,
        input_transform=input_transform,
        batch_size=1, # Don't touch this
        device="cuda",
        data_root=DATASET_PATH
    )
except:
    pass

print("MEASURING ACCURACY...")
try:
    ImageNet.benchmark(
        model=model,
        input_transform=input_transform,
        batch_size=1, # Don't touch this
        device="cpu",
        data_root=DATASET_PATH
    )
except:
    pass
