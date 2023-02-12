# %% [markdown]
# # Set up environment

# %%
!pip install transformers

# %%
from IPython.display import clear_output
# !pip3 install transformers
clear_output()

# %%
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import pipeline, SegformerImageProcessor, SegformerForSemanticSegmentation
import requests
from PIL import Image
import urllib.parse as parse
import os

# %%
# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

# a function to load an image
def load_image(image_path):
    """Helper function to load images from their URLs or paths."""
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)

# %% [markdown]
# # Load Image

# %%
img_path = "https://shorthaircatbreeds.com/wp-content/uploads/2020/06/Urban-cat-crossing-a-road-300x180.jpg"
image = load_image(img_path)

# %%
image

# %%
# convert PIL Image to pytorch tensors
transform = transforms.ToTensor()
image_tensor = image.convert("RGB")
image_tensor = transform(image_tensor)
image_tensor.shape

# %% [markdown]
# # Helper functions

# %%
def color_palette():
  """Color palette to map each class to its corresponding color."""
  return [[0, 128, 128],
          [255, 170, 0],
          [161, 19, 46],
          [118, 171, 47],
          [255, 255, 0],
          [84, 170, 127],
          [170, 84, 127],
          [33, 138, 200],
          [255, 84, 0],
          [255, 140, 208]]

# %%
def overlay_segments(image, seg_mask):
  """Return different segments predicted by the model overlaid on image."""
  H, W = seg_mask.shape
  image_mask = np.zeros((H, W, 3), dtype=np.uint8)
  colors = np.array(color_palette())

  # convert to a pytorch tensor if seg_mask is not one already
  seg_mask = seg_mask if torch.is_tensor(seg_mask) else torch.tensor(seg_mask)
  unique_labels = torch.unique(seg_mask)

  # map each segment label to a unique color
  for i, label in enumerate(unique_labels):
    image_mask[seg_mask == label.item(), :] = colors[i]

  image = np.array(image)
  # percentage of original image in the final overlaid iamge
  img_weight = 0.5 

  # overlay input image and the generated segment mask
  img = img_weight * np.array(image) * 255 + (1 - img_weight) * image_mask

  return img.astype(np.uint8)

# %%
def replace_label(mask, label):
  """Replace the segment masks values with label."""
  mask = np.array(mask)
  mask[mask == 255] = label
  return mask

# %% [markdown]
# # Image segmentation using Hugging Face Pipeline

# %%
# load the entire image segmentation pipeline
img_segmentation_pipeline = pipeline('image-segmentation', 
                                     model="nvidia/segformer-b5-finetuned-ade-640-640")

# %%
output = img_segmentation_pipeline(image)
output

# %%
output[0]['mask']

# %%
output[2]['mask']

# %%
# load the feature extractor (to preprocess images) and the model (to get outputs)
W, H = image.size
segmentation_mask = np.zeros((H, W), dtype=np.uint8)

for i in range(len(output)):
  segmentation_mask += replace_label(output[i]['mask'], i)

# %%
# overlay the predicted segmentation masks on the original image
segmented_img = overlay_segments(image_tensor.permute(1, 2, 0), segmentation_mask)

# convert to PIL Image
Image.fromarray(segmented_img)

# %% [markdown]
# # Image segmentation using custom Hugging Face models

# %%
# load the feature extractor (to preprocess images) and the model (to get outputs)
feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

# %%
def to_tensor(image):
  """Convert PIL Image to pytorch tensor."""
  transform = transforms.ToTensor()
  image_tensor = image.convert("RGB")
  image_tensor = transform(image_tensor)
  return image_tensor

# a function that takes an image and return the segmented image
def get_segmented_image(model, feature_extractor, image_path):
  """Return the predicted segmentation mask for the input image."""
  # load the image
  image = load_image(image_path)
  # preprocess input
  inputs = feature_extractor(images=image, return_tensors="pt")
  # convert to pytorch tensor
  image_tensor = to_tensor(image)
  # pass the processed input to the model
  outputs = model(**inputs)
  print("outputs.logits.shape:", outputs.logits.shape)
  # interpolate output logits to the same shape as the input image
  upsampled_logits = F.interpolate(
      outputs.logits, # tensor to be interpolated
      size=image_tensor.shape[1:], # output size we want
      mode='bilinear', # do bilinear interpolation
      align_corners=False)

  # get the class with max probabilities
  segmentation_mask = upsampled_logits.argmax(dim=1)[0]
  print(f"{segmentation_mask.shape=}")
  # get the segmented image
  segmented_img = overlay_segments(image_tensor.permute(1, 2, 0), segmentation_mask)
  # convert to PIL Image
  return Image.fromarray(segmented_img)

# %%
get_segmented_image(model, feature_extractor, "https://shorthaircatbreeds.com/wp-content/uploads/2020/06/Urban-cat-crossing-a-road-300x180.jpg")

# %%
get_segmented_image(model, feature_extractor, "http://images.cocodataset.org/test-stuff2017/000000000001.jpg")

# %%



