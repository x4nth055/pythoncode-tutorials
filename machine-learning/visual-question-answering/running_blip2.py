# %%
!pip install transformers accelerate

# %%
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os

device = torch.device("cuda", 0)
device

# %%
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

# %%
model.to(device)

# %%
import urllib.parse as parse
import os

# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
    
# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)

# %%
raw_image = load_image("http://images.cocodataset.org/test-stuff2017/000000007226.jpg")

# %%
question = "a"
inputs = processor(raw_image, question, return_tensors="pt").to(device, dtype=torch.float16)

# %%
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# %%
question = "a vintage car driving down a street"
inputs = processor(raw_image, question, return_tensors="pt").to(device, dtype=torch.float16)

# %%
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# %%
question = "Question: What is the estimated year of these cars? Answer:"
inputs = processor(raw_image, question, return_tensors="pt").to(device, dtype=torch.float16)

# %%
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# %%
question = "Question: What is the color of the car? Answer:"
inputs = processor(raw_image, question, return_tensors="pt").to(device, dtype=torch.float16)

# %%
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# %%



