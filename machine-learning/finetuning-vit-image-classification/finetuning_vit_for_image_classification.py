# %%
!pip install transformers evaluate datasets

# %%
import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# the model name
model_name = "google/vit-base-patch16-224"
# load the image processor
image_processor = ViTImageProcessor.from_pretrained(model_name)
# loading the pre-trained model
model = ViTForImageClassification.from_pretrained(model_name).to(device)

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
def get_prediction(model, url_or_path):
  # load the image
  img = load_image(url_or_path)
  # preprocessing the image
  pixel_values = image_processor(img, return_tensors="pt")["pixel_values"].to(device)
  # perform inference
  output = model(pixel_values)
  # get the label id and return the class name
  return model.config.id2label[int(output.logits.softmax(dim=1).argmax())]

# %%
get_prediction(model, "http://images.cocodataset.org/test-stuff2017/000000000128.jpg")

# %% [markdown]
# # Loading our Dataset

# %%
from datasets import load_dataset

# download & load the dataset
ds = load_dataset("food101")

# %% [markdown]
# ## Loading a Custom Dataset using `ImageFolder`
# Run the three below cells to load a custom dataset (that's not in the Hub) using `ImageFolder`

# %%
import requests
from tqdm import tqdm

def get_file(url):
  response = requests.get(url, stream=True)
  total_size = int(response.headers.get('content-length', 0))
  filename = None
  content_disposition = response.headers.get('content-disposition')
  if content_disposition:
      parts = content_disposition.split(';')
      for part in parts:
          if 'filename' in part:
              filename = part.split('=')[1].strip('"')
  if not filename:
      filename = os.path.basename(url)
  block_size = 1024 # 1 Kibibyte
  tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
  with open(filename, 'wb') as file:
      for data in response.iter_content(block_size):
          tqdm_bar.update(len(data))
          file.write(data)
  tqdm_bar.close()
  print(f"Downloaded {filename} ({total_size} bytes)")
  return filename

# %%
import zipfile
import os

def download_and_extract_dataset():
  # dataset from https://github.com/udacity/dermatologist-ai
  # 5.3GB
  train_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip"
  # 824.5MB
  valid_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip"
  # 5.1GB
  test_url  = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip"
  for i, download_link in enumerate([valid_url, train_url, test_url]):
    data_dir = get_file(download_link)
    print("Extracting", download_link)
    with zipfile.ZipFile(data_dir, "r") as z:
      z.extractall("data")
    # remove the temp file
    os.remove(data_dir)

# comment the below line if you already downloaded the dataset
download_and_extract_dataset()

# %%
from datasets import load_dataset

# load the custom dataset
ds = load_dataset("imagefolder", data_dir="data")

# %% [markdown]
# # Exploring the Data

# %%
ds

# %%
labels = ds["train"].features["label"]
labels

# %%
labels.int2str(ds["train"][532]["label"])

# %%
import random
import matplotlib.pyplot as plt

def show_image_grid(dataset, split, grid_size=(4,4)):
    # Select random images from the given split
    indices = random.sample(range(len(dataset[split])), grid_size[0]*grid_size[1])
    images = [dataset[split][i]["image"] for i in indices]
    labels = [dataset[split][i]["label"] for i in indices]
    
    # Display the images in a grid
    fig, axes = plt.subplots(nrows=grid_size[0], ncols=grid_size[1], figsize=(8,8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')
        ax.set_title(ds["train"].features["label"].int2str(labels[i]))
    
    plt.show()

# %%
show_image_grid(ds, "train")

# %% [markdown]
# # Preprocessing the Data

# %%
def transform(examples):
  # convert all images to RGB format, then preprocessing it
  # using our image processor
  inputs = image_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
  # we also shouldn't forget about the labels
  inputs["labels"] = examples["label"]
  return inputs

# %%
# use the with_transform() method to apply the transform to the dataset on the fly during training
dataset = ds.with_transform(transform)

# %%
for item in dataset["train"]:
  print(item["pixel_values"].shape)
  print(item["labels"])
  break

# %%
# extract the labels for our dataset
labels = ds["train"].features["label"].names
labels

# %%
import torch

def collate_fn(batch):
  return {
      "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
      "labels": torch.tensor([x["labels"] for x in batch]),
  }

# %% [markdown]
# # Defining the Metrics

# %%
from evaluate import load
import numpy as np

# load the accuracy and f1 metrics from the evaluate module
accuracy = load("accuracy")
f1 = load("f1")

def compute_metrics(eval_pred):
  # compute the accuracy and f1 scores & return them
  accuracy_score = accuracy.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)
  f1_score = f1.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids, average="macro")
  return {**accuracy_score, **f1_score}

# %% [markdown]
# # Training the Model

# %%
# load the ViT model
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True,
)

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./vit-base-food", # output directory
  # output_dir="./vit-base-skin-cancer",
  per_device_train_batch_size=32, # batch size per device during training
  evaluation_strategy="steps",    # evaluation strategy to adopt during training
  num_train_epochs=3,             # total number of training epochs
  # fp16=True,                    # use mixed precision
  save_steps=1000,                # number of update steps before saving checkpoint
  eval_steps=1000,                # number of update steps before evaluating
  logging_steps=1000,             # number of update steps before logging
  # save_steps=50,
  # eval_steps=50,
  # logging_steps=50,
  save_total_limit=2,             # limit the total amount of checkpoints on disk
  remove_unused_columns=False,    # remove unused columns from the dataset
  push_to_hub=False,              # do not push the model to the hub
  report_to='tensorboard',        # report metrics to tensorboard
  load_best_model_at_end=True,    # load the best model at the end of training
)


# %%
from transformers import Trainer

trainer = Trainer(
    model=model,                        # the instantiated 🤗 Transformers model to be trained
    args=training_args,                 # training arguments, defined above
    data_collator=collate_fn,           # the data collator that will be used for batching
    compute_metrics=compute_metrics,    # the metrics function that will be used for evaluation
    train_dataset=dataset["train"],     # training dataset
    eval_dataset=dataset["validation"], # evaluation dataset
    tokenizer=image_processor,          # the processor that will be used for preprocessing the images
)

# %%
# start training
trainer.train()

# %%
# trainer.evaluate(dataset["test"])
trainer.evaluate()

# %%
# start tensorboard
# %load_ext tensorboard
%reload_ext tensorboard
%tensorboard --logdir ./vit-base-food/runs

# %% [markdown]
# ## Alternatively: Training using PyTorch Loop
# Run the two below cells to fine-tune using a regular PyTorch loop if you want.

# %%
# Training loop
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader

batch_size = 32

train_dataset_loader = DataLoader(dataset["train"], collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
valid_dataset_loader = DataLoader(dataset["validation"], collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

# define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

log_dir = "./image-classification/tensorboard"
summary_writer = SummaryWriter(log_dir=log_dir)

num_epochs = 3
model = model.to(device)
# print some statistics before training
# number of training steps
n_train_steps = num_epochs * len(train_dataset_loader)
# number of validation steps
n_valid_steps = len(valid_dataset_loader)
# current training step
current_step = 0
# logging, eval & save steps
save_steps = 1000

def compute_metrics(eval_pred):
  accuracy_score = accuracy.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids)
  f1_score = f1.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids, average="macro")
  return {**accuracy_score, **f1_score}

# %%
for epoch in range(num_epochs):
    # set the model to training mode
    model.train()
    # initialize the training loss
    train_loss = 0
    # initialize the progress bar
    progress_bar = tqdm(range(current_step, n_train_steps), "Training", dynamic_ncols=True, ncols=80)
    for batch in train_dataset_loader:
      if (current_step+1) % save_steps == 0:
        ### evaluation code ###
        # evaluate on the validation set
        # if the current step is a multiple of the save steps
        print()
        print(f"Validation at step {current_step}...")
        print()
        # set the model to evaluation mode
        model.eval()
        # initialize our lists that store the predictions and the labels
        predictions, labels = [], []
        # initialize the validation loss
        valid_loss = 0
        for batch in valid_dataset_loader:
            # get the batch
            pixel_values = batch["pixel_values"].to(device)
            label_ids = batch["labels"].to(device)
            # forward pass
            outputs = model(pixel_values=pixel_values, labels=label_ids)
            # get the loss
            loss = outputs.loss
            valid_loss += loss.item()
            # free the GPU memory
            logits = outputs.logits.detach().cpu()
            # add the predictions to the list
            predictions.extend(logits.argmax(dim=-1).tolist())
            # add the labels to the list
            labels.extend(label_ids.tolist())
        # make the EvalPrediction object that the compute_metrics function expects
        eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
        # compute the metrics
        metrics = compute_metrics(eval_prediction)
        # print the stats
        print()
        print(f"Epoch: {epoch}, Step: {current_step}, Train Loss: {train_loss / save_steps:.4f}, " + 
              f"Valid Loss: {valid_loss / n_valid_steps:.4f}, Accuracy: {metrics['accuracy']}, " +
              f"F1 Score: {metrics['f1']}")
        print()
        # log the metrics
        summary_writer.add_scalar("valid_loss", valid_loss / n_valid_steps, global_step=current_step)
        summary_writer.add_scalar("accuracy", metrics["accuracy"], global_step=current_step)
        summary_writer.add_scalar("f1", metrics["f1"], global_step=current_step)
        # save the model
        model.save_pretrained(f"./vit-base-food/checkpoint-{current_step}")
        image_processor.save_pretrained(f"./vit-base-food/checkpoint-{current_step}")
        # get the model back to train mode
        model.train()
        # reset the train and valid loss
        train_loss, valid_loss = 0, 0
      ### training code below ###
      # get the batch & convert to tensor
      pixel_values = batch["pixel_values"].to(device)
      labels = batch["labels"].to(device)
      # forward pass
      outputs = model(pixel_values=pixel_values, labels=labels)
      # get the loss
      loss = outputs.loss
      # backward pass
      loss.backward()
      # update the weights
      optimizer.step()
      # zero the gradients
      optimizer.zero_grad()
      # log the loss
      loss_v = loss.item()
      train_loss += loss_v
      # increment the step
      current_step += 1
      progress_bar.update(1)
      # log the training loss
      summary_writer.add_scalar("train_loss", loss_v, global_step=current_step)
        

# %% [markdown]
# # Performing Inference

# %%
# load the best model, change the checkpoint number to the best checkpoint
# if the last checkpoint is the best, then ignore this cell
best_checkpoint = 7000
# best_checkpoint = 150
model = ViTForImageClassification.from_pretrained(f"./vit-base-food/checkpoint-{best_checkpoint}").to(device)
# model = ViTForImageClassification.from_pretrained(f"./vit-base-skin-cancer/checkpoint-{best_checkpoint}").to(device)

# %%
get_prediction(model, "https://images.pexels.com/photos/858496/pexels-photo-858496.jpeg?auto=compress&cs=tinysrgb&w=600&lazy=load")

# %%
def get_prediction_probs(model, url_or_path, num_classes=3):
    # load the image
    img = load_image(url_or_path)
    # preprocessing the image
    pixel_values = image_processor(img, return_tensors="pt")["pixel_values"].to(device)
    # perform inference
    output = model(pixel_values)
    # get the top k classes and probabilities
    probs, indices = torch.topk(output.logits.softmax(dim=1), k=num_classes)
    # get the class labels
    id2label = model.config.id2label
    classes = [id2label[idx.item()] for idx in indices[0]]
    # convert the probabilities to a list
    probs = probs.squeeze().tolist()
    # create a dictionary with the class names and probabilities
    results = dict(zip(classes, probs))
    return results

# %%
# example 1
get_prediction_probs(model, "https://images.pexels.com/photos/406152/pexels-photo-406152.jpeg?auto=compress&cs=tinysrgb&w=600")

# %%
# example 2
get_prediction_probs(model, "https://images.pexels.com/photos/920220/pexels-photo-920220.jpeg?auto=compress&cs=tinysrgb&w=600")

# %%
# example 3
get_prediction_probs(model, "https://images.pexels.com/photos/3338681/pexels-photo-3338681.jpeg?auto=compress&cs=tinysrgb&w=600")

# %%
# example 4
get_prediction_probs(model, "https://images.pexels.com/photos/806457/pexels-photo-806457.jpeg?auto=compress&cs=tinysrgb&w=600", num_classes=10)

# %%
get_prediction_probs(model, "https://images.pexels.com/photos/1624487/pexels-photo-1624487.jpeg?auto=compress&cs=tinysrgb&w=600")


