# -*- coding: utf-8 -*-
"""VQA-COCO-Mult-T5Transformer-EfficientNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c5ltulz4qb0Sq01S4MssFJCKHgKgwv9X
"""

!pip install --quiet transformers
!pip install --quiet pytorch-lightning
!pip install --quiet tokenizers
!pip install --quiet timm

import tensorflow as tf
import json
import os
import cv2
from google.colab.patches import cv2_imshow
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import timm
from torch import nn
import PIL
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import (AdamW,T5ForConditionalGeneration, AutoTokenizer as Tokenizer)

from google.colab import drive
drive.mount('/content/drive')

"""# **Pre-Process the Data**"""

_URL = 'http://images.cocodataset.org/zips/val2014.zip'
zip_dir = tf.keras.utils.get_file('/content/MSCOCOVAL2014.zip', origin=_URL, extract=False,archive_format='auto')
fname = '/content/MSCOCOVAL2014.zip'
!unzip -q $fname -d /content/

path = '/content/drive/MyDrive/VQA/'

with open(os.path.join(path, 'v2_OpenEnded_mscoco_val2014_questions.json'), 'r') as f:
    val_questions = json.load(f)['questions']
with open(os.path.join(path, 'v2_mscoco_val2014_annotations.json'), 'r') as f:
    val_answers = json.load(f)['annotations']

val_data = []
for question, annotation in zip(val_questions, val_answers):
    question_text = question['question']
    image_id = annotation['image_id']
    answer = annotation['answers'][0]['answer']
    image_filename = 'COCO_val2014_{:012d}.jpg'.format(image_id)
    image_path = os.path.join('/content/', 'val2014', image_filename)
    val_data.append({'question': question_text, 'image_path': image_path, 'answer': answer})

# Convert the array of dictionaries to a DataFrame
df = pd.DataFrame(val_data)

df

def show_sample(idx=0):
  print("Q : ",df.iloc[idx]['question'])
  image = cv2.imread(df.iloc[idx]['image_path'])
  image = cv2.resize(image, (224, 224))  
  cv2_imshow(image)
  print("A : ",df.iloc[idx]['answer'])

show_sample(20)



"""# **DataLoader**"""

def has_three_channels(image_path):
    with PIL.Image.open(image_path) as img:
        return img.mode == 'RGB'

# Filter the DataFrame to keep only the images with 3 channels
df = df[df['image_path'].apply(has_three_channels)]

max_length = len(df['answer'].max())
print(max_length)

max_length = len(df['question'].max())
print(max_length)

MODEL_NAME = 't5-base'

df = df[:5000]

tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

train_df, val_df = train_test_split(df,test_size=0.1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

class VQADataset(Dataset):
    def __init__(self, df, transform,tokenizer):
        self.df = df
        self.transform = transform
        self.tokenizer = tokenizer

        self.embed_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.linear = torch.nn.Linear(768, 64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        question = row['question']
        image_path = row['image_path']
        answer = row['answer']

        question = self.tokenizer(question, padding=True,truncation = True, return_tensors='pt')
        
        with torch.no_grad():
            embedding = self.embed_model.encoder(input_ids=question['input_ids']).last_hidden_state.mean(dim=1)
            queembed = self.linear(embedding)


        answer = self.tokenizer(answer, padding='max_length',truncation = True,return_attention_mask = True,add_special_tokens = True, max_length=64, return_tensors='pt')

        labels = answer["input_ids"]
        labels[labels == 0] = -100
        
        # load the image and apply transform
        img = PIL.Image.open(image_path)
        img = transform(img)

        return dict(
        question = queembed,
        image = img,
        labels = labels.flatten())

class VQADataModule(pl.LightningDataModule):
  def __init__(self,train_df : pd.DataFrame,test_df : pd.DataFrame,transform : transform,tokenizer : tokenizer,batch_size : int = 8):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.transform = transform


  def setup(self,stage=None):
    self.train_dataset = VQADataset(self.train_df,self.transform,self.tokenizer)
    self.test_dataset = VQADataset(self.test_df,self.transform,self.tokenizer)

  def train_dataloader(self):
    return DataLoader(self.train_dataset,batch_size = self.batch_size,shuffle=True,num_workers=4)

  def val_dataloader(self):
    return DataLoader(self.test_dataset,batch_size = self.batch_size,num_workers=4)

  def test_dataloader(self):
    return DataLoader(self.test_dataset,batch_size = self.batch_size,num_workers=4)

BATCH_SIZE = 8
N_EPOCHS = 10

data_module = VQADataModule(train_df,val_df,transform,tokenizer,batch_size = BATCH_SIZE)
data_module.setup()

"""# **Model Architecture**"""

class VQAModel(pl.LightningModule):
  def __init__(self):
    super().__init__()

    self.text_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME,return_dict=True)
    self.lossfn = nn.CrossEntropyLoss()
    self.tokenizer = tokenizer

    self.img_model = timm.create_model('efficientnet_b0', pretrained=True)
    self.img_model.aux_logits=False

    # Freeze training for all layers
    for param in self.img_model.parameters():
        param.requires_grad = False
    
    self.img_model.classifier = torch.nn.Sequential(
                      torch.nn.Linear(self.img_model.classifier.in_features, 256),
                      torch.nn.Dropout(0.5),
                      torch.nn.ReLU(inplace=True),
                      torch.nn.BatchNorm1d(256),
                      torch.nn.Linear(256, 64))
                  

  def forward(self,image,question,labels=None):
    img_output = self.img_model(image)
    img_output = img_output.squeeze()
    question = question.squeeze()
    # Convert the embeddings to strings
    question_str = [' '.join(map(str, q.tolist())) for q in question]
    image_str = [' '.join(map(str, i.tolist())) for i in img_output]

    # Concatenate the question and image strings using f-string for each batch element
    queimg = [f"question {q} image {i}" for q, i in zip(question_str, image_str)]
    
    queimg = self.tokenizer(queimg, padding='max_length',truncation = True,return_attention_mask = True,
                            add_special_tokens = True, max_length=512, return_tensors='pt')
    input_ids= queimg['input_ids'].to('cuda')
    attention_masks= queimg['attention_mask'].to('cuda')
    
    output = self.text_model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
    return output.loss, output.logits

  def training_step(self,batch,batch_idx):
    question = batch['question']
    image = batch['image']
    labels = batch['labels']
    loss, logits = self(image,question,labels)
    self.log("train_loss",loss,prog_bar=True,logger=True)
    return loss

  def validation_step(self,batch,batch_idx):
    question = batch['question']
    image = batch['image']
    labels = batch['labels']
    loss, logits = self(image,question,labels)
    self.log("val_loss",loss,prog_bar=True,logger=True)
    return loss


  def test_step(self,batch,batch_idx):
    question = batch['question']
    image = batch['image']
    labels = batch['labels']
    loss, logits = self(image,question,labels)
    self.log("test_loss",loss,prog_bar=True,logger=True)
    return loss


  def configure_optimizers(self):
    return AdamW(self.parameters(),lr = 0.0001)

"""# **Training**"""

model = VQAModel()

checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = 'best_cp',
    save_top_k = 1,
    verbose = True,
    monitor = 'val_loss',
    mode = 'min')

trainer = pl.Trainer(gpus=1,
    callbacks=[checkpoint_callback],
    max_epochs = N_EPOCHS)

trainer.fit(model,data_module)