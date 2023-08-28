from data import *
from dataloader import *
from model import *
from model2 import *
from model3 import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
import os
import random

batch_size = 8
plm = 'beomi/KcELECTRA-base-v2022'
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")


train = openfile('train')
a = train['input']
b = train['output']
train = {'input' : a,
            'output' : b}

test = openfile('test')

train_dataset = HateDataset(train['input'], train['output'], tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

# model = HateClassification_2 (plm, 2)
model = HateClassification_3 (plm,2 )

trainer = pl.Trainer(max_epochs=1 , logger=TensorBoardLogger(save_dir='logs/', name='hate_model'))
# 학습을 어떻게 할지 관장하는 것, callback --> pl.Trainer
# 얼리 스탑핑을 넣는다, 10을 넣고, 에폭 돌때마다 validation 
# last hidden state 는 token outputs[0] / outputs[1] --> CLS (pooler output)

trainer.fit(model, train_dataloader)

os.makedirs('saved_models', exist_ok=True)
trainer.save_checkpoint('saved_models/Detection_model2.ckpt')

# model = HateClassification.load_from_checkpoint('saved_models/Detection_model.ckpt', plm = plm, num_labels = 2)
# model.eval()

# dev = openfile('dev')
# a = dev['input']
# b = dev['output']
# dev = {'input' : a }
# print(dev)

# inputs = tokenizer(dev['input'],  truncation=True, padding=True, return_tensors="pt", max_length = 512)

# with torch.no_grad():
#     logits = model(inputs['input_ids'], inputs['attention_mask'])

# probs = torch.nn.functional.softmax(logits, dim=-1)
# prediction = torch.argmax(probs, dim=-1)

# counter_t = 0
# counter_f = 0


# prediction = prediction.tolist()

# for i in range(len(b)):
#     if b[i] == prediction[i]:
#         flag = f"same {prediction[i]} = {b[i]}"
#         counter_t += 1
#         print(flag)
#     else:
#         flag = f"not same"
#         counter_f += 1 
#         print(flag)

# print(counter_t)
# print(counter_f)
