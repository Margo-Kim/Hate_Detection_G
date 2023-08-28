from data import *
from dataloader import *
from model import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
import os
import json
import pandas as pd

# Global 로 선언해서 다른 py에서 변수로 쓸수있는지 급 궁금?
batch_size = 8 
plm = 'beomi/KcELECTRA-base-v2022'
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

model = HateClassification.load_from_checkpoint('saved_models/Detection_model.ckpt', plm = plm, num_labels = 2)
model.eval()

file = open('youtube_comment.csv', 'r')
# hmm..이걸 pandas DF 로 처리할지 아님 csv 파일로 처리할지..
# 시키는 주제를 한다 // 뭐라도 해라..교수님과 컨택된것은 아님..컨택이 되는 교수님의 주제로 가는걸로..진짜 가르쳐줄 사람..잘못하면 MBA 뭔가를 얻을려면 
# 석사를 쳐주는게 연구에 머리를 박는다 // 대학도보고 연구하는게 의미가 없을 수도..쓴 논문..질문 2개만 하면 티가 난다..^^..연구를 했느냐..^^..
# 취업 --> 제일 좋은거는 본인이 가는 그 트랙으로 간 한국인으로// seq2seq

data_dict = {'id': [],
             'input':[]
             }

for line in file:
    a = line.split(',')
    if len(a) >= 2:
        data_dict['id'].append(a[0])
        data_dict['input'].append(a[1])
    

inputs = tokenizer(data_dict['input'],truncation=True, padding=True, return_tensors="pt", max_length = 512)

with torch.no_grad():
    logits = model(inputs['input_ids'], inputs['attention_mask'])

probs = torch.nn.functional.softmax(logits, dim=-1)
prediction = torch.argmax(probs, dim=-1)

prediction = prediction.tolist()
inference = []
for each in prediction:
    each = str(each)
    if each == '0':
        each = '긍정'
    elif each == '1':
        each = '부정'
    
    inference.append(each)
    

for i in range(len(prediction)):
    output = [(sentence, infer) for sentence, infer in zip(data_dict['input'], inference)]

dataframe = pd.DataFrame(output, columns = ['input', 'prediction'])

save = dataframe.to_csv('youtube_prediction.csv', encoding = 'utf-8')