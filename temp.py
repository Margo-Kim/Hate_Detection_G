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

def eval():
    batch_size = 8
    plm = 'beomi/KcELECTRA-base-v2022'
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")


    model = HateClassification.load_from_checkpoint('saved_models/Detection_model.ckpt', plm = plm, num_labels = 2)
    model.eval()

    test = openfile('test')
    a = test['id']
    b = test['input']
    test = { 'id' : a,
        'input' : b }
    print(test)

    inputs = tokenizer(test['input'],  truncation=True, padding=True, return_tensors="pt", max_length = 512)

    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        
    probs = torch.nn.functional.softmax(logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1)


    prediction = prediction.tolist()

    output_list = [{'id': id_, 'input': input_, 'output': prediction} for id_, input_, prediction in zip(test['id'], test['input'], prediction)]

    # Save output list to a jsonl file
    with open('predictions.jsonl', 'w', encoding='UTF-8') as f:
        for item in output_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            

