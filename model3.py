from pprint import pprint
import pytorch_lightning as pl
from typing import Optional
import torch.nn as nn
import torch
from torch.nn import functional as F
from collections import defaultdict
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import BinaryAccuracy, Accuracy

from transformers import (
    PreTrainedModel,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertModel,
    RobertaModel,
    BertConfig,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)

#  여기서 BertConfig 더 찾아보기
# Max pooling 에서 kernal size? batch Normalization

class HateClassification_3 (pl.LightningModule):
    
    def __init__(self, 
                 model_name_or_path: str,
                 num_labels,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        
        self.conv1 = torch.nn.Conv1d(768, 512, kernel_size = 1, stride=1)
        self.conv2 = torch.nn.Conv1d(512, 512, kernel_size=2)
        self.conv3 = torch.nn.Conv1d(512, 512, kernel_size=3)
        self.conv4 = torch.nn.Conv1d(512, 512, kernel_size=4)
        
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size = 2)
        self.avgpool = torch.nn.AvgPool1d(kernel_size = 2)
        
        self.fc1 = nn.Linear(1024*474, 256)
        self.fc2 = nn.Linear(256, num_labels)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(num_labels)
        
        self.loss_fct = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask) :
        bert_output =  self.bert (input_ids = input_ids, attention_mask = attention_mask
                            ) #(8,512,768)
        bert_output = bert_output[0].permute(0, 2, 1) #(8,768,512)
        # CNN (bsz, num_channel, length)
        conv1_avg = self.avgpool(self.relu(self.conv1(bert_output))) #conv1 (8,512,512) --> (8, 512, 256)
        conv1_max = self.maxpool(self.relu(self.conv1(bert_output))) #(8,512,512) --> (8, 512, 256)
        
        conv2_max = self.maxpool(self.relu(self.conv2(conv1_max))) #conv2 (8,512,255) --> max (8,512,127)
        conv2_avg = self.avgpool(self.relu(self.conv2(conv1_max))) 
        
        conv3_max = self.maxpool(self.relu(self.conv3(conv2_max)))#conv3 (8,512,125) --> max (8,512,62)
        conv3_avg = self.avgpool(self.relu(self.conv3(conv2_max)))
        
        conv4_max = self.maxpool(self.relu(self.conv4(conv3_max)))#conv4 (8,512,59) --> max (8,512,29)
        conv4_avg = self.avgpool(self.relu(self.conv4(conv3_max)))
        
        out_max = torch.cat((conv1_max, conv2_max, conv3_max, conv4_max), dim = 2) #(8,512,474)
        #Question : 여기서 dim2가 맞는지 궁금 
        out_avg = torch.cat((conv1_avg, conv2_avg, conv3_avg, conv4_avg), dim=2) #(8,512,474)
        
        # Concat Max + AVG pooling
        total_out = torch.cat((out_max, out_avg), dim = 1) #(8, 1024, 474)
        # 음 여기서 dim을 1로 줘야할지..2로 줘야할지..쉐입이 똑같아서 사실 뭔 차인지 궁금
        total_out = total_out.view(total_out.shape[0], -1)
        #(8, 1024*474)
        # FC layer 태우기
        out_1 = self.dropout(self.batchnorm(self.fc1(total_out))) #(8,256)
        out_2 = self.dropout(self.batchnorm2(self.fc2(out_1)))
        
        return out_2
        
        
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['label']
        
        logits = self(input_ids, attention_mask)
        
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        acc = self.accuracy(logits, batch['label'])
        
        self.log('Train/Loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('Train/Accuracy', acc, on_step=True, on_epoch=True, logger=True)
        
        print(f"Training Accuracy: {acc}")
        return loss
        
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=2e-5)
        
    def accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).float()
        acc = correct.sum() / len(correct)
        return acc
        

        
        
        
        
        # self.cnn1 = torch.nn.Sequential(
        #     torch.nn.Conv1d(768, 512, kernel_size = 1, stride=1),
        #     torch.nn.ReLu(),
        #     torch.nn.MaxPool1d(kernel_size = 2),
        #     torch.nn.AvgPool1d(kernel_size = 2))
        
        # self.cnn2 = torch.nn.Sequential(
        #     torch.nn.Conv1d(256, 512, kernel_size=2), #여기서 in channel 확인해야함
        #     torch.nn.Relu(),
        #     torch.nn.MaxPool1d(kernel_size= 2),
        #     torch.nn.AvgPool1d(kernel_size = 2)
            
        # )
        
        # self.cnn3 = torch.nn.Sequential(
        # torch.nn.Conv1d(128, 512, kernel_size=2), #여기서 in channel 확인해야함
        # torch.nn.Relu(),
        # torch.nn.MaxPool1d(kernel_size= 2),
        # torch.nn.AvgPool1d(kernel_size = 2)
        # )
        
        