import pytorch_lightning as pl
from transformers import AdamW, AutoModelForSequenceClassification
import torch.nn as nn
import torchmetrics 
import torch.optim as optim
import torchmetrics 
import logging
import torch


class HateClassification (pl.LightningModule):
    
    def __init__(self, plm, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(plm, num_labels=num_labels)
        

        
    def forward(self, input_ids, attention_mask):
        output = self.model (input_ids = input_ids, attention_mask = attention_mask
                            )
       
        return output.logits
        
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['label']
        
        logits = self(input_ids, attention_mask)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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