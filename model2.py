import pytorch_lightning as pl
from transformers import AdamW, AutoModelForSequenceClassification, AutoModel
import torch.nn as nn
import torchmetrics 
import torch.optim as optim
import torchmetrics 
import logging
import torch


class HateClassification_2 (pl.LightningModule):
    
    def __init__(self, plm, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(plm, output_hidden_states = True)
        # self.conv1d = nn.Conv1d(in_channels= self.bert.config.hidden_size * self.bert.config.num_hidden_layers, out_channels=128, kernel_size=3) #out_filter = 128
        # self.bert.config.num_hidden_layers 가 12 로 나오는데..아..!! 밑에꺼는 최종 hidden_state 까지 다 더하는거구나 그래서 13 layers
        self.conv1d = nn.Conv1d(in_channels= 9984, out_channels=128, kernel_size=3) #out_filter = 128
        self.fc = nn.Linear(170, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        

        
    def forward(self, input_ids, attention_mask):
        output = self.bert (input_ids = input_ids, attention_mask = attention_mask, output_hidden_states=True
                            )
        # 호호..튜플은 한번 선언된 값은 변경이 안된느데..
        hidden_states = output.hidden_states # [n_layer, bsz, token_len, dim]
        #모든 인코더의 hidden state output // 
        # 아니 output[2]는 왜 안되는거임...??하여튼 뜯어보면 encoder layer = 13, 1 layer hidden (8, 512,768) --> 13 개
        cat_hidden = torch.cat(hidden_states, dim = -1)   # [bsz, token_len, n_layer*dim]
        
        # 넓게 펴주기 size = [8,512, 9984], dim = -1 없이는 (104,512,768) 104는 bsz*13layers 왜냐면 dim=0 이 default
        # dim=-1 은 마지막 차원을 기준으로 더해주는거! hence 768 * 13
        # 정말 architecture 에 충실할려면 CLS 포함한 최종 layer를 사용 X 
        
        transposed_output = cat_hidden.permute(0,2,1)  # [bsz, n_layer*dim(9984), token_len(512)]
        #CNN은 (bsz, num_channel, length) --> (8, 9984, 512)
        #permute 는 들어온 [ , , ] 의 위치를 다 지정해서 한번에 바꾸는거 / transpose는 두 개의 차원을 맞교환하는것
        
        conv_output = self.conv1d(transposed_output)   # [bsz, 128, 510]
        pooled_output = torch.nn.functional.max_pool1d(conv_output, kernel_size=3).squeeze(2)
        #conv_output [8, 128, 510] , kernal_size = 510? 3? / pooled_output = [8, 128]
        logits = self.fc(pooled_output)
        # 128 * 170 170 *2 -> 128*2
       
        return logits
        
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