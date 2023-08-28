import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
from data import *



class HateDataset(Dataset):
    def __init__ (self, inputs, outputs, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = inputs
        self.outputs = outputs
        self.encodings = self.tokenizer(self.inputs, max_length = 512, 
                      truncation = True, padding = 'max_length', return_token_type_ids = True)
    
    def __getitem__ (self, idx):
        
        return {'input_ids' :  torch.tensor(self.encodings['input_ids'][idx], dtype = torch.long),
                'attention_mask' : torch.tensor(self.encodings['attention_mask'][idx], dtype = torch.long),
                'token_type_ids' : torch.tensor(self.encodings['token_type_ids'][idx], dtype = torch.long),
                'label' : torch.tensor (self.outputs[idx], dtype = torch.long)
                }
        
    def __len__ (self):
        
        return len(self.encodings['input_ids'])
    








    



