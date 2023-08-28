import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch



def openfile (fileName):
    filepath = f"/Users/margokim/Documents/pytorch/Hate Detection/nikluge-au-2022-{fileName}.jsonl"
    file = open(filepath, 'r', encoding='UTF-8' )
    
    data_dict = { 'id' : [],
            'input' : [],
            'output' : []}
    
    for line in file:
        data = json.loads(line)
        data_dict['id'].append(data['id'])
        data_dict['input'].append(data['input'])
        if 'output' in data:
            data_dict['output'].append(data['output'])
        else:
            None
        
    return data_dict
    


# filetype = 'train'
# filePath = f"/Users/margokim/Documents/pytorch/Hate Detection/nikluge-au-2022-{filetype}.jsonl"

# file = open("/Users/margokim/Documents/pytorch/Hate Detection/nikluge-au-2022-train.jsonl", 'r', encoding='UTF-8')

# train = {'input' : [],
#         'output' : []}


# for line in file:
    
#     data = json.loads(line)
    
#     train['input'].append(data['input'])
#     train['output'].append(data['output'])