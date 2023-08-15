import torch
from torch.utils.data import Dataset
import pandas as pd 
from tqdm import tqdm

from utils import get_tokenizer


class GPT2Dataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.tokenizer = get_tokenizer()
        self.dataset = []
        
        df = pd.read_csv(data_dir)
        df = df[df['label'] == 1]
        df = df[df['score'] > 30]
        df.reset_index(drop=True, inplace=True)
        df.drop(labels=[
            'label', 'author', 'subreddit', 'ups', 'downs', 'date', 'created_utc'
        ], axis=1, inplace=True)
        df['len_comment'] = [len(str(x)) for x in df['comment']]
        df['len_parent'] = [len(str(x)) for x in df['parent_comment']]
        df = df[(df['len_parent'] < 1000) & (df['len_parent'] < 1000)]
        # df = df[df['len_comment'] < 1000]
        
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                reply_raw = row['comment']
                parent_raw = row['parent_comment']
                
                parent = self.tokenizer.encode(parent_raw) 
                reply = self.tokenizer.encode(reply_raw)
                
                if len(parent) > 0 and len(reply) > 0 and len(parent) + len(reply) <= 1022:
                    self.dataset.append({
                        'parent': parent,
                        'reply': reply
                    })
                    
            except BaseException:
                continue
            
        self.length = len(self.dataset)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.dataset[index]
        
        context = self.tokenizer.encode(self.tokenizer.pad_token) * 1024
        text = data['parent'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['reply'] + self.tokenizer.encode(self.tokenizer.eos_token)
        
        context[:len(text)] = text 
        
        context = torch.tensor(context)
        
        return {'context': context, 'loc_sep': len(data['parent'])}
        
        
class CustomDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.tokenizer = get_tokenizer()
        self.dataset = []
        
        df = pd.read_csv(data_dir)
        df.reset_index(drop=True, inplace=True)
        df.drop_duplicates(inplace=True)
        
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                reply_raw = row['Answer']
                parent_raw = row['Question']
                
                parent = self.tokenizer.encode(parent_raw)
                reply = self.tokenizer.encode(reply_raw)
                
                if len(parent) > 0 and len(reply) > 0 and len(parent)+len(reply) <= 1022:
                    self.dataset.append({
                        'parent': parent,
                        'reply': reply 
                    })
            
            except BaseException:
                continue
            
        self.length = len(self.dataset)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.dataset[index]
        
        context = self.tokenizer.encode(self.tokenizer.pad_token) * 1024
        text = data['parent'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['reply'] + self.tokenizer.encode(self.tokenizer.eos_token)
        
        context[:len(text)] = text 
        context = torch.tensor(context)
        
        return {'context': context, 'loc_sep': len(data['parent'])}
    