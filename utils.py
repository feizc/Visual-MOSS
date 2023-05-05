from torch.utils.data import Dataset 
import torch 
import json 
from PIL import Image 

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
from urllib.parse import urlparse
import random 


prompts = [
    '详细描述图片内容',
    '看一眼图片然后描述你的所见',
    '请提供对于图片的详细描述',
    '可以描述图片内容的吗？'
]


class ImageTextDataset(Dataset): 
    def __init__(self, data_path, tokenizer, process):
        self.data_list = json.load(open(data_path, 'r')) 
        self.tokenizer = tokenizer 
        self.process = process 

    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, index):
        data_pair = self.data_list[index] 
        image_path = data_pair['image'] 
        image = Image.open(image_path)
        image = self.process(images=image, return_tensors="pt")['pixel_values'] 

        text = data_pair['text'] 
        text_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        
        return image.squeeze(0), text_ids.input_ids.squeeze(0), text_ids.attention_mask.squeeze(0)



class UrlTextDataset(Dataset): 
    def __init__(self, data_path, tokenizer, process):
        self.data_list = json.load(open(data_path, 'r')) 
        self.tokenizer = tokenizer 
        self.process = process 

    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, index):
        data_pair = self.data_list[index] 
        url = data_pair['url'] 

        if 'http' in url:
            domain = urlparse(url).netloc
            url = url.replace(domain, 'p.vip.sankuai.com')+'@384w'
            session = requests.Session()
            retry = Retry(connect=3, read=3, redirect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            response = session.get(url)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(str(url)) 

        image = self.process(images=image, return_tensors="pt")['pixel_values'] 

        text = data_pair['text'] 
        prompt_index = random.randint(0, len(prompts) - 1)
        text_ids = self.tokenizer(
            '<Image> <|Human|>: ' + prompts[prompt_index] + ' <eoh>\n <|MOSS|>: ' + text + ' <eom>\n',
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        
        return image.squeeze(0), text_ids.input_ids.squeeze(0), text_ids.attention_mask.squeeze(0)


import pickle 
class FastImageTextDataset(Dataset): 
    def __init__(self, data_path, tokenizer, process):
        self.file = open(data_path, 'rb') 
        self.data_path = data_path 
        self.tokenizer = tokenizer 
        self.process = process 

    def __len__(self):
        return 1000
    
    def __getitem__(self, index): 
        try: 
            item = pickle.load(self.file) 
        except:
            self.file.close() 
            self.file = open(self.data_path, 'rb') 
            item = pickle.load(self.file) 

        image = item['image']

        text = item['text'] 
        prompt_index = random.randint(0, len(prompts) - 1)
        text_ids = self.tokenizer(
            '<Image> <|Human|>: ' + prompts[prompt_index] + ' <eoh>\n <|MOSS|>: ' + text + ' <eom>\n',
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        
        return image.squeeze(0), text_ids.input_ids.squeeze(0), text_ids.attention_mask.squeeze(0)
        