from torch.utils.data import Dataset 
import torch 
import json 
from PIL import Image 


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
        return image, text_ids.input_ids, text_ids.attention_mask



