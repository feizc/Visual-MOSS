from transformers import ChineseCLIPVisionModel, ChineseCLIPProcessor, ChineseCLIPModel 
from PIL import Image 
import json 
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
from urllib.parse import urlparse
import torch 


def test_chinese_clip():
    model_path = './ckpt/cn_clip'
    model = ChineseCLIPModel.from_pretrained(model_path)
    process = ChineseCLIPProcessor.from_pretrained(model_path) 

    image = Image.open('cat.png') 
    inputs = process(images=image, return_tensors="pt") 
    print(inputs)

    image_features, vision_outputs = model.get_image_features(pixel_values=inputs['pixel_values'])
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True) 
    print(image_features.size())


def toy_data_create():
    data_list = [] 
    for i in range(10): 
        item = {
            'image': 'cat.png',
            'text': '<Image> <|Human|>: 你能描述这张图片吗？<eoh>\n <|MOSS|>: 一只小猫在草地上 <eom>\n',
        }
        data_list.append(item)

    print(len(data_list))
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4)


def coyo700M_data_selection(): 
    data_path = 'coyo.txt' 
    target_path = 'data.json'
    data_list = [] 
    item_num = 500000
    count = 0 

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('\t') 
            item = {
                'url': line[0],
                'text': line[1].strip(),
            }
            data_list.append(item) 
            count += 1
            if count > item_num:
                break 

    print(data_list[-1]) 
    with open(target_path, 'w', encoding='utf-8') as f: 
        json.dump(data_list, f)



def fast_coyo700M_data_selection():
    import pickle
    from tqdm import tqdm 
    data_path = 'coyo.txt' 
    target_path = 'train_50.pkl'
    item_num = 500000
    count = 0 
    model_path = './ckpt/cn_clip'
    model = ChineseCLIPModel.from_pretrained(model_path)
    process = ChineseCLIPProcessor.from_pretrained(model_path) 
    model.cuda()

    pkl_f = open(target_path, 'wb')
    repet = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if repet < 70000:
                repet += 1
                continue

            line = line.split('\t') 
            url = line[0]
            text = line[1].strip()
            try:
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
            
                inputs = process(images=image, return_tensors="pt") 
            except:
                continue
            image_features, vision_outputs = model.get_image_features(pixel_values=inputs['pixel_values'].cuda())
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            image_features = image_features.cpu()

            item = {
                'image': image_features,
                'text': text
            }

            pickle.dump(item, pkl_f)
            
            count += 1 
            if count % 10 == 0:
                print(count) 

            if count > item_num:
                break


def pkl_combine(path_list): 
    import pickle 
    target_path = 'train_target.pkl'
    com_pkl_f = open(target_path, 'wb')
    num = 0 

    for path in path_list: 
        file = open(path, 'rb') 
        try: 
            item = pickle.load(file) 
            num += 1
        except:
            continue 
        pickle.dump(item, com_pkl_f) 
    
    print(num)
