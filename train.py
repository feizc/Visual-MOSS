import torch 
import os 
from models import VisualMossModel, MossConfig, MossTokenizer
from typing import Dict
from tqdm import tqdm 
import transformers 
import torch.utils.tensorboard as tensorboard 
from transformers import ChineseCLIPProcessor

from utils import FastImageTextDataset, UrlTextDataset 


DEFAULT_IMAGE_TOKEN = "<Image>" 
DEFAULT_PAD_TOKEN = "[PAD]"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.transformer.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def main(): 
    # hyper-parameter 
    moss_path = './ckpt/moss' 
    clip_model_path = './ckpt/cn_clip' 
    save_path = './ckpt/visual_moss'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    max_length = 256
    learning_rate = 5e-6
    data_path = 'train.pkl'
    train_batch_size = 1
    gradient_accumulation_steps = 8
    epochs = 5
    tensorboard_path = 'log'
    url_data = False

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    writer = tensorboard.SummaryWriter(tensorboard_path)

    process = ChineseCLIPProcessor.from_pretrained(clip_model_path) 
    config = MossConfig.from_pretrained(moss_path) 
    config.mm_vision_tower = clip_model_path
    config.mm_embd = 1024
    
    print('load visual moss')
    model = VisualMossModel(config) 
    model.load_state_dict(torch.load('out.pt'))
    model = model.to(device) 

    """
    model.transformer.requires_grad_(False) 
    model.lm_head.requires_grad_(False) 
    model.vision_tower.requires_grad_(False) 
    """
    model.requires_grad_(False) 
    for p in model.mm_projector.parameters(): 
        p.requires_grad = True 

    tokenizer = MossTokenizer.from_pretrained(
        moss_path,
        model_max_length=max_length, 
        padding_side="right",
        use_fast=False,
    )
    print('vocab', len(tokenizer))

    if tokenizer.pad_token is None: 
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        ) 
    print('vocab+pad', len(tokenizer))
    num_new_tokens = tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
    model.transformer.resize_token_embeddings(len(tokenizer)) 
    image_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN])
    print(tokenizer.pad_token, image_token_id[0], len(tokenizer))

    if num_new_tokens > 0: 
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    optimizer_class = torch.optim.AdamW 
    optimizer = optimizer_class(
        model.mm_projector.parameters(),
        lr=learning_rate,
    )

    # Dataset and DataLoaders creation 
    print('dataset prepare')
    if url_data == True:
        train_dataset = UrlTextDataset(data_path=data_path, tokenizer=tokenizer, process=process)
    else:
        train_dataset = FastImageTextDataset(data_path=data_path, tokenizer=tokenizer, process=process) 
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=True,
    )
    
    model.transformer.eval() 
    model.lm_head.eval() 
    model.mm_projector.train()
    for epoch in range(epochs): 
        loss_cum = 0
        iteration = 0
        num_batches_per_epoch = len(train_dataloader)
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader)) as t: 
            for step, batch in t: 
                log_step = num_batches_per_epoch * epoch + step
                iteration += 1 
                image, input_ids, attention_mask = batch 
                image, input_ids, attention_mask = image.to(device), input_ids.to(device), attention_mask.to(device) 
                #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, image_features=image, labels=input_ids) 
                loss = outputs.loss 

                loss = loss / gradient_accumulation_steps 
                loss.backward() 

                if iteration % gradient_accumulation_steps == 0: 
                    torch.nn.utils.clip_grad_norm_(model.mm_projector.parameters(), 1.0) 
                    optimizer.step()
                    optimizer.zero_grad() 
                loss_cum += loss.item() 
                t.set_description('Epoch %i' % epoch)
                t.set_postfix(loss=loss_cum / (step + 1))
                
                if step % 10 == 0:
                    writer.add_scalar("train/loss", loss.item(), log_step)

                if iteration % 20000 == 0:
                    torch.cuda.empty_cache() 
                    print('save model') 
                    torch.save(model, os.path.join(save_path, 'ckpt.pt')) 
                    tokenizer.save_pretrained(save_path)
                    config.save_pretrained(save_path)
        

if __name__ == "__main__":
    main()
