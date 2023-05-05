import torch 
from PIL import Image 
import os 
from models import MossTokenizer 
from transformers import ChineseCLIPModel, ChineseCLIPProcessor 
import gradio as gr


os.environ['CUDA_VISIBLE_DEVICES'] = "1" 
model_path = './ckpt/visual_moss'
clip_model_path = './ckpt/cn_clip'

print('clip model load')
clip_model = ChineseCLIPModel.from_pretrained(clip_model_path)
process = ChineseCLIPProcessor.from_pretrained(clip_model_path)
clip_model = clip_model.cuda()
tokenizer = MossTokenizer.from_pretrained(model_path)
model = torch.load("./ckpt/visual_moss/ckpt.pt")
model = model.cuda()
image_features = None 
image_dig = False


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def add_text(history, text): 
    text = parse_text(text)
    history = history + [(text, None)]
    return history, ""


def add_file(history, file): 
    global image_features
    global image_dig 
    images = Image.open(file.name).convert("RGB") 
    images = process(images=images, return_tensors="pt")['pixel_values'].cuda() 
    image_features, vision_outputs = clip_model.get_image_features(pixel_values=images)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True) 
    image_dig = True 
    history = []
    history = history + [((file.name,), None)]
    return history


def bot(history): 
    global image_features
    global image_dig

    history_sentence = '' 
    for i in range(1, len(history)): 
        if image_dig == True: 
            if i == 0:
                history_sentence += ' <Image> '
                continue

        if i == len(history) - 1:
            history_sentence += ' <|Human|>: ' + history[i][0] + ' <eoh>' 
        else: 
            history_sentence += ' <|Human|>: ' + history[i][0] + ' <eoh>' 
            history_sentence +=  history[i][1] 
    
    history_sentence += '<|MOSS|>: '
    print(history_sentence)    
    inputs = tokenizer(history_sentence, return_tensors="pt")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask.cuda(),
            image_features=image_features,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=256,
        )
    response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response = parse_text(response)
    history[-1][1] = response
    print(history)
    return history

title = """<h1 align="center">基于MiniGPT-4多模态对话模型</h1>"""
description = """<h3>汇总和评估开源和自己训练的多模态语言模型。问题反馈: feizhengcong</h3>"""


with gr.Blocks() as demo: 
    gr.Markdown(title)
    gr.Markdown(description)
    
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    btn.upload(add_file, [chatbot, btn], [chatbot])
    # .then(
    #    bot, chatbot, chatbot
    #)

demo.launch(enable_queue=True, server_name='10.164.111.73',server_port=8418)

