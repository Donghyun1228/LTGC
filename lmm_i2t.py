import torch
from torchvision import transforms
from lt_dataloaders import ImageNetLTDataLoader
from data_txt.imagenet_label_mapping import get_readable_name
# from gpt4v import gpt4v_observe
from transformers import AutoProcessor, AutoModelForVision2Seq, MllamaForConditionalGeneration
from ultis import sample_counter
import os
import json
import csv
import argparse


# hyper-param
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-d', '--data_dir', default='Imagenet', type=str,
                    help='Directory for the dataset')
parser.add_argument('-m', '--max_num', default=100, type=int,
                    help='Maximum number of images')
parser.add_argument('-f', '--class_number_file', default='data_txt/ImageNet_LT/imagenetlt_class_count.txt', type=str,
                    help='File path to class number file')
parser.add_argument('-exi', '--existing_description_path', default='descriptions_data/existing_description_list.csv', type=str,
                    help='File path to class number file')
args = parser.parse_args()



imagenet_loader = ImageNetLTDataLoader(data_dir=args.data_dir, 
                                       batch_size=1, 
                                       shuffle=False, 
                                       num_workers=4, 
                                       training=True)
#%%

# llava model
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# model_id = "llava-hf/llava-v1.6-34b-hf"
# model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = MllamaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map=device
).eval()

#%%
if not os.path.exists(args.class_number_file):
    sample_counter(imagenet_loader)
    with open(args.class_number_file, 'r') as file:
        dict_class_number = json.load(file)
else:
    with open(args.class_number_file, 'r') as file:
        dict_class_number = json.load(file)
data_to_write = []

for epoch, pack in enumerate(imagenet_loader):
    data, target, index = pack

    if dict_class_number[str(int(target))] < args.max_num:
        real_name = get_readable_name(int(target)).split(", ")[0]
        # text_prompt="Template: A photo of the class "+real_name+", {with distinctive features}{in specific scenes}. Please use the Template to briefly describe the image of the class " + real_name + '.'
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "It is a photo of the class "+real_name+". Describe the image and distinctive features of the "+real_name+" in the image in detail."}
            ]}
        ]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        data = data * std + mean

        # img_description = gpt4v_observe(data, text_prompt)['choices'][0]['message']['content']
        # img_description = gpt4v_observe(data, text_prompt)
        # get response from llava
        image_tensor = data.squeeze(0)
        image_tensor = transforms.ToPILImage()(image_tensor)
        inputs = processor(images=image_tensor, text=text_prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=300)
        img_description = processor.batch_decode(output, skip_special_tokens=True)[0].split("assistant\n")[-1]
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Here's a description of a photo of the class "+real_name+".\n" + img_description},
                {"type": "text", "text": "<Template>A photo of the class "+real_name+", {with distinctive features}{in specific scenes}</Template>. Please complete the Template using the description of the image to briefly summarize the image of the class " + real_name + '. Answer in a single sentence.'}
            ]}
        ]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text_prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=100)
        img_description = processor.batch_decode(output, skip_special_tokens=True)[0].split("assistant\n\n")[-1]
        print("***", img_description)
        
        if img_description[0] == 'A':
            data_to_write.append((int(target), img_description))

            if epoch % 5 == 0:
                with open(args.existing_description_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data_to_write)
                data_to_write = []
# %%
