#%%
# from openai import OpenAI
import pandas as pd
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from data_txt.imagenet_label_mapping import get_readable_name
import csv
import argparse
# %%
# hyper-param
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-exi', '--existing_description_path', default='descriptions_data/existing_description_list.csv', type=str,
                    help='File path to the existing description file')
parser.add_argument('-m', '--max_generate_num', default=200, type=int,
                    help='Maximum number of generated images')
parser.add_argument('-ext', '--extended_description_path', default='descriptions_data/extended_description.csv', type=str,
                    help='File path to the extended description file')
args = parser.parse_args()

# api_key = "Replace with your own OPENAI KEY."

# client = OpenAI(api_key=api_key)
# %%
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

df = pd.read_csv(args.existing_description_path, header=None, names=['label', 'text'])
grouped_texts = df.groupby('label')['text'].apply(lambda x: '\n'.join(x)).to_dict()
grouped_list = df.groupby('label')['text'].apply(list).to_dict()

for label, text in grouped_texts.items():
    current_all_description = grouped_list[label]
    # print(f"Label {label}:\n{text}\n")

    while len(current_all_description) < args.max_generate_num:
      real_name = get_readable_name(int(label)).split(", ")[0]

      system_content = "You will follow the Template to describe the object. Template: A photo of the class " + real_name + " {with distinctive features}{in specific scenes}. "
      current_description = text

      # print(current_description)

      ### self-reflection
      user_content = "Besides these descriptions mentioned above, please use the same Template to list other possible {distinctive features} and {specific scenes} for the class " + real_name

      # completion = client.chat.completions.create(
      #   # model="gpt-3.5-turbo",
      #   model="gpt-4-turbo-preview",
      #   messages=[
      #     {"role": "system", "content": system_content},
      #     {"role": "user", "content": current_description},
      #     {"role": "user", "content": user_content}
      #   ]
      # )

      # output = completion.choices[0].message.content
      # sentences = output.split(". ")

      messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": current_description},
        {"role": "user", "content": user_content}
      ]
      input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
      inputs = processor(
          input_text,
          add_special_tokens=False,
          return_tensors="pt"
      ).to(model.device)

      output = model.generate(**inputs, max_new_tokens=0)
      output = processor.decode(output[0])

      if '\n\n- ' in output:
        sentences = output.split("\n\n")
      elif '\n\n' in output:
        sentences = output.split("\n\n")
      elif '\n- ' in output:
        sentences = output.split("\n- ")
      elif '\n ' in output:
        sentences = output.split("\n")

      current_all_description.extend(sentences)

      with open(args.extended_description_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        for s in sentences:
            writer.writerow([label, s])

      # print(completion.choices[0].message.content)

# %%
