import torch
import base64
import requests
from PIL import Image
import io

from torchvision import transforms

from transformers import AutoProcessor, AutoModelForVision2Seq

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


  


def llava_observe(image_tensor, text_prompt):
  model_id = "llava-hf/llava-1.5-7b-hf"
  processor = AutoProcessor.from_pretrained(model_id)
  model = AutoModelForVision2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
  ).eval()
  
  image_tensor = image_tensor.squeeze(0)
  image_tensor = transforms.ToPILImage()(image_tensor)
  inputs = processor(images=image_tensor, text=text_prompt, return_tensors="pt").to(device)
  output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
  response = processor.batch_decode(output, skip_special_tokens=True)[0]
  return response
