import torch
import base64
import requests
from PIL import Image
import io

from torchvision import transforms

from transformers import AutoProcessor, AutoModelForVision2Seq

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def encode_tensor_image(tensor):
    # print(tensor.shape)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  
    toPIL = transforms.ToPILImage() 
    image = toPIL(tensor)
    return image


def llama_observe(image_tensor, text_prompt):
  model_id = "llava-hf/llava-1.5-7b"
  processor = AutoProcessor.from_pretrained(model_id)
  model = AutoModelForVision2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
  ).eval()
  
  image = encode_tensor_image(image_tensor)
  inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
  with torch.no_grad():
     output = model.generate(**inputs, max_new_tokens=50)
  response = processor.batch_decode(output, skip_special_tokens=True)[0]
  return response
