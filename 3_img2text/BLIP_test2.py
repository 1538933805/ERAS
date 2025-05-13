import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# 使用local_files_only=True可以强制离线，避免连内网时代码不运行
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",
                                                     cache_dir="./_huggingface_model/blip",
                                                     local_files_only=True)
processor = BlipProcessor.from_pretrained(pretrained_model_name_or_path="Salesforce/blip-image-captioning-large",
                                          cache_dir="./_huggingface_model/blip",
                                          local_files_only=True)

image_folder = './3_img2text/imgs' # 修改为你的图片所在文件夹路径
# image_filename = 'img1.jpg'  # 修改为你想要处理的图片文件名
image_filename = 'img0.jpg'  # 修改为你想要处理的图片文件名
image_path = os.path.join(image_folder, image_filename)
# 打开本地图片文件并转换为 RGB 模式
raw_image = Image.open(image_path).convert('RGB')

# conditional image captioning
text = "an image of"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
