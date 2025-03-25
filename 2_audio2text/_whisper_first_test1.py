import whisper
import torch
device = torch.device('cuda:0')

# 首先创建一个空的模型实例
model = whisper.load_model(name="base",
                           device=device,
                           download_root="./_whisper_model",
                           ).cuda()
model.eval()  # 设置为评估模式，如果模型用于推理的话
print(model)
print("---------------------------------------------------------------")
