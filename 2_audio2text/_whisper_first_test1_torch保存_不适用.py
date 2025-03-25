import whisper
import torch

# 首先创建一个空的模型实例
model = whisper.load_model("base").cuda()
model.eval()  # 设置为评估模式，如果模型用于推理的话
print(model)
print("---------------------------------------------------------------")

# 然后加载之前保存的状态字典
model_path = './_whisper_model/base.pt'
torch.save(model.state_dict(), model_path)
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置为评估模式，如果模型用于推理的话
print(model)
print("---------------------------------------------------------------")