import whisper
import torch
device = torch.device('cuda:0')

# 首先创建一个空的模型实例
model = whisper.load_model(name="small",
                           device=device,
                           download_root="./_whisper_model",
                           ).cuda()
model.eval()  # 设置为评估模式，如果模型用于推理的话
print(model)
print("---------------------------------------------------------------")


result = model.transcribe("./2_audio2text/audios/test1.mp3", language="en")
print(result["text"])

result = model.transcribe("./2_audio2text/audios/test2.m4a", language="en")
print(result["text"])

# result = model.transcribe("./2_audio2text/audios/test1.mp3", language="zh")
# print(result["text"])