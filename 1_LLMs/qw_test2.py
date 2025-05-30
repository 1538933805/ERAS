import os
from openai import OpenAI

client = OpenAI(
    # # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # api_key=os.getenv("DASHSCOPE_API_KEY"), 
    api_key="sk-25269b2b171c47fba1392a22cb061a36", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-plus", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant, and an extraordinary scientist/expert in robot learning.'},
        {'role': 'user', 'content': '分解行星齿轮系的装配步骤，只给出调用api的序列（不进行其他补充说明或无关的表达）'}],
    )
    
print(completion.model_dump_json())

