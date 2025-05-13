import os
from openai import OpenAI
from datetime import datetime
import json
import sys
import re
import _main_command_ctrl
"""补充blip模块--开始"""
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
# 使用local_files_only=True可以强制离线，避免连内网时代码不运行
# model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",
#                                                      cache_dir="./_huggingface_model/blip",
#                                                      local_files_only=True)
# processor_blip = BlipProcessor.from_pretrained(pretrained_model_name_or_path="Salesforce/blip-image-captioning-large",
#                                           cache_dir="./_huggingface_model/blip",
#                                           local_files_only=True)
# raw_image = Image.open('./3_img2text/imgs/img0.jpg').convert('RGB')
# # unconditional image captioning
# inputs = processor_blip(raw_image, return_tensors="pt")
# out = model_blip.generate(**inputs)
# scene_description = "Here is the description of the scene: \n" + processor_blip.decode(out[0], skip_special_tokens=True)
# print(scene_description)
scene_description = "Here is the description of the scene: \n" + "there is a metal object that is sitting on a table"
"""补充blip模块--结束"""
conversation_file_folder = "./1_LLMs/conversation_history_1"
skill_listName_path = "./AssembleSkillList1.txt"
api_key_path = "./1_LLMs/api_key.txt"
api_key = open(api_key_path).read().strip()
MainCommandCtrl = _main_command_ctrl.MainCommandCtrl()

try:
    with open(skill_listName_path, 'r', encoding='utf-8') as file:
        skill_list_content = file.read()
        print(skill_list_content)
except FileNotFoundError:
    print(f"未能找到技能库名字列表的文件: {skill_listName_path}")
    sys.exit(1)  # 使用非零状态码表示错误退出
except Exception as e:
    print(f"读取文件时出错: {e}")
    sys.exit(1)  # 使用非零状态码表示错误退出
print("Skill List:\n", skill_list_content)

client = OpenAI(
    api_key=api_key, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 初始化对话
conversation_history = [
    {'role': 'system', 'content': 'You are a helpful assistant, and an extraordinary scientist/expert in robot learning.'}
]

def get_user_input():
    return input("User: ")

def chat_with_model(conversation_history):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=conversation_history,
    )
    response = completion.choices[0].message.content.strip()
    return response

prompt1 = """
Imagine your are serving an Embodied AI Assembly System. Your job is to help me analyzing the assembly task through command, then output the code command that achieves the desired goal. 
Your task is to call specific functions (api) which are predefined. The requirement is to generate code commands concisely.

You can temporarily use the following functions:
initialize_assembly(string specific_assembly_skill)
assemble(string specific_assembly_skill)
finish_assembly(string specific_assembly_skill)

"""

prompt2 = """
All of your outputs need to be identified by one of the following tags:
<question> Always ask me a clarification questions if you are unsure </question>
<reason> Explain why you did something the way you did it </reason>
<command> Output code command that achieves the desired goal </command>

For example:
Me: I want to assemble that axis with 6 corners and that object with 2 shafts.
You: <command>initialize_assembly("hexagonal axis");assemble("hexagonal axis");finish_assembly("hexagonal axis");initialize_assembly("two-axis socket 1");assemble("two-axis socket 1");finish_assembly("two-axis socket 1");</command><reason>I deduct that your command is to assemble the "hexagonal axis" and "two-axis socket 1" which can be found in the skill list, so I call the correspond api.</reason>

Are you ready?
"""

def first_conversation():
    knowledge = """
Assembly Process Knowledge Base: 
1. Planetary Gear System Assembly
Prerequisite Check: Verify if the flanged shaft is properly assembled, if not mentioned, you need to assemble the flanged shaft first.
Sequence: First, assemble the flanged shaft, then install the center gear, finally assemble the planetary gears around it.
Critical Note: Ensure proper meshing alignment to avoid gear tooth interference.
2. Ball Bearing Installation
Prerequisite Check: Confirm the housing bore is clean and free of burrs.
Sequence: Press-fit the outer ring first, then align and insert the inner ring with rolling elements.
Critical Note: Avoid misalignment during press-fitting to prevent bearing seizure.
3.Shaft-Coupling Assembly
Prerequisite Check: Ensure both shaft ends have matching keyways and are deburred.
Sequence: Slide the coupling hub onto one shaft, align, then secure with set screws or keys.
Critical Note: Check concentricity post-assembly to minimize vibration.
"""
    user_message = knowledge + scene_description + prompt1 + skill_list_content + prompt2
    conversation_history.append({'role': 'user', 'content': user_message})
    response = chat_with_model(conversation_history)
    print(f"Assistant: {response}")
    conversation_history.append({'role': 'assistant', 'content': response})

def main_loop():
    print("开始对话，输入 'exit' 以结束对话。")
    for i in range(100):  # 最多100次对话
        user_message = get_user_input()
        if user_message.lower() == 'exit':
            print("对话已结束。")
            break
        conversation_history.append({'role': 'user', 'content': user_message})
        response = chat_with_model(conversation_history)
        print(f"Assistant: {response}")
        conversation_history.append({'role': 'assistant', 'content': response})
        



if __name__ == "__main__":
    first_conversation()
    main_loop()