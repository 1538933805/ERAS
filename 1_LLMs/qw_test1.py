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
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '进行关于机器人学习领域的前沿调研'}],
    )
    
print(completion.model_dump_json())


"""
(py3.9-robolearn1) PS D:\0_Projects_2024\openai_gpt_use1> python -u "d:\0_Projects_2024\openai_gpt_use1\qw_test1.py"
{"id":"chatcmpl-8e863287-09f0-9e17-bd60-c93d0dd99ca0","choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":"### 机器人学习领域的前沿调研\n\n机器人学习（Robot Learning）是人工智能与机器人技术
的交叉领域，旨在通过让机器人从经验中学习，以提高其在复杂环境中的自主性和适应性。近年来，随着深度学习、强化学习等技术的发展，机器人学习取得了显著进展。以下是该领域的几个前沿方向：\n\n#### 1. **深度强化学习（Deep Reinforcement Learning, DRL）**\n   - **概述**：深度强化学习结合了深度神经网络和强化学习算法，使机器人能够在复杂环境中通过试错的方式学习最优策略。DRL在机器人控制、导航、抓取等方面表现出色。\n   - **挑战**：DRL的训练过程通常需要大量的样本数据，且容易陷入局部最优解。此外，真实世界的物理环境难以模拟，导致训练效率较低。\n   - **前沿研究**：\n     - **稀疏奖励问题**：如何在仅有稀疏奖励的情况下快速学习有效策略。\n     - **多智能体强化学习**：多个机器 人或代理之间的协作学习，尤其是在竞争或合作环境中。\n     - **元学习（Meta-Learning）**：使机器人能够快速适应新任务，减少对大量数据的需求。\n\n#### 2. **模仿学习（Imitation Learning）**\n   - **概述**：模仿学习通过观察人 类或其他机器人的行为来学习任务。它分为直接模仿（Behavior Cloning）和逆向强化学习（Inverse Reinforcement Learning, IRL），后者试图从专家的行为中推断出潜在的奖励函数。\n   - **前沿研究**：\n     - **高效的数据收集**：如何 通过少量的示范数据训练出高性能的模型。\n     - **泛化能力**：如何让机器人在未见过的场景中也能表现良好。\n     - **混合学习**：将模仿学习与强化学习相结合，利用两者的优点。\n\n#### 3. **自监督学习（Self-Supervised Learning）**\n   - **概述**：自监督学习是一种无需人工标注数据的学习方法，机器人通过探索环境中的内在结构进行学习。这种方法可以减少对大量标记数据的依赖，尤其适用于大规模数据集。\n   - **前沿研究**：\n     - **无监督表征学习**：通 过自动编码器、对比学习等技术，机器人可以从原始感知数据中提取有用的特征。\n     - **动作预测**：通过预测未来状态或动作，机器人可以更好地理解环境并做出决策。\n\n#### 4. **迁移学习（Transfer Learning）**\n   - **概述**：迁 移学习旨在将一个任务中学到的知识迁移到另一个相关任务中，从而加快学习速度或提高性能。这对于机器人来说尤为重要，因为现实世界中的任务往往具有相似性。\n   - **前沿研究**：\n     - **跨域迁移**：如何将从模拟环境中学到的知识迁移到真实环境中。\n     - **零样本学习（Zero-Shot Learning）**：使机器人能够在没有见过的任务上表现良好。\n     - **持续学习（Lifelong Learning）**：机器人能够在不断变化的环境中持续学习新知识，而不会忘记旧知识。\n\n#### 5. **人机协作与安全**\n   - **概述**：随着机器人越来越多地参与到人类的工作环境中，如何确保它们的安全性和与人类的有效协作成为重要课题。这涉及到机器人对人类意图的理解、安全约束的引入以及伦理问题的考虑。\n   - **前沿研究**：\n     - **共享控制（Shared Autonomy）**：通过人机协作的方式，使机器人能够根据人类的意图调整自己的行为。\n     - **安全约束学习**：如何在强化学习过程中引入安全约束，避免危险行为。\n     - **伦理与法律问题**：探讨机器人在 社会中的角色，特别是当它们可能对人类造成伤害时的责任归属。\n\n#### 6. **物理仿真与模拟到现实（Simulation-to-Reality Transfer）**\n   - **概述**：由于在真实世界中进行大量实验的成本高昂且耗时，研究人员越来越依赖于物理仿真 环境来训练机器人。然而，仿真与现实之间存在差距，如何缩小这种差距是当前的研究热点。\n   - **前沿研究**：\n     - **域随机化（Domain Randomization）**：通过在仿真中引入大量随机变化，使机器人在现实中更具鲁棒性。\n     - ** 混合现实（Mixed Reality）**：结合虚拟和物理环境，创建更真实的训练场景。\n     - **硬件在环（Hardware-in-the-Loop）**：将实际硬件与仿真环境结合起来，进行更高效的测试和验证。\n\n#### 7. **可解释性与透明度**\n   - **概述** ：随着机器人系统的复杂性增加，理解和解释其决策过程变得越来越重要。特别是在涉及安全和伦理的应用中，如自动驾驶汽车或医疗机器人，可解释性是关键。\n   - **前沿研究**：\n     - **解释性模型**：开发能够提供清晰解释的机器学习模型，帮助用户理解机器人的决策依据。\n     - **可视化工具**：通过可视化手段展示机器人的感知、推理和行动过程。\n     - **因果推理**：使机器人能够基于因果关系进行推理，而不是仅仅依赖于统计相关性。\n\n### 总结\n机器人学习是一个快速发展的领域，涵盖了从基础理论到实际应用的广泛研究。当前的前沿研究不仅关注技术上的突破，还涉及到伦理、安全和社会影响等多个方面。随着计算能力的提升和新算法的不断涌现，机器人学习有望在未来几年内取得更多突破，并在工业、医疗、农业等领域发挥更大的作用。","refusal":null,"role":"assistant","audio":null,"function_call":null,"tool_calls":null}}],"created":1735789626,"model":"qwen-plus","object":"chat.completion","service_tier":null,"system_fingerprint":null,"usage":{"completion_tokens":1153,"prompt_tokens":26,"total_tokens":1179,"completion_tokens_details":null,"prompt_tokens_details":{"audio_tokens":null,"cached_tokens":0}}}
(py3.9-robolearn1) PS D:\0_Projects_2024\openai_gpt_use1> 
"""