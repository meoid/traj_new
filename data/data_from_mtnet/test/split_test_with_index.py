'''
此文件将含有index的json文件均分为10份
把原始对应的数据另存，以便mtnet使用和对比
qa_split_with_index_{i}.json: 1/10 问答对 包含索引i
raw_data_split_with_index_{i}.pt: 对应的原始三元组 , 不包含索引
real_trajs:真实轨迹 ， 不包含索引
'''
import torch
import json
import os
from math import ceil
# 加载原始轨迹数据
trajs_1, tdpts_1, tcosts_1 = torch.load('../processed_test_data_half1_for_test.pt')

# 加载QA JSON数据（包含 index 字段）
with open("./test_qa_pairs_with_index.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# 创建输出文件夹
os.makedirs("splits", exist_ok=True)

#分割10份
n = len(qa_data)
split_size = ceil(n / 10)

for i in range(10):
    start_idx = i * split_size
    end_idx = min((i + 1) * split_size, n)
    split_qa = qa_data[start_idx:end_idx]

    # 保存这一份的 QA JSON
    with open(f"splits/qa_split_with_index_{i}.json", "w", encoding="utf-8") as f:
        json.dump(split_qa, f, indent=2, ensure_ascii=False)
    print('问答对已保存')
    # 找出这一份 QA 中的所有索引
    indices = [item["index"] for item in split_qa]

    # 提取对应原始轨迹数据
    split_trajs = [trajs_1[idx] for idx in indices]
    split_tdpts = [tdpts_1[idx] for idx in indices]
    split_tcosts = [tcosts_1[idx] for idx in indices]

    # 保存完整三元组 pt,直接用于mtnet的generate
    torch.save((split_trajs, split_tdpts, split_tcosts), f"splits/raw_data_split_with_index_{i}.pt")
    print('元组已保存')
    # 只保存 trajs
    torch.save(split_trajs, f"splits/real_trajs_split_with_index{i}.pt")

    # （可选）保存 trajs 为 json 格式，便于人类查看
    with open(f"splits/real_trajs_split_with_index{i}.json", "w", encoding="utf-8") as f:
        json.dump([traj.tolist() if isinstance(traj, torch.Tensor) else traj for traj in split_trajs], f)
    print('真实轨迹已保存')



