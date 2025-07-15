'''
此文件用于把processed_train_data.pt文件处理成轨迹序列文件
torch.save((train_trajs, train_tdpts, train_tcosts), './processed_train_data.pt')
无需打乱顺序，已经打乱了
'''

import torch
import pandas as pd

train_data = torch.load('../processed_train_data.pt')
train_trajs = train_data[0]
if isinstance(train_trajs, torch.Tensor):
    # 如果是Tensor，直接转换
    df = pd.DataFrame(train_trajs.numpy())
elif isinstance(train_trajs, list):
    # 如果是 list of list/tuple，则直接转换
    df = pd.DataFrame(train_trajs)
else:
    raise ValueError("Unsupported type for train_trajs")

# 保存为 CSV 文件
df.to_csv('./train_trajs.csv', index=False, header=False)
print('done')
print(len(train_trajs)) #=239980
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('./train_trajs.csv')

# 保存为 TXT 文件（以逗号分隔）
df.to_csv('./train_trajs.txt', index=False, sep=',', header=False)
