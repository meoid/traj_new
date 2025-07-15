'''
此文件用于把processed_test_data_half1_for_test.pt文件处理成轨迹序列文件
torch.save((train_trajs, train_tdpts, train_tcosts), './processed_train_data.pt')
无需打乱顺序，已经打乱了
'''
import torch
import pandas as pd

# 加载数据
test_data = torch.load('../processed_test_data_half1_for_test.pt')
test_trajs = test_data[0]

# 转换为 list of list，每个元素都是整数，而不是 tensor
if isinstance(test_trajs, torch.Tensor):
    # 如果是 Tensor，直接转为 numpy 后再转为 list
    trajs_list = test_trajs.numpy().tolist()
elif isinstance(test_trajs, list):
    # 如果是 list，进一步转换其中的 tensor 元素为 int
    trajs_list = []
    for traj in test_trajs:
        traj_list = [int(x) if isinstance(x, torch.Tensor) else x for x in traj]
        trajs_list.append(traj_list)
else:
    raise ValueError("Unsupported type for test_trajs")

# 保存为 CSV 文件（无表头）
df = pd.DataFrame(trajs_list)
df.to_csv('./test_trajs.csv', index=False, header=False)

# 读取 CSV 再保存为 TXT 文件（以逗号分隔）
df.to_csv('./test_trajs.txt', index=False, sep=',', header=False)

# 输出确认
print('done')
print(len(trajs_list))  # 打印轨迹数量 29998
