'''
此文件用于将测试数据分为两份 10%用于验证，10%用于测试，并构造问答对，保留真实的答案，以便后续评估
在生成QA时，加上INDEX
'''
import json
import os
from tqdm import tqdm
import random
import torch
import csv

# 加载数据
trajs_test, tdpts_test, tcosts_test = torch.load('./processed_test_data.pt')

# 确保类型一致，打包成元组方便一起打乱
data = list(zip(trajs_test, tdpts_test, tcosts_test))

# 打乱顺序! 每次执行代码得到的测试数据和评估数据都不一样
random.shuffle(data)
# 拆分成两组
half = len(data) // 2
group1 = data[:half]
group2 = data[half:]
# 解压成各自的三元组
trajs_1, tdpts_1, tcosts_1 = zip(*group1)
trajs_2, tdpts_2, tcosts_2 = zip(*group2)
# 如果你需要的是 list/tensor 格式，可以选择转回来：
trajs_1, tdpts_1, tcosts_1 = list(trajs_1), list(tdpts_1), list(tcosts_1) #测试集
trajs_2, tdpts_2, tcosts_2 = list(trajs_2), list(tdpts_2), list(tcosts_2) #验证集，用于微调时使用
torch.save((trajs_1, tdpts_1, tcosts_1), './processed_test_data_half1_for_test.pt')
torch.save((trajs_2, tdpts_2, tcosts_2), './processed_test_data_half2_for_eval.pt')

#读取路段属性和轨迹信息
def load_edge_properties():
    edge_properties = {}
    with open('../demo/edge_property.txt', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            try:
                road_id = int(row[0])
                road_len = float(row[1])
                road_type = int(row[2])
                heading = float(row[3])
                wkt = row[4]
                edge_properties[road_id] = {
                    'road_len': road_len,
                    'road_type': road_type,
                    'heading': heading,
                    'WKT': wkt
                }
            except Exception as e:
                print(f"[跳过行{i}] 异常：{e}，内容为：{row}")
    print(f'done! 加载了 {len(edge_properties)} 条边属性')
    return edge_properties
with open('./trajectory_properties.json', 'r') as f:
    trajectory_properties = json.load(f)

#时间戳转换函数
from datetime import datetime, timedelta
# 定义函数将时间戳转换为北京时间
def timestamp_to_beijing_time(timestamp):
    # 将时间戳转换为 UTC 时间
    utc_time = datetime.utcfromtimestamp(timestamp)
    # 北京时间是 UTC+8
    beijing_time = utc_time + timedelta(hours=8)
    # 格式化为 '年-月-日 时:分:秒' 格式
    formatted_time = beijing_time.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

#构造测试集数据问答对，包含答案!!
def generate_sample_qa_with_index(trajs, trajectory_properties, edge_properties):
    '''
    :param trajs: 轨迹序列
    :param trajectory_properties: 包含轨迹序列、轨迹起始时间、轨迹行驶时间的字典
    :param edge_properties: 路段信息
    :return: 构造的问答对
    '''

    qa_pairs = []
    #构建反向索引
    traj_index = {
        tuple(data['road_id']): (data['start_time'], data['travel_time'])
        for data in trajectory_properties.values()
    }
    for i, traj in enumerate(tqdm(trajs, total= len(trajs), desc='generate sample qa')):
        if isinstance(traj, torch.Tensor):
            traj_nopad = traj[traj > 0].tolist()
        else:
            traj_nopad = [t for t in traj if t > 0]
        #traj_nopad = [t for t in traj if t>0] #把轨迹末端填充的0去除,list
        start_time, travel_time = traj_index.get(tuple(traj_nopad), (None, None))#查找对应的出发时间和行驶时间
        if start_time is None:
            continue #跳过无法找到的
        start_links = traj_nopad[:3]  # 输入前三个轨迹
        start_links_time = travel_time[:3] #前三个轨迹的行驶时间
        start_links_pro = [edge_properties[link] for link in start_links if link > 0]#前三个轨迹的道路属性字典列表
        slens = [prop['road_len'] for prop in start_links_pro] #道路长度
        stypes = [prop['road_type'] for prop in start_links_pro] #道路类型

        next_links = traj_nopad[3:] #后缀轨迹
        next_links_time = travel_time[3:] #后续轨迹的行驶时间
        next_links_pro = [edge_properties[link] for link in next_links if link > 0]  # 后续轨迹的道路属性字典列表
        nlens = [prop['road_len'] for prop in next_links_pro] #道路长度
        ntypes = [prop['road_type'] for prop in next_links_pro] #道路类型
        #构造问答对
        context = (
        f"在成都市区内的一辆车从‘{start_time}’出发，"
        f"依次经过路段‘{start_links}’，已知对应的路段长度为{slens}，"
        f"路段类型为{stypes}，"
        f"路段行驶时间为{start_links_time}秒。"
         )
        question = (f"请预测该车辆接下来会经过的路段编号和对应的行驶时间")
        answer = (f'该车辆会依次经过的路段编号为{next_links}.此路段的行驶时间为{next_links_time}')
        # 将问题和答案存入字典
        qa_pair = {
            'index': i,
            'context': context,
            "question": question,
            "answer": answer
        }
        qa_pairs.append(qa_pair)
    return qa_pairs
edge_properties = load_edge_properties()
generate_test_qa = generate_sample_qa_with_index(trajs_1, trajectory_properties, edge_properties)
os.makedirs('./test', exist_ok=True)
with open('./test/test_qa_pairs_with_index.json', 'w', encoding='utf-8') as f:
    json.dump(generate_test_qa, f, ensure_ascii=False, indent=2)
print('done')
generate_eval_qa = generate_sample_qa_with_index(trajs_2, trajectory_properties, edge_properties)
os.makedirs('./eval', exist_ok=True)
with open('./eval/eval_qa_pairs_with_index.json', 'w', encoding='utf-8') as f:
    json.dump(generate_eval_qa, f, ensure_ascii=False, indent=2)
print('done')