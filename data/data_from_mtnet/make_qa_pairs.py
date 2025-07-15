'''
此文件用于将训练数据构造为问答对 80%
并且将全部轨迹的 100% 道路属性和时间信息整理成字典保存
'''
import torch
import json
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import csv

# 加载保存的数据
trajs_train, tdpts_train, tcosts_train = torch.load('./processed_train_data.pt')
print(tdpts_train.shape)
print(tcosts_train.shape)
print(trajs_train.shape)
#加载路段属性数据
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
    with open('./edge_properties.json', 'w') as f:
        json.dump(edge_properties, f)
    print(f'done! 加载了 {len(edge_properties)} 条边属性')
    return edge_properties

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

def load_trajs_law():
    '''
    此文件用于处理轨迹文件和时间文件，得到需要的出发时间和轨迹行驶时间
    :return:
    '''
    # 初始化空字典，用于存储轨迹编号及其对应的道路编号和行驶时间，出发时间
    trajectory_properties = {}
    with open('../demo/trajs_demo.csv', "r") as ftraj:
        with open('../demo/tstamps_demo.csv', 'r') as ft:  # read trajs and tstamps
            # 统计行数
            traj_lines = sum(1 for _ in ftraj)
            ft_lines = sum(1 for _ in ft)
            # 确保两个文件的行数一致
            assert traj_lines == ft_lines, "The two files have different number of lines!"
            # 重新打开文件并使用 range 进行迭代
            ftraj.seek(0)  # 重置文件指针
            ft.seek(0)  # 重置文件指针
            for i in range(traj_lines):  # 根据行数遍历
                line_traj = ftraj.readline().strip()  # 读取一行并去除换行符
                line_t = ft.readline().strip()  # 读取一行并去除换行符
                # 将每一条轨迹数据的道路编号以整数形式存储为列表
                traj = [int(x) for x in line_traj.split()]  # 道路编号
                if True: traj = traj[:-1]  # 去掉最后一个元素 0
                # 将每一条时间数据处理成出发时间和行驶时间
                tdpt = [float(x) for x in line_t.split()]
                travel_time = tdpt[1:]  # 行驶时间
                start_time = timestamp_to_beijing_time(tdpt[0])  # 出发时间(年月日形式）
                #轨迹段数应与行驶时间长度一致
                if len(traj) != len(travel_time):
                    raise ValueError(
                        f'The length of traj ({len(traj)}) and travel_time ({len(travel_time)}) do not match!')
                # 构建轨迹数据字典，从1开始编号轨迹
                trajectory_properties[i + 1] = {
                    'road_id': traj,
                    'start_time': start_time,
                    'travel_time': travel_time,
                }
    with open('./trajectory_properties.json', 'w') as f:
        json.dump(trajectory_properties, f)
    print('done!')
    return trajectory_properties

#构造训练数据问答对
def generate_sample_qa(trajs, trajectory_properties, edge_properties):
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
    for traj in tqdm(trajs, total= len(trajs), desc='generate sample qa'):
        if isinstance(traj, torch.Tensor):
            traj_nopad = traj[traj > 0].tolist()
        else:
            traj_nopad = [t for t in traj if t > 0]
        #traj_nopad = [t for t in traj if t>0] #把轨迹末端填充的0去除,list
        start_time, travel_time = traj_index.get(tuple(traj_nopad), (None, None))#查找对应的出发时间和行驶时间

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
            'context': context,
            "question": question,
            "answer": answer
        }
        qa_pairs.append(qa_pair)
    os.makedirs('./train', exist_ok=True)
    with open('./train/train_qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print('done')
    return qa_pairs

edge_properties = load_edge_properties()
trajectory_properties = load_trajs_law()
generate_train_qa = generate_sample_qa(trajs_train, trajectory_properties, edge_properties)
print('done!')

