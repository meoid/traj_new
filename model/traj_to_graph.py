import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
import os
import csv

def plot_roads_and_trajectories(road_txt_path,
                                trajectory_txt_path='',
                                num_samples=1,
                                output_png_path='',
                                csv_output_path='./connectivity_scores.csv'):
    # 读取道路数据
    road_dict = {}
    with open(road_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            road_id = parts[0]
            wkt_str = ','.join(parts[4:]).strip('"')
            try:
                geom = wkt.loads(wkt_str)
                if geom.geom_type == 'LineString':
                    road_dict[int(road_id)] = geom
            except Exception as e:
                print(f"解析WKT失败: {wkt_str}, 错误: {e}")

    # 加载轨迹数据-txt文件
    if trajectory_txt_path and os.path.exists(trajectory_txt_path):
        with open(trajectory_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        trajectories = []
        for line in lines:
            tokens = line.strip().split(',')
            if tokens:
                traj = [int(t) for t in tokens if t.strip().isdigit()]
                trajectories.append(traj)
        max_len = max(len(t) for t in trajectories)
        traj_tensor = torch.zeros((len(trajectories), max_len), dtype=torch.long)
        for i, traj in enumerate(trajectories):
            traj_tensor[i, :len(traj)] = torch.tensor(traj, dtype=torch.long)

        num_traj = traj_tensor.shape[0]
        sample_indices = random.sample(range(num_traj), min(num_samples, num_traj))
        print(f"采样轨迹索引: {sample_indices}")

    else:
        print("未提供轨迹文件，使用默认轨迹")
        traj_list = [6072, 6660, 7070, 7073, 7075, 7082, 7080, 7078, 7099, 4310, 5915, 5893, 5871, 5849, 5827, 48, 0, 0, 0, 0]
        traj_tensor = torch.tensor([traj_list], dtype=torch.long)
        sample_indices = [0]

    # 开始画图
    fig, ax = plt.subplots(figsize=(12, 12))

    # 先画道路网络
    for road_id, line in road_dict.items():
        x, y = line.xy
        ax.plot(x, y, color='lightgray', linewidth=0.8)

    # 配色方案
    colors = plt.cm.get_cmap('tab10', len(sample_indices))
    scores = []
    # 绘制每条轨迹
    for idx, sample_idx in enumerate(sample_indices):
        traj = traj_tensor[sample_idx]
        traj = traj[traj != 0]  # 去掉padding
        points = []

        for road_id in traj.tolist():
            if road_id in road_dict:
                line = road_dict[road_id]
                mid_idx = len(line.coords) // 2
                mid_point = line.coords[mid_idx]
                points.append(mid_point)
            else:
                print(f"警告：轨迹中road_id {road_id} 不在道路数据里！")

        if len(points) >= 2:
            x, y = zip(*points)
            ax.plot(x, y, marker='o', markersize=3, linewidth=2,
                    color=colors(idx), label=f'Traj {idx}')

            # 连通性得分
            dists = np.linalg.norm(np.diff(np.array(points), axis=0), axis=1)
            connectivity_score = np.exp(-np.mean(dists))
            #print(f"轨迹 {idx} 的连通性得分: {connectivity_score:.4f}")
            scores.append((idx, connectivity_score))
        else:
            print(f"轨迹 {idx} 点数不足，无法计算连通性得分。")
            scores.append((idx, connectivity_score))

    #写入csv文件
    if csv_output_path:
        with open(csv_output_path, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['idx', 'connectivity_score'])
            for idx, score in scores:
                writer.writerow([idx, score])
        # 打印平均得分
    valid_scores = [s for _, s in scores if s > 0]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\n平均连通性得分: {avg_score:.4f}")
    else:
        print("没有有效轨迹用于计算平均连通性得分。")


    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Generated Trajectories on Road Network')

    if len(sample_indices) > 1:
        ax.legend()

    if output_png_path:
        plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.show()

# 示例调用
plot_roads_and_trajectories(
    road_txt_path='../../data/demo/edge_property.txt',
    trajectory_txt_path='../../GraphPathLM/output/generated_trajectories.txt',
    num_samples=10,
    output_png_path='./generated_map_2.png'
)
print('done!')
