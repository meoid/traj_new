# TrajLM 
# Trajectory Generation with GNN + Large Language Model

本项目结合图神经网络（GNN）与大语言模型（如 Qwen2.5）用于 **多步轨迹生成任务（Multi-step Trajectory Generation）**，适用于交通预测、路径规划、时空行为建模等场景。

---

## 📸 项目预览（Project Overview）

- 🔁 基于图结构的道路网络编码（GATConv）
- 🧠 结合预训练语言模型 Qwen2.5-1.5B 进行轨迹点生成
- ⚙️ 使用 LoRA 方法高效微调大模型
- 📊 多步预测轨迹点与时间间隔
- 📦 支持训练/推理分离、模块化开发

---

## 📁 项目结构

```bash
traj/
├── checkpoints/
│   ├── Qwen2.5-1.5B/       # 基础大模型
├── data/ 
│   ├── demo/               # 原始道路数据，轨迹数据                 
│   ├── data_from_mtnet/    # 从mtnet获取的数据集及数据处理结果
├── model/
│   ├── train.py            # 训练主程序
│   └── generate_test_dataset.py         # 多步轨迹生成推理脚本
│   └── traj_to_graph.py    # 可视化轨迹生成结果
└── README.md               # 项目教程文档

---

# 安装与运行
## 克隆项目
bash git clone https://github.com/your-username/traj——new.git cd traj 

## 创建环境并安装依赖 
bash conda create -n trajgen python=3.10 conda activate trajgen pip install -r requirements.txt 

项目依赖包括：transformers, torch, torch_geometric, peft, wandb 等

## 数据处理
运行文件并输出结果

## 启动训练
bash python train.py 

## 运行推理
bash python generate_test_dataset.py 

# 数据说明
· 支持多城市轨迹数据
· 使用ROAD_ID + edge_attr 构建图
· 每条轨迹包括： 起点、位置序列

# 模型架构
输入道路图嵌入 → 融合 LLM Token → 预测下一个轨迹点
Road Graph --> GNN Encoder --> [Qwen2.5 + LoRA] --> Trajectory Generator





