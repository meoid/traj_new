import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from shapely import wkt
from shapely.geometry import LineString
from tqdm import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model

'''
此文件为模型训练脚本
使用完整训练数据集进行微调
'''
# 设置参数
num_roads = 8263
feat_dim = 32
hidden_dim = 256
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#曲率计算函数
def compute_curvature(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 3:
        return 0.0
    total_angle = 0.0
    for i in range(1, len(coords) - 1):
        a, b, c = coords[i - 1], coords[i], coords[i + 1]
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        total_angle += abs(angle2 - angle1)
    return total_angle / line.length if line.length > 0 else 0.0
#图神经网络
class RoadGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, edge_dim=6, add_self_loops=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=6, add_self_loops=False)
        self.out_proj = nn.Identity() if out_dim is None else nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        return self.out_proj(x)
#轨迹数据集
class TrajDataset(Dataset):
    def __init__(self, traj_list, tokenizer: PreTrainedTokenizer, road_embeds):
        self.traj_list = traj_list
        self.tokenizer = tokenizer
        self.road_embeds = road_embeds
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    def __len__(self):
        return len(self.traj_list)
    def __getitem__(self, idx):
        traj = self.traj_list[idx]
        if traj[-1] == 0:
            traj = traj[:-1]
        prompt_ids = traj[:3]
        target_ids = traj[3:]
        prompt_text = "轨迹：[" + ", ".join([f"ROAD{i}" for i in prompt_ids]) + "]\n预测后续："
        target_text = ", ".join([f"ROAD{i}" for i in target_ids])
        enc = self.tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=256)
        dec = self.tokenizer(target_text, return_tensors='pt', truncation=True, max_length=256)
        input_ids = torch.cat([enc['input_ids'], dec['input_ids'][:, :-1]], dim=1)
        labels = torch.cat([
            torch.full(enc['input_ids'].shape, -100),
            dec['input_ids']
        ], dim=1)
        return {
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0),
            "prompt_ids": torch.tensor(prompt_ids)
        }
#自定义数据聚合函数
def traj_collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    prompt_ids = [item["prompt_ids"] for item in batch]
    encodings = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    attention_mask = encodings["attention_mask"]
    max_len = encodings["input_ids"].size(1)
    padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :min(label.size(0), max_len)] = label[:max_len]
    padded_prompt_ids = torch.stack(
        [torch.cat([prompt, torch.full((max_len - prompt.size(0),), -100, dtype=torch.long)]) for prompt in prompt_ids])
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": attention_mask,
        "labels": padded_labels,
        "prompt_ids": padded_prompt_ids,
    }
#混合模型类定义
class QwenWithInjectedEmbedding(nn.Module):
    def __init__(self, model, road_embeddings, tokenizer):
        super().__init__()
        self.model = model
        self.road_embeddings = road_embeddings
        self.tokenizer = tokenizer
        self.embedding_layer = model.get_input_embeddings()
        # 构造 token_id 到 road_id 的反向映射表
        self.tokenid_to_roadid = {
            tokenizer.convert_tokens_to_ids(f"ROAD{i}"): i
            for i in range(len(road_embeddings))
        }

    def forward(self, input_ids, attention_mask=None, labels=None, prompt_ids=None):
        inputs_embeds = self.embedding_layer(input_ids)
        # 遍历 input_ids 找到 ROADxxx token
        for i in range(input_ids.shape[0]):  # batch size
            for j in range(input_ids.shape[1]):  # sequence length
                token_id = input_ids[i, j].item()
                if token_id in self.tokenid_to_roadid:
                    road_id = self.tokenid_to_roadid[token_id]
                    inputs_embeds[i, j, :] = self.road_embeddings[road_id]
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
#道路图结构加载函数
def load_road_graph():
    node_features = torch.rand((num_roads, feat_dim))
    edge_starts = []
    with open('../data/demo/adjacency.csv', 'r') as f:
        for line in f:
            src = int(line.strip().split(',')[0]) - 1
            edge_starts.append(src)
    with open('../data/demo/adjacency.csv', 'r') as f:
        adj_lines = [line.strip().split(',') for line in f]
    src_nodes, dst_nodes = [], []
    for src in edge_starts:
        neighbors = [int(x.strip()) for x in adj_lines[src] if int(x.strip()) != -1]
        if not neighbors:
            raise ValueError(f"No neighbor for {src}")
        dst = neighbors[0] - 1
        src_nodes.append(src)
        dst_nodes.append(dst)
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attrs = []
    with open('../data/demo/edge_property.csv', 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            length, level, angle = float(parts[1]), float(parts[2]), float(parts[3])
            line_str = ','.join(parts[4:]).strip('"')
            line_geom = wkt.loads(line_str)
            num_pts = len(line_geom.coords)
            avg_seg_len = length / (num_pts - 1) if num_pts > 1 else 0
            curvature = compute_curvature(line_geom)
            edge_attrs.append([length, level, angle, curvature, num_pts, avg_seg_len])
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
#轨迹数据加载
def load_trajectory_data(file_path):
    traj_list = []
    with open(file_path, 'r') as f:
        for line in f:
            traj = list(map(int, line.strip().split(',')))
            traj_list.append(traj)
    return traj_list
#训练函数 tag
def train(model, train_dataloader, val_dataloader, optimizer, num_epochs, log_interval=50, patience=3):
    model.train()
    epoch_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        step_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            prompt_ids = batch["prompt_ids"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, prompt_ids=prompt_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            # 每 log_interval 步记录一次 loss
            if (step + 1) % log_interval == 0:
                avg_step_loss = total_loss / (step + 1)
                step_losses.append(avg_step_loss)
                wandb.log({
                    "train/step_loss": avg_step_loss,
                    "epoch": epoch + 1,
                    "step": step + 1
                })
            progress_bar.set_postfix(loss=loss.item())
        avg_epoch_loss = total_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)
        wandb.log({
            "train/epoch_loss": avg_epoch_loss,
            "epoch": epoch + 1
        })
        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                prompt_ids = batch["prompt_ids"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, prompt_ids=prompt_ids)
                loss = outputs.loss
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        wandb.log({
            "val/loss": avg_val_loss,
            "epoch": epoch + 1
        })
        print(f"Epoch {epoch + 1}: Train Loss = {avg_epoch_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
        # ===== Early Stopping =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "./best_model.pt")  # 保存当前最佳模型
        else:
            patience_counter += 1
            print(f"EarlyStopping Counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

#主函数
def main():
    wandb.init(project="graphpathlm", name="run1")#tag

    graph_data = load_road_graph()
    gnn_model = RoadGNN(feat_dim, hidden_dim).to(device)
    with torch.no_grad():
        road_embeddings = gnn_model(graph_data.to(device))

    model_path = "../checkpoints/Qwen2.5-1.5B"
    base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none"
    )
    base_model = get_peft_model(base_model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    mixed_model = QwenWithInjectedEmbedding(base_model, road_embeddings, tokenizer).to(device)
    #修改训练数据集和验证数据集 tag
    train_traj_list = load_trajectory_data("../data/data_from_mtnet/train/train_trajs.txt")#tag
    val_traj_list = load_trajectory_data('../data/data_from_mtnet/eval/val_trajs.txt')
    #traj_list = load_trajectory_data("../data/data_from_mtnet/train/train_trajs.txt")[:24000]
    train_dataset = TrajDataset(train_traj_list, tokenizer, road_embeddings)
    val_dataset = TrajDataset(val_traj_list, tokenizer, road_embeddings)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: traj_collate_fn(x, tokenizer))
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True,
                                  collate_fn=lambda x: traj_collate_fn(x, tokenizer))
    optimizer = torch.optim.AdamW(mixed_model.parameters(), lr=5e-5)
    train(mixed_model, train_dataloader, val_dataloader, optimizer, num_epochs=5)#tag

    #保存模型参数
    save_dir = './model/train_traj_model_1'#tag
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "model_state_dict": mixed_model.state_dict(), #保存混合模型状态字典
        "road_embeddings": road_embeddings.cpu(),  #保证兼容 CPU 加载，保存道路嵌入
    }, os.path.join(save_dir, "mixed_model.pth"))
    tokenizer.save_pretrained(save_dir) #保存分词器
    base_model.save_pretrained(save_dir) #保存大模型
    wandb.finish()
    print("模型训练完成并保存")

if __name__ == "__main__":
    main()
