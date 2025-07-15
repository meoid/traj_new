import os
import re
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel
from train import QwenWithInjectedEmbedding
import argparse
from tqdm import tqdm
import random
'''
此文件用于对train.py的模型进行推理
使用全部测试集'''
def predict_trajectory_prefix(model, tokenizer, road_prefix: list, max_gen_len: int = 10, device: str = "cuda" if torch.cuda.is_available() else "cpu"
, num_beams = 3,num_return_sequences = 1):
    model.eval()
    model.to(device)
    current_traj = road_prefix.copy()
    while len(current_traj) < max_gen_len:
        # 构造输入文本
        prompt_text = "轨迹：[" + ", ".join([f"ROAD{i}" for i in current_traj]) + "]\n预测后续："
        inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=256).to(device)

        # 生成后续 token
        with torch.no_grad():
            outputs = model.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10,
                #do_sample=False,  # greedy decoding 尝试num_beams
                num_beams=num_beams,  # Beam search
                length_penalty=1.0,  # 可调整惩罚系数, 0.8-1.2 , 可以防止生成过长或过短的序列
                early_stopping=True,  # 让 beam search 早停，提高效率
                num_return_sequences=num_return_sequences,  # 返回多个序列
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        # 把张量数据传到CPU上
        outputs_cpu = outputs.cpu()
        # 解码并提取新增轨迹编号
        generated_text = tokenizer.decode(outputs_cpu[0], skip_special_tokens=True)
        #print('生成的字符串为：', generated_text)
        # 解析生成文本中的 ROAD 编号
        # 原prompt之后的新ROAD编号
        gen_part = generated_text.split("预测后续：")[-1]
        gen_roads = [int(r.replace('ROAD','')) for r in re.findall(r"ROAD\d+", gen_part)]
        if not gen_roads:
            break  # 没预测出可用内容
        # 添加一个新的轨迹编号
        next_id = gen_roads[0]
        current_traj.append(next_id)
    return current_traj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./model/train_traj_model_1/mixed_model.pth', help='路径：训练好的模型checkpoint')#tag
    parser.add_argument('--model_name_or_path', type=str, default='./model/train_traj_model_1', help='微调后的基础语言模型路径')#tag
    #parser.add_argument('--prefix', type=str, default='6072,6060,7070', help='逗号分隔的初始轨迹点 ID')
    parser.add_argument('--prefix_file', type=str, default='../data/data_from_mtnet/test/test_trajs.txt',
                        help='包含多条逗号分隔轨迹的txt文件路径')
    parser.add_argument('--output_file', type=str, default='./output/generated_trajectories.txt',
                        help='保存生成轨迹的txt文件路径')
    parser.add_argument('--max_gen_len', type=int, default=21, help='最大生成轨迹长度')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # === 加载微调后的 tokenizer 和基础模型 ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # === 加载 checkpoint ===
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    road_embeddings = checkpoint["road_embeddings"]
    model_state_dict = checkpoint["model_state_dict"]

    # === 构造混合模型 ===
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True).to(device)  # 注意：QwenWithInjectedEmbedding 内部会自动加载 base_model
    lora_model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
    model = QwenWithInjectedEmbedding(lora_model, road_embeddings, tokenizer).to(device)
    model.load_state_dict(model_state_dict)

    # === 读取轨迹文件，构造轨迹前缀
    random.seed(42)
    with open(args.prefix_file, 'r') as f:
        prefix_lines = f.readlines()
    sample_size = 500
    sampled_prefix_lines = random.sample(prefix_lines, sample_size)
    with open('./sampled_500_prefixes.txt', 'w') as fout:
        fout.write('\n'.join(sampled_prefix_lines))
    print('抽样数据已保存至./sampled_500_prefixes.txt')
    # === 打开输出文件准备写入，开始推理 ===
    with open(args.output_file, 'w') as fout:
        print("开始批量推理并保存...")
        count = 0
        for i, line in enumerate(tqdm(sampled_prefix_lines, desc='生成轨迹')):
            line = line.strip()
            if not line:
                continue
            try:
                line_ids = list(map(int, line.split(',')))
                prefix_ids = line_ids[:3]
                result = predict_trajectory_prefix(model, tokenizer, prefix_ids, max_gen_len=args.max_gen_len)
                result_str = ','.join(map(str, result))
                fout.write(f'{result_str}\n')
                count += 1
            except Exception as e:
                print(e)
    print("全部轨迹已完成，共生成：", count)

if __name__ == '__main__':
    main()

#生成轨迹： [6072, 6660, 7070, 7073, 7075, 7082, 7080, 7078, 7099, 4310, 5915, 5893, 5871, 5849, 5827, 48, 0, 0, 0, 0]
#生成轨迹： [6072, 6660, 7070, 7073, 7075, 7082, 7080, 7078, 7099, 4310, 5915, 5893, 5871, 5849, 5827, 48, 0, 0, 0, 0, 0] len=21
#生成轨迹： [6072, 6060, 7070, 7073, 7075, 7082, 7080, 7078, 7099, 4310, 5915, 5893, 5871, 5849, 5827, 48, 0, 0, 0, 0, 0]
