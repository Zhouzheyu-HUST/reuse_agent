from custom.sim.query_to_npy import encode_queries_from_json
from custom.sim.user_to_index import load_old_tasks
import json
import os
import numpy as np
import re
from pathlib import Path
from utils import (
    read_json
)

GTE_MODEL_NAME = "gte"

def check_and_encode(base_dir):
    json_path = os.path.join(base_dir, "querys.json")
    npy_path = os.path.join(base_dir, "querys.npy")

    if not os.path.exists(json_path):
        print(f"[WARN] 找不到 {json_path}，跳过检查。")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持两种 json 格式
    if isinstance(data, dict) and "sentences" in data:
        sentences = data["sentences"]
    elif isinstance(data, list):
        sentences = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and "query" in item:
                sentences.append(item["query"])
            else:
                raise ValueError(f"{json_path} 格式错误：第 {i} 项缺少 'query' 字段或不是对象")
    else:
        raise ValueError(f"不支持的历史指令 JSON 格式：{json_path}")

    # 判断 NPY 文件是否存在 (包含之前提到的防崩溃逻辑)
    if not os.path.exists(npy_path):
        print(f"[INFO] 找不到 {npy_path}，准备进行初次编码...")
        encode_queries_from_json(json_path, npy_path, GTE_MODEL_NAME)
        print(f"[INFO] {base_dir} 编码完成，已保存到 {npy_path}")
        return

    # 存在则加载并判断长度
    embeddings = np.load(npy_path)
    if embeddings.shape[0] != len(sentences):
        print(f"{json_path} 中的句子数量与 {npy_path} 的向量数量不一致, 重新编码...")  
        os.remove(npy_path)
        print(f"[INFO] 正在将 {json_path} 编码为 {npy_path} ...")
        encode_queries_from_json(json_path, npy_path, GTE_MODEL_NAME)
        print(f"[INFO] {base_dir} 编码完成，已保存到 {npy_path}")
    else:
        print(f"{json_path} 中的句子数量与 {npy_path} 的向量数量一致，无需重新编码。")  

def encode_npy():
    """
    检查并编码 database 目录下的 tasks 和 part_reuse_tasks 文件夹
    """
    target_dirs = [
        "database/tasks",
        "database/part_reuse_tasks"
    ]
    
    for directory in target_dirs:
        print(f"\n>>> 开始检查目录: {directory}")
        check_and_encode(directory)
        
    return 0

if __name__ == "__main__":
    encode_npy()