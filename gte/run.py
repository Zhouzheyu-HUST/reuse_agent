from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import time

# ===== 1. 加载模型 =====
model = SentenceTransformer("/root/gte")
model.max_seq_length = 32

# ===== 2. 加载已有的句子向量 =====
# 假设之前保存的向量文件是 embeddings.npy
start = time.perf_counter()
embeddings_all = np.load("embeddings.npy")  # shape: (N, D)

# ===== 3. 要比较的新句子 =====
sentences = ['帮我点一杯星巴克的拿铁']

# ===== 4. 编码新句子 =====
embedding_new = model.encode(sentences)  # shape: (1, D)
#print(f"编码耗时: {end - start:.4f} 秒")

# ===== 5. 计算余弦相似度 =====
# cos_sim 会自动广播，返回一个 (1, N) 的相似度矩阵
similarities = cos_sim(embedding_new, embeddings_all)[0]  # shape: (N,)

# ===== 6. 找出最相似的句子编号（从 1 开始） =====
best_idx = int(np.argmax(similarities))  # 0-based
best_score = float(similarities[best_idx])
print(f"最相似句子的编号: {best_idx + 1}")
end = time.perf_counter()
print(f"编码和比对总耗时: {end - start:.4f} 秒")
#print(f"相似度得分: {best_score:.4f}")
