import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(".")
model.max_seq_length = 32

with open("query.json", "r", encoding="utf-8") as f:
    data = json.load(f)
sentences = data["sentences"] if isinstance(data, dict) else data

embeddings = model.encode(sentences, show_progress_bar=True)

np.save("embeddings.npy", embeddings)
print(f"编码完成，共 {len(sentences)} 条句子，已保存至 embeddings.npy")
