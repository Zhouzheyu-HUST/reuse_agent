import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import os


def encode_queries_from_json(
    json_path: str,
    save_path: str,
    model_path: str = "../gte",
    batch_size: int = 32
) -> np.ndarray:
    """
    从指定 JSON 文件中读取 query 字段，使用 SentenceTransformer 编码，
    并按顺序保存为 .npy 文件。

    参数：
        json_path: JSON 文件路径
        save_path: 生成的 .npy 文件保存路径
        model_path: SentenceTransformer 模型路径
        batch_size: 编码 batch size

    返回：
        embeddings: numpy.ndarray，shape = (N, D)
    """

    # 1. 读取 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(json_path)
    print(save_path)
    # 2. 提取 query 列表（保持原顺序）
    queries: List[str] = [
        item["query"]
        for item in data
        if "query" in item and isinstance(item["query"], str)
    ]

    if not queries:
        raise ValueError("JSON 中未找到有效的 query 字段")

    # model_path = os.path.abspath("../gte")

    # 3. 加载模型
    model = SentenceTransformer(model_path, local_files_only=True)

    # 4. 编码
    embeddings = model.encode(
        queries,
        convert_to_numpy=True,
        batch_size=batch_size
    )

    # 5. 保存为 .npy
    np.save(save_path, embeddings)

    print(f"已保存: {save_path}")
    print(f"query 数量: {len(queries)}")
    print(f"向量形状: {embeddings.shape}")

    return embeddings


# def test_encode_queries_from_json():
#     """
#     测试 encode_queries_from_json 函数的基本功能
#     """
#     json_path = "querys.json"
#     save_path = "query_test.npy"
#     model_path = "../../gte"
#
#     # 1. 调用主函数
#     embeddings = encode_queries_from_json(
#         json_path=json_path,
#         save_path=save_path,
#         model_path=model_path
#     )
#
#     # 2. 基本断言
#     assert isinstance(embeddings, np.ndarray), "返回结果不是 numpy.ndarray"
#     assert embeddings.ndim == 2, "embedding 维度不正确，应该是二维数组"
#     assert embeddings.shape[0] > 0, "embedding 数量为 0"
#
#     # 3. 检查文件是否成功保存
#     assert os.path.exists(save_path), ".npy 文件未生成"
#
#     # 4. 重新加载并验证一致性
#     loaded_embeddings = np.load(save_path)
#     assert np.allclose(embeddings, loaded_embeddings), "保存与加载的 embedding 不一致"
#
#     print("✅ 测试通过")
#     print(f"embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    a = 1 #占位
