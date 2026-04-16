import json
import numpy as np
from typing import List, Tuple, Optional
import time
import re
import os
import requests  # 替代 DashScope SDK
from pathlib import Path
from rank_bm25 import BM25Okapi
from http import HTTPStatus

# ==== GTE ====
from sentence_transformers import SentenceTransformer

def load_api_config(filename="api_settings.json"):
    """
    从当前脚本所在目录向上递归查找 custom/api_settings.json
    """
    # 获取当前文件所在的绝对目录
    current_path = Path(__file__).resolve().parent
    
    # 向上遍历所有父级目录 (包含当前目录)
    for parent in [current_path] + list(current_path.parents):
        # 检查是否存在 configs 文件夹下的配置文件
        config_file = parent / "custom" / filename
        if config_file.exists():
            try:
                # print(f"[Info] Loaded config from: {config_file}")
                return json.loads(config_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error loading config: {e}")
                print(f"未成功找到配置文件 {filename}，请确保其存在于 'configs' 文件夹中。")
                return {}
    
    print(f"Warning: {filename} not found in 'configs' folder of any parent directory.")
    return {}

_config = load_api_config()
API_URL = _config.get("llm_endpoints")
API_KEY = _config.get("llm_api_key")
MODEL_NAME = _config.get("llm_model")
LLM_CHECK_WAY = _config.get("llm_check_way")

'''
#这里还有可以手动设置的部分，以防出现问题无法载入。
API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "sk-1234"           
MODEL_NAME = "Qwen3-VL-8B-Instruct"  
'''
# ================== 原有配置区 ==================
# 旧指令文本 & 向量
OLD_QUERY_JSON = "sim/old_query.json"  # { "sentences": [ ... ] }
OLD_QUERY_NPY = "sim/old_query.npy"    # shape = (N, D)
# TOP-K
TOP_K = 4
# embedding筛选阈值
SIM_THRESHOLD = 0.7
# 这是一个旧路径，如果本地不再加载权重，其实可以忽略，但为了保持变量存在，保留它
QWEN_MODEL_DIR = "/root/autodl-tmp/Qwen3-8B"


# ================== 新增：本地 LLM 调用包装器 ==================
class TextLlmWrapper:
    """
    专门用于处理纯文本请求的包装器，包含重试机制。
    替换原有的 Generation.call 功能。
    """
    RETRY_WAITING_SECONDS = 2
    MAX_RETRY = 3

    def __init__(self, api_key: str, model_name: str, api_url: str, check_way: str = "openai"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.check_way = check_way

    def predict(self, messages: List[dict], temperature: float = 0.7) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
            "stream": False
        }

        if self.check_way == "openai":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        elif self.check_way == "csb":
            headers = {
                "Content-Type": "application/json",
                "csb-token": self.api_key
            }

        counter = self.MAX_RETRY
        wait_seconds = self.RETRY_WAITING_SECONDS

        while counter > 0:
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60  # 设置超时
                )

                if response.ok:
                    data = response.json()
                    # 兼容 OpenAI 格式返回
                    if "choices" in data and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"]
                
                print(f"Error calling LLM: Status {response.status_code}, Body: {response.text}")
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1

            except Exception as e:
                print(f"Exception calling LLM: {e}")
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1

        return None

# 初始化全局 Wrapper
llm_wrapper = TextLlmWrapper(API_KEY, MODEL_NAME, API_URL, LLM_CHECK_WAY)


# ================== 工具函数 ==================

def load_old_tasks(json_path: str, npy_path: str) -> Tuple[List[str], np.ndarray]:
    """加载历史指令（自然语言 + 向量）"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持两种 json 格式：
    # 1) 旧格式：{ "sentences": [ ... ] }
    # 2) 新格式（querys.json）：[ { "task_id": ..., "query": ..., ... }, ... ]
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

    embeddings = np.load(npy_path)
    if embeddings.shape[0] != len(sentences):
        raise ValueError(f"{json_path} 中的句子数量与 {npy_path} 的向量数量不一致！")
    return sentences, embeddings


def call_cloud_for_candidates(user_input: str) -> List[str]:
    """
    使用本地大模型 API 对用户指令做语义归一化，
    返回候选指令列表（>=3 条）。
    """
    # 保持原有的完整 Prompt
    system_prompt_content = '''你是一个面向 GUI agent 的语义归一化助手。你的任务是：
你是一个面向 GUI agent 的语义归一化助手。你的任务是：

理解用户输入的自然语言指令。

在内心判断该指令最适合归类到以下五种指令类别中的哪一种（只在思考中判断，不要把类别名称输出给用户）。

选定一个最合适的类别后，严格按照该类别对应的语义模板，生成至少 3 条语义等价但表述略有差异的候选指令。

所有候选指令必须是自然语言句子，用户可以直接理解。
你只能输出候选指令本身，每条指令单独一行，不允许输出任何解释、类别名称、步骤说明、JSON、代码或列表结构。

以下是五类指令类别和模板，你需要在内部使用它们进行分类和生成。

类别 1：控制类或配置类指令
适用范围：调节系统或应用的状态，如音量、亮度、网络、开关、模式等。
语义模板（概念框架）：
[动作] + [对象] + [属性/参数/程度/类型/量化]
动作示例：调高、调低、打开、关闭、设置、切换、启用、禁用
对象示例：音量、屏幕亮度、蓝牙、WiFi、飞行模式、省电模式
参数示例：一级、两格、50%、夜间模式、静音模式
示例（不要原样输出）：
调高音量一级
设置屏幕亮度为五十百分比
打开飞行模式

类别 2：路径类或多层级导航指令
适用范围：逐层进入多个界面或菜单，例如“设置 → 隐私 → 定位”。
语义模板（概念框架）：
[动作] + [路径节点1] + [路径节点2] + [路径节点3或最终操作]
动作示例：进入、打开、前往、切换到
路径示例：设置、隐私、定位服务、相册、最近、通知设置
示例（不要原样输出）：
进入设置 进入隐私 关闭定位服务
打开相册 进入最近文件夹
进入微信 进入支付页面

类别 3：可视控件类或界面控件操作指令
适用范围：按钮、弹窗、标签页、图标等控件操作。
语义模板（概念框架）：
[动作] + [控件的定位信息] + [可选的结果]
动作示例：点击、轻点、长按、双击、切换、关闭、展开
控件定位信息可描述为：按钮文本、屏幕位置、控件类型、顺序位置等
示例（不要原样输出）：
点击右上角菜单按钮 打开更多选项
点击允许按钮 同意权限申请
关闭提示窗口
切换到底部第二个标签页

类别 4：手势类指令
适用范围：滑动、拖动、缩放、长按等触摸手势。
语义模板（概念框架）：
[手势动作] + [方向/目标区域/对象] + [可选结果]
手势动作示例：上滑、下滑、左滑、右滑、长按、双击、拖动、缩放
目标示例：解锁界面、通知栏、页面底部、当前卡片、图片
示例（不要原样输出）：
上滑解锁屏幕
下滑打开通知栏
左滑返回上一页
长按应用图标 打开快捷菜单

类别 5：任务类或内容操作类指令
适用范围：发送消息、创建记录、搜索内容等具有内容目标的操作。
语义模板（概念框架）：
[动作] + [对象/载体] + [任务内容或文本内容]
动作示例：发送、创建、新建、记录、搜索、查找、添加
对象示例：消息、短信、备忘录、日程、联系人、搜索栏
示例（不要原样输出）：
给小明发送消息 我到了
在备忘录创建新笔记
搜索手机壳优惠信息

你的工作流程：
内部判断用户输入属于五类中的哪一类，不要输出类别。
只使用该类对应的语义模板，生成三条语义等价的候选自然语言指令。
指令必须符合模板中的槽位结构，但不输出方括号，占位符需要替换为自然语言。
只输出候选指令，不输出任何说明。
每条指令必须独立一行，不能有序号或前缀符号。


请对用户接下来的每一句输入，生成三条符合上述要求的候选指令,最后不要有句号。
 
'''

    messages = [
        {'role': 'system', 'content': system_prompt_content},
        {'role': 'user', 'content': user_input}
    ]

    start_time = time.time()
    
    # 替换点：使用本地 wrapper
    output_text = llm_wrapper.predict(messages, temperature=0.7)
    
    end_time = time.time()
    elapsed = end_time - start_time

    if output_text:
        # 去除首尾空白
        output_text = output_text.strip()
        
        # 解析每行候选句
        candidate_sentences = [line.strip()
                               for line in output_text.splitlines()
                               if line.strip()]

        if len(candidate_sentences) < 3:
            print("⚠ 云端返回候选不足 3 条，当前为：", len(candidate_sentences))
        print("\n候选句列表:")
        for i, sent in enumerate(candidate_sentences, start=1):
            print(f"{i}. {sent}")
        # print(f"\n语义归一化用时：{elapsed:.2f} 秒")

        return candidate_sentences
    else:
        print("❌ 请求错误: LLM 返回为空")
        raise RuntimeError("云端 LLM 调用失败")


def cosine_sim_with_all(new_vec: np.ndarray, old_vecs: np.ndarray) -> np.ndarray:
    """
    计算 new_vec 与 old_vecs 中每一行的余弦相似度
    """
    new_norm = new_vec / (np.linalg.norm(new_vec) + 1e-12)
    old_norm = old_vecs / \
        (np.linalg.norm(old_vecs, axis=1, keepdims=True) + 1e-12)
    sims = old_norm @ new_norm  # shape = (N,)
    return sims


# ================== Qwen3-8B 辅助智能体 ==================

class QwenSelector:
    def __init__(self, api_key: str):
        # 保持签名一致，但内部使用全局 llm_wrapper，所以这个 api_key 参数实际上是为了兼容性
        self.api_key = api_key 

    def select_best(self, user_instruction: str, candidates: List[str]) -> int:
        """
        使用本地 Qwen API 从 candidates 中选出语义最相似的一条
        返回候选的局部序号（0-based）
        """
        # 输出有多少候选指令
        #print(f"\n生成了 {len(candidates)} 条候选指令 ...")
        numbered = "\n".join([f"{i + 1}. {sent}" for i, sent in enumerate(candidates)])
        #print(f"\nQwenSelector 的系统提示:\n{numbered}\n用户指令:\n{user_instruction}\n")
        system_prompt = (
            "你是一个语义相似度判别工具，现在有这" + str(len(candidates)) + "条老指令,每条老指令前面有一个数字编号,它们是：" +
            numbered +
            "，现在给你一条新指令，从老指令中找出最相似的一条，只输出该指令的编号，不要有任何解释和复述。如果在老指令中没有在语义上相似的指令（注意是人类理解上的相似，不是字词上的相似），则输出0"
            
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"新指令：{user_instruction}"}
        ]

        # 替换点：使用本地 wrapper，且 result_format 处理方式改为字符串提取
        # 判别任务建议 temperature 低一点
        resp_content = llm_wrapper.predict(messages, temperature=0.1)
        #print(f"\nQwenSelector 原始输出: {resp_content}")
        if not resp_content:
             raise RuntimeError(f"云端调用失败")

        resp = resp_content.strip()

        # 提取数字
        m = re.search(r"\d+", resp)
        if not m:
            return -1

        idx = int(m.group()) - 1
        if not (0 <= idx < len(candidates)):
            return -1

        return idx


# ================== 主流程 ==================

def find_best_match(chosen_instruction: str, gte_model_name: str, old_query_json_path: str, old_query_npy_path: str) -> int:
    """
    输入：用户最终选择的一条规范化指令（字符串）
    输出：
        - 匹配到的全局序号（1-based）
        - 如果没有可复用任务，则返回 0
    """

    if not os.path.exists(old_query_json_path):
        return 0

    if not os.path.exists(old_query_npy_path):
        return 0

    # embedding相似度筛选TopK
    embedder = SentenceTransformer(gte_model_name, local_files_only=True)
    new_vec = embedder.encode(chosen_instruction, convert_to_numpy=True)

    old_sentences, old_embeddings = load_old_tasks(
        old_query_json_path, old_query_npy_path)

    embed_sims = cosine_sim_with_all(new_vec, old_embeddings)

    # ===== Embedding 阈值过滤 =====
    valid_mask = embed_sims >= SIM_THRESHOLD
    if not np.any(valid_mask):
        print("Embedding 全部低于阈值")
        return 0

    # BM25筛选TopK（字符级）
    tokenized_corpus = [list(doc) for doc in old_sentences]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(chosen_instruction)
    bm25_scores = bm25.get_scores(tokenized_query)

    # ===== 分数归一化 =====
    bm25_norm = bm25_scores / (np.max(bm25_scores) + 1e-12)
    embed_norm = embed_sims  # cosine本身0~1

    # ===== 混合得分 =====
    hybrid_scores = 0.7 * embed_norm + 0.3 * bm25_norm

    # 找到所有有效索引
    valid_indices = np.where(valid_mask)[0]

    # 取这些位置的 hybrid score
    valid_scores = hybrid_scores[valid_indices]

    # 在有效集合里排序
    sorted_local_idx = np.argsort(-valid_scores)

    # 取 TopK
    top_local_idx = sorted_local_idx[:TOP_K]

    # 映射回原始全局 index
    candidate_global_indices = valid_indices[top_local_idx]

    candidate_texts = [old_sentences[i] for i in candidate_global_indices]

    if len(candidate_global_indices) == 0:
        print("\n无可复用任务")
        return 0

    qwen = QwenSelector(API_KEY)
    
    local_idx = qwen.select_best(chosen_instruction, candidate_texts)
    if local_idx == -1:
        global_idx = -1
    else:
        global_idx = candidate_global_indices[local_idx]

    return global_idx + 1  # 返回1~based序号

def find_best_match_multi(chosen_instruction: str, gte_model_name: str, json_paths: list, npy_paths: list):
    """
    在多个历史 query 文件中进行匹配，返回匹配到的 (local_index_1based, source_json_path)。
    如果没有匹配到，返回 (0, None)。
    """
    # 收集所有存在的语句与向量，并记录每条句子来源 (json_path, local_idx)
    all_sentences = []
    embeddings_parts = []
    index_map = []  # maps global idx -> (json_path, local_idx)

    for json_path, npy_path in zip(json_paths, npy_paths):
        if not os.path.exists(json_path) or not os.path.exists(npy_path):
            continue
        try:
            sents, embs = load_old_tasks(json_path, npy_path)
        except Exception:
            continue

        for i, s in enumerate(sents):
            index_map.append((json_path, i))
        all_sentences.extend(sents)
        embeddings_parts.append(embs)

    if not all_sentences:
        return 0, None

    # 合并 embeddings
    try:
        embeddings_concat = np.concatenate(embeddings_parts, axis=0)
    except Exception:
        # fallback: if concatenation fails, abort
        return 0, None

    # 下面采用与单文件 find_best_match 相同的策略：
    # - 计算 embedding 相似度并应用阈值过滤
    # - 计算 BM25 分数并归一化
    # - 计算混合得分并在有效集合中取 TopK
    embedder = SentenceTransformer(gte_model_name, local_files_only=True)
    new_vec = embedder.encode(chosen_instruction, convert_to_numpy=True)

    embed_sims = cosine_sim_with_all(new_vec, embeddings_concat)

    # Embedding 阈值过滤
    valid_mask = embed_sims >= SIM_THRESHOLD
    if not np.any(valid_mask):
        print("Embedding 全部低于阈值")
        return 0, None

    # BM25 计算
    tokenized_corpus = [list(doc) for doc in all_sentences]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(chosen_instruction)
    bm25_scores = bm25.get_scores(tokenized_query)

    # 分数归一化
    bm25_norm = bm25_scores / (np.max(bm25_scores) + 1e-12)
    embed_norm = embed_sims

    # 混合得分
    hybrid_scores = 0.7 * embed_norm + 0.3 * bm25_norm

    # 在有效集合中排序并取 TopK
    valid_indices = np.where(valid_mask)[0]
    valid_scores = hybrid_scores[valid_indices]
    sorted_local_idx = np.argsort(-valid_scores)
    top_local_idx = sorted_local_idx[:TOP_K]
    candidate_global_indices = valid_indices[top_local_idx]

    if len(candidate_global_indices) == 0:
        print("\n无可复用任务")
        return 0, None

    candidate_texts = [all_sentences[i] for i in candidate_global_indices]

    qwen = QwenSelector(API_KEY)
    local_idx = qwen.select_best(chosen_instruction, candidate_texts)
    if local_idx == -1:
        return 0, None

    global_idx = candidate_global_indices[local_idx]
    source_json, local_idx0 = index_map[global_idx]

    return local_idx0 + 1, source_json