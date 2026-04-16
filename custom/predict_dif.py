import time
import json
import os
import requests
from pathlib import Path
from typing import List, Optional

# ================== 本地 API 配置区 ==================
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


# ================== 新增：本地 LLM 调用包装器 ==================
class LlmWrapper:
    """
    Wrapper for local Qwen-VL API calls with retry logic.
    """
    RETRY_WAITING_SECONDS = 2
    MAX_RETRY = 3

    def __init__(self, api_key: str, model_name: str, api_url: str, check_way: str = "openai"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.check_way = check_way

    def predict(self, messages: List[dict], temperature: float = 0.1) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 512,
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
                    timeout=30
                )

                if response.ok:
                    data = response.json()
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
llm_wrapper = LlmWrapper(API_KEY, MODEL_NAME, API_URL, LLM_CHECK_WAY)


def predict_dif(base_workflow, base_query, new_query, apikey):
    """
    比较工作流差异，返回不同步骤编号列表。

    参数：
        base_workflow: list[str]，基准工作流
        base_query: str，基准指令
        new_query: str，新指令
        apikey: str，API Key

    返回：
        list[int]，new_workflow 与 base_workflow 的不同步骤编号
    """

    # System Prompt
    messages = [
        {
            "role": "system",
            "content": """你是一个用于分析工作流差异的助手。

给定：
- 一个基准工作流（base_workflow），包含按顺序执行的步骤；
- 一个基准指令（base_query）；
- 一个新的指令（new_query）；

你的任务是：
1. 根据新的指令，在基准工作流（base_workflow）中寻找可复用步骤；
2. 输出除此之外的不可复用步骤的编号（从1开始）；
3. 如果所有步骤都可复用，输出空列表：[]；
4. 注意，可能会出现不可复用步骤处于工作流中间的情况，需要正确识别复用步骤的范围，也就是只需要选择出确实无法复用的那些步骤即可；
5. 有些步骤天然不可复用，需要你区分“语义内容”与“结构索引”：即点击具体标题（如：点击矩阵分析视频）这种不可复用，而点击固定索引（如：点击第一个.....）可复用；
6. 仅输出一个 Python 列表，例如：[6,7]；
7. 不要输出多余解释、描述或文字。

示例：
输入：
base_workflow = [
    "打开美团", "点击搜索框", "输入星巴克","点击星巴克商家",
    "搜索冰美式", "下单"
]
base_query = "帮我点一杯星巴克的冰美式"
new_query = "帮我点一杯星巴克的拿铁"
输出：
[4,5]

输入：
base_workflow = ["进入个人中心", "查看我的收藏", "播放第一条收藏视频"]
base_query = "在B站播放我的收藏中的第一条视频"
new_query = "在B站播放我的收藏中的第二条视频"
输出：
[3]

"""
        }
    ]

    # 用户输入
    user_input = f"""
base_workflow = {base_workflow}
base_query = "{base_query}"
new_query = "{new_query}"
"""
    messages.append({"role": "user", "content": user_input})

    # 调用模型
    start_time = time.time()
    
    # 替换 Generation.call 为本地调用
    output_text = llm_wrapper.predict(messages, temperature=0.1)
    
    elapsed = time.time() - start_time
    print(f"请求耗时：{elapsed:.2f} 秒")

    if output_text:
        output_text = output_text.strip()
        # 尝试直接 eval 成 Python list
        try:
            return eval(output_text)
        except Exception:
            # 如果解析失败，返回原始文本
            return output_text
    else:
        raise RuntimeError(f"请求出错: 本地模型无响应")


def get_base_query_by_index(index, file_path=None):
    """
    从历史查询文件中读取所有句子，根据序号提取 base_query。

    支持 JSON 格式:(querys.json): [ { "task_id": ..., "query": ..., ... }, ... ]

    如果未传入 `file_path`，默认使用项目中的新格式路径：
    'newformat_data/com.huawei.hmsapp.music/tasks/querys.json'
    """
    # 默认路径（相对工作目录）
    if file_path is None:
        file_path = os.path.join(
            "newformat_data", "com.huawei.hmsapp.music", "tasks", "querys.json"
        )

    # 读取 JSON 文件
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    for i, item in enumerate(data):
        if isinstance(item, dict) and "query" in item:
            sentences.append(item["query"])
        else:
            raise ValueError(f"{file_path} 格式错误：第 {i} 项缺少 'query' 字段或不是对象")

    if not (1 <= index <= len(sentences)):
        raise ValueError(f"序号 {index} 超出范围 1~{len(sentences)}")

    # 提取指定句子
    return sentences[index - 1]


def load_base_workflow_by_index(index, folder_path="database/tasks"):
    """
    根据序号读取任务目录下的子文件夹（例如 tasks/{index}），
    每个步骤保存在单独的 JSON 文件（例如 1.json, 2.json ...），
    从每个步骤文件中提取 `thought` 作为 base_workflow 的一项，
    并同时收集对应的 `action_id` 到 action_list 中。

    返回：(base_workflow: List[str], action_list: List[Optional[int]])
    文件按文件名（去掉扩展名后）尝试按整数排序，以保证步骤顺序。
    """
    folder_path = os.path.join(folder_path, str(index))
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"{folder_path} 不存在")

    # 列出该子文件夹下所有 .json 文件
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]

    if not files:
        raise FileNotFoundError(f"在 {folder_path} 中未找到任何 .json 步骤文件")

    # 尝试按文件名（无扩展名）按整数排序，非整数名保持原顺序
    def _sort_key(fn):
        name = os.path.splitext(fn)[0]
        try:
            return int(name)
        except Exception:
            return name

    files.sort(key=_sort_key)

    base_workflow = []
    action_list = []
    app_list = []

    for fname in files:
        file_path = os.path.join(folder_path, fname)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            # 无法读取或解析该步骤文件则跳过
            continue

        # 每个步骤文件期望为 dict，包含 thought 和 action_id
        if isinstance(data, dict):
            thought = data.get('thought')
            action_id = data.get('action_id')
            app_id = data.get('app_id')
            # 保持顺序，即使 thought 缺失也放入空字符串以占位
            base_workflow.append(thought if thought is not None else "")
            action_list.append(action_id if action_id is not None else None)
            app_list.append(app_id if app_id is not None else None)

    return base_workflow, action_list, app_list


# ====== 使用示例 ======
if __name__ == "__main__":
    # index = int(input("请输入句子序号（1-10）："))
    index = 1
    base_query = get_base_query_by_index(
        index, "newformat_data/yylx.danmaku.bili/tasks/querys.json")
    
    # 注意：load_base_workflow_by_index 返回了 3 个值，这里需要根据您的实际逻辑进行接收
    # 原代码是: base_workflow, action_list = ... 
    # 但根据您提供的函数定义，它返回 (base_workflow, action_list, app_list)
    # 建议修改为：
    base_workflow, action_list, app_list = load_base_workflow_by_index(index)
    
    print("选中的 base_query:", base_query)
    print("base_workflow:", base_workflow)
    print("action_list:", action_list)
    
    apikey = "sk-70d4e7e320a740a1862c10a4e7715d71"
    new_query = "用微信给周喆宇发消息 今天晚上一起讨论吧"
    
    diff_steps = predict_dif(base_workflow, base_query, new_query, apikey)
    print("差异步骤:", diff_steps)