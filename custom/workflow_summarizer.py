import json
import os
from pathlib import Path
import base64
import re
import time
import requests
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
API_URL = "http://localhost:5000/v1/chat/completions"
API_KEY = "sk-1234"
MODEL_NAME = "Qwen3-VL-8B-Instruct"
'''


# ================== 新增：本地 LLM 调用包装器 ==================
class TextLlmWrapper:
    RETRY_WAITING_SECONDS = 2
    MAX_RETRY = 3

    def __init__(self, api_key: str, model_name: str, api_url: str, check_way: str = "openai"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.check_way = check_way

    def predict(self, messages: List[dict], temperature: float = 0) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,  # workflow 处理可能需要较长输出
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
                    timeout=60
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
llm_wrapper = TextLlmWrapper(API_KEY, MODEL_NAME, API_URL, LLM_CHECK_WAY)

# ===== 坐标归一化 =====
def normalize_coordinate(x, y, width, height):
    rx = int(x / width * 1000)
    ry = int(y / height * 1000)
    return rx, ry


# ===== 读取record =====
def load_record(root):
    record_path = Path(root) / "record.json"

    with open(record_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = [x for x in data if "step_id" in x]

    return steps


def extract_query(root):
    record_path = Path(root) / "record.json"

    with open(record_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "query" in item:
            return item["query"]

    return ""


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def image_to_data_url(image_path):
    base64_str = encode_image_to_base64(image_path)
    return f"data:image/jpeg;base64,{base64_str}"


SYSTEM_PROMPT = f"""
你是一个GUI操作理解助手，需要根据动作和前后截图概括用户执行的操作。

要求：
1. 只描述用户的“操作动作”，不要推测用户意图
2. 只根据界面可见元素描述
3. 语言必须简洁，最多10个字
4. 不要出现任务目标或推测内容
5. 不要复述输入文本

动作定义：
done：任务完成
scroll：滑动
longclick：长按
click：点击
set_text：输入文本
open：打开应用
back：返回上一级
home：返回桌面
enter：确认/提交

请根据：

1 当前动作
2 动作执行前截图
3 动作执行后截图

输出一句简洁的动作描述。
所有点击坐标已经归一化到 0-1000 的屏幕坐标系，
(0,0) 为左上角，(1000,1000) 为右下角。

只输出一句话，例如：
点击搜索按钮
输入“老番茄”
打开哔哩哔哩
"""

# ===== 调用Agent (需要你接入自己的LLM) =====
def call_agent(model_input, before_img, after_img):
    before_data = image_to_data_url(before_img)
    after_data = image_to_data_url(after_img)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(model_input, ensure_ascii=False)
                },

                {
                    "type": "text",
                    "text": "Before Screenshot:"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": before_data}
                },

                {
                    "type": "text",
                    "text": "After Screenshot:"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": after_data}
                }
            ]
        }
    ]
    response_content = llm_wrapper.predict(messages, temperature=0)
    clean_content = response_content.strip()

    # 清理 ```json
    if clean_content.startswith("```"):
        clean_content = clean_content.replace("```json", "").replace("```", "")
    clean_content = clean_content.strip()

    return clean_content


def compress_workflow(root, summaries):
    full_history_path = Path(root) / "full_history.json"
    output_path = Path(root) / "workflow_compressed.json"

    with open(full_history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    step_index = 0

    for msg in history:
        if msg["role"] != "assistant":
            continue

        try:
            content_obj = json.loads(msg["content"])
        except:
            continue

        if "thought" in content_obj and step_index < len(summaries):

            # 用 summary 覆盖 thought
            content_obj["thought"] = summaries[step_index]["summary"]

            # 写回字符串
            msg["content"] = json.dumps(content_obj, ensure_ascii=False)

            step_index += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Compressed workflow saved to: {output_path}")


# ===== 主流程 =====
def summarize_workflow(root, phone_width, phone_height):
    steps = load_record(root)
    query = extract_query(root)

    results = []

    for i, step in enumerate(steps):

        action = step["action_seq"][0]
        action_type = action["type"]
        params = action["params"]

        # ===== 坐标归一化 =====
        if "points" in params:
            if "click" in action_type:
                x, y = params["points"]
                rx, ry = normalize_coordinate(x, y, phone_width, phone_height)

                params["points"] = [rx, ry]
            elif "scroll" in action_type:
                x, y,x2,y2 = params["points"]
                rx, ry = normalize_coordinate(x, y, phone_width, phone_height)
                rx2, ry2 = normalize_coordinate(x2, y2, phone_width, phone_height)
                params["points"] = [rx, ry, rx2, ry2]
                
        # ===== 图片路径 =====
        before_img = Path(root) / step["screenshot"]

        after_img = None
        if i + 1 < len(steps):
            after_img = Path(root) / steps[i + 1]["screenshot"]
        else:
            after_img = before_img

        model_input = {
            "task": query,
            "action": action
        }

        # ===== agent =====
        summary = call_agent(model_input, before_img, after_img)

        results.append({
            "step_id": step["step_id"],
            "action": action,
            "before_img": str(before_img),
            "after_img": str(after_img) if after_img else None,
            "summary": summary
        })

    compress_workflow(root, results)

    return results


# ===== main =====
if __name__ == "__main__":

    root = r"C:\Users\86159\Desktop\appagent_test\results\20260302_105431_在B站播放老番茄的最新一期视频_pass@3"

    phone_width = 1316
    phone_height = 2832

    summaries = summarize_workflow(root, phone_width, phone_height)

    with open("step_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)