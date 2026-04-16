import json
import re
import time
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
class TextLlmWrapper:
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
                    timeout=300
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


SYSTEM_PROMPT = """你是一个“动作主导型意图重建器”（Action-Dominant Intent Reconstructor）。

你将处理一个 GUI-Agent workflow。为了防止语义干扰动作判定，输入的每一个 step 数据已经被拆分为两个核心部分：
1. `action_fields`: 真实的 GUI 物理动作字段（绝对真理）。
2. `original_thought`: 原始的上下文描述（仅供参考名词目标，其包含的动作动词极可能是错误的）。

────────────────────────────────
【全局任务目标（Question）】
- 全局任务目标由 user 消息中的 <Question>...</Question> 提供。
- 所有 short-thought 必须服务于全局目标。

────────────────────────────────
【你的任务：严格的两步推导法】

在生成每一个 short-thought 时，你必须在脑内严格执行以下两步：

👉 第一步：锁定动作类型（无视 original_thought）
只能基于 `action_fields` 包含的键来唯一确定动作类型：
- 【仅】包含 `point` → 动作类型 = 点击
- 包含 `point` 和 `duration` → 动作类型 = 长按
- 包含 `point` 和 `to` → 动作类型 = 滑动
- 包含 `to` → 动作类型 = 滑动
- 包含 `type` → 动作类型 = 输入
- 包含 `open` → 动作类型 = 打开应用
- 包含 `press` → 动作类型 = 系统按键
- 包含 `status = finish` → 动作类型 = 任务完成

👉 第二步：提取目标并选择合法意图（结合 original_thought）
- 从 `original_thought` 中提取操作对象（如：防误触模式、搜索框、设置）。
- **绝对禁止**使用 `original_thought` 中的动作动词（如：把点击误认为滑动）。
- 将提取出的操作对象，套入第一步锁定的动作类型对应的【允许意图集合】中。

────────────────────────────────
【允许意图集合（白名单，强制遵守）】

────────
1️⃣ 点击（point）
只允许：打开某应用 / 进入某界面 / 打开某功能入口 / 选中某对象 / 确认某操作 / 执行搜索 / 播放某内容 / 同意用户协议
❌ 致命错误警告：只要是 point，即使 original_thought 说了“滑动寻找”、“向下浏览”，你也绝对不能输出包含“滑动”、“查找”等词汇！必须修正为类似“点击某对象”的格式。

────────
2️⃣ 打开应用（open）
只允许：打开某应用 / 打开某系统功能

────────
3️⃣ 滑动（to）
只允许：向上滑动 / 向下滑动 / 向左滑动 / 向右滑动 / 滑动到某区域

────────
4️⃣ 输入（type）
只允许：输入搜索内容 / 输入文本

────────
5️⃣ 系统按键（press）
press=back → 返回上一级；press=home → 回到桌面；press=enter → 确认操作 或 提交搜索

────────
6️⃣ status = finish
只允许：完成任务

────────────────────────────────
【short-thought 书写规范】
- 中文，4–10 字。

────────────────────────────────
【输出格式（严格）】
只输出 JSON 数组，长度必须与输入的 step_count 完全一致，禁止合并或跳过：
[
  {"thought": "<第1个 step 的 short-thought>"},
  {"thought": "<第2个 step 的 short-thought>"}
]
"""


def extract_question(workflow):
    """
    从 user message 中提取 <Question>...</Question>
    """
    for msg in workflow:
        if msg["role"] == "user":
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    match = re.search(r"<Question>(.*?)</Question>", text)
                    if match:
                        return match.group(1).strip()
    return ""


def compress_workflow(input_path, output_path) -> bool:
    with open(input_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # 1. 提取 question
    question = extract_question(workflow)

    # 2. 收集 assistant steps
    # 2. 收集 assistant steps
    steps = []
    for msg in workflow:
        if msg["role"] == "assistant":
            try:
                step_obj = json.loads(msg["content"])
                
                # 核心修改：分离“物理动作”与“语义参考”
                # 把 thought 提取出来作为单独的参考字段，剩下的全部作为 action_fields
                original_thought = step_obj.pop("thought", "")
                
                refined_step = {
                    "action_fields": step_obj,  # 现在这里面只剩 point, to, type 等纯物理动作
                    "original_thought": original_thought # 降权为参考文本
                }
                steps.append(refined_step)
                
            except Exception:
                steps.append({"action_fields": {"status": "error"}, "original_thought": msg["content"]})

    print("Assistant Steps:")
    print(steps)

    model_input = {
        "question": question,
        "step_count": len(steps),
        "steps": steps
    }

    assistant_step_count = len(steps)

    MAX_FIX_RETRY = 5

    # 初始化 messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(model_input, ensure_ascii=False)
        }
    ]

    retry_count = 0

    while retry_count < MAX_FIX_RETRY:

        print(f"\n===== LLM Try {retry_count + 1} =====")

        response_content = llm_wrapper.predict(messages, temperature=0.1)

        if not response_content:
            raise RuntimeError("Model returned no content.")

        print("\nLLM Raw Output:")
        print(response_content)

        # ---------- 清理 markdown ----------
        clean_content = response_content.strip()

        if clean_content.startswith("```"):
            clean_content = clean_content.replace("```json", "").replace("```", "")

        clean_content = clean_content.strip()

        # ---------- JSON 解析 ----------
        try:
            compressed = json.loads(clean_content)
        except Exception as e:
            print("JSON parse error:", e)
            compressed = None

        # ---------- step 校验 ----------
        if isinstance(compressed, list) and len(compressed) == assistant_step_count:
            print("✅ LLM output valid")
            break

        # ---------- 输出错误 ----------
        retry_count += 1

        print("❌ Step count mismatch")

        error_feedback = f"""
你的输出是错误的。

错误原因：
输出 step 数量 = {0 if compressed is None else len(compressed)}
期望 step 数量 = {assistant_step_count}

你的输出：
{clean_content}

请重新生成，并严格遵守：

1. JSON数组长度必须等于 期望 step 数量
2. 每个 step 必须对应一个 assistant step
3. 顺序必须完全一致
4. 不允许新增或删除 step
5. 只输出 JSON

重新生成。
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(model_input, ensure_ascii=False)
            },
            {
                "role": "user",
                "content": error_feedback
            }
        ]

    else:
        raise RuntimeError("LLM failed to generate valid output after retries.")

    print("\nFinal compressed result:")
    print(compressed)

    # 4. 写回 short-thought
    idx = 0
    for msg in workflow:
        if msg["role"] == "assistant":
            step_json = json.loads(msg["content"])
            step_json["thought"] = compressed[idx]["thought"]
            msg["content"] = json.dumps(step_json, ensure_ascii=False)
            idx += 1

    # 5. 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)

    print(f"\nSaved compressed workflow → {output_path}")

    return True


if __name__ == "__main__":
    compress_workflow("../full_history.json", "workflow_compressed.json")