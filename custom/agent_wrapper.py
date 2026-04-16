import abc
import base64
import io
import os
import time
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests
import json
from pathlib import Path
from jsonschema import Draft7Validator
import ast
import re
from jsonschema.exceptions import ValidationError
ERROR_CALLING_LLM = "Error calling LLM"
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

    print(
        f"Warning: {filename} not found in 'configs' folder of any parent directory.")
    return {}


_config = load_api_config()
API_URL = _config.get("agent_endpoints")
API_KEY = _config.get("agent_api_key")
MODEL_NAME = _config.get("agent_model")
CHECK_WAY = _config.get("agent_check_way")
'''
#这里还有可以手动设置的部分，以防出现问题无法载入。
API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "sk-1234"
MODEL_NAME = "Qwen3-VL-8B-Instruct"
'''
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)


def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)


ACTION_SCHEMA = json.load(
    open(os.path.join(current_dir, "new_schema.json"), encoding="utf-8")
)
items = list(ACTION_SCHEMA.items())
insert_index = 3  # 假设要插入到索引1的位置
items.insert(insert_index, ("required", ["thought"]))
# items.insert(insert_index, ("optional", ["thought"]))
ACTION_SCHEMA = dict(items)
SYSTEM_PROMPT = """你是一个精通鸿蒙系统操作的视觉辅助助手。
请根据【用户的指令】、【过去的历史信息】、【上一步骤反馈】和【当前屏幕截图】，输出下一步的最优操作。

=== 核心特权与铁律 (必须绝对服从) ===
1. 【纯净JSON格式】：必须以紧凑的 JSON 格式输出并严格遵循 Schema 约束！【严禁】在 JSON 块之外附加任何解释性文字。
2. 【单步单动作】：一次仅输出一步操作，每个 JSON 中只能存在 [open, point, type, press, status] 中的唯一一个动作。【严禁】同时输出多步操作！
3. 【应用瞬移特权】：需要打开任何应用时，必须且只能输出open命令。【严禁】先返回桌面，【严禁】通过点击图标打开！尤其在涉及多个应用的任务中，当上一个应用中的任务完成，【严禁】返回，直接用open命令打开下一个应用。
4. 【搜索绝对优先】：在“设置”或任何包含顶部搜索框的页面中，若目标选项未直接显示在当前屏幕，【严禁】输出滑动进行盲目翻找或点击看似相关项！【必须】优先点击搜索框，随后进行输入搜索。
5. 【文本框内容无效】：当你发现文本框（如搜索框，聊天框）在你【输入内容之前】就出现文本时，无视该文本，【严禁】将其误认为是有效内容，【只有】你用type指令输入的文本才是【唯一】有效的。
6. 【键盘限制】：当屏幕上出现键盘且需要输入文本时，【只能】使用type指令，【严禁】阅读键盘上的文本内容，【严禁】点击键盘上的任何按钮。
7. 【状态防反转铁律】：点赞/开关等状态切换步骤，若当前图标视觉状态已达成该步目标（如要求点赞且已变红/要求取消且已变灰），【严禁】再次点击！
8. 【死循环熔断】：若发现历史操作在同一页面反复执行相同动作未果，必须立即改变策略或使用"press: back"返回。
9. 【任务终结条件】：仅在明确判定任务已彻底完成时，才允许输出"status": "finish"，不可提前结束。

## 决策逻辑
1. 观察分析：识别截图中的按钮、图标、文本框及其相对位置。
2. 思考推理：结合用户指令，判断当前应进行点击、滑动、输入还是返回或是打开一个应用，用户的指令后可能还跟着上一步的执行后反馈，如果出现反馈则证明上一步未达到预期效果或是你陷入了死循环，需要结合反馈思考推理。
3. 坐标规范：所有坐标均使用 0-1000 之间的归一化数值。

## Schema
""" + json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':')) + """
## 输出示例
下面是合法的输出格式参考（注意：你每次回复只能输出一个JSON对象，严禁输出多个对象组成的列表）：
打开应用：{"thought": "当前页面中未出现音乐应用图标，调用系统命令打开音乐。", "open": "音乐"}
点击：{"thought": "当前在设置页面，未直接看到目标，页面上有“搜索设置项”按钮，点击搜索。", "point": [220, 240]}
长按：{"thought": "需要删除此任务，长按该条目弹出菜单。", "point": [300, 450], "duration": 1500}
滑动：{"thought": "当前页面出现时间选择，滑动选择具体时刻。", "point": [500, 800], "to": [500, 200]}
输入：{"thought": "点击搜索框后，发现文本框中已有内容，无视该文本，直接输入。", "type": "纵此生"}
返回：{"thought": "任务进入死循环，执行返回。", "press": "back"}
回到桌面：{"thought": "任务进入错误页面，执行返回。", "press": "home"}
任务完成：{"thought": "视频已点赞，无须再次点击点赞按钮，任务结束。", "status": "finish"}
  """

EXTRACT_SCHEMA = json.load(
    open(os.path.join(current_dir, "new_extraction.json"), encoding="utf-8")
)
validator = Draft7Validator(EXTRACT_SCHEMA)


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Converts a numpy array into a byte string for a JPEG image."""
    image = Image.fromarray(image)
    return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format="PNG")
    # Reset file pointer to start
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    return img_bytes


class LlmWrapper(abc.ABC):
    """Abstract interface for (text only) LLM."""

    @abc.abstractmethod
    def predict(
            self,
            text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.

        Returns:
          Text output, is_safe, and raw output.
        """


class MultimodalLlmWrapper(abc.ABC):
    """Abstract interface for Multimodal LLM."""

    @abc.abstractmethod
    def predict_mm(
            self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.
          images: List of images as numpy ndarray.

        Returns:
          Text output and raw output.
        """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (types.HarmBlockThreshold.BLOCK_NONE),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (types.HarmBlockThreshold.BLOCK_NONE),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class Qwen3AgentWrapper(MultimodalLlmWrapper):
    RETRY_WAITING_SECONDS = 20

    def __init__(
            self,
            max_retry: int = 3,
            temperature: float = 0.3,
            use_history: bool = False,
            history_size: int = 10,
            api_key: str = "sk-1234",
            model_name: str = "Qwen3-VL-8B-Instruct"
    ):
        # self.model = "Qwen3-VL-8B-Instruct"
        self.api_key = API_KEY
        self.model = MODEL_NAME
        self.check_way = CHECK_WAY
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        self.use_history = use_history
        self.history_size = max(history_size, 1)
        self.history: list[dict] = []
        self.full_history: list[dict] = []
        self.raw_history: list[dict] = []  # 真正完整history（含图片）
        self.open_settings = False

    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        return base64.b64encode(array_to_jpeg_bytes(image)).decode("utf-8")

    def _push_history(self, role: str, content: Any):
        self.raw_history.append({
            "role": role,
            "content": content
        })

        # ===== 清理 user 历史中的图片：直接删掉 =====
        if role == "user" and isinstance(content, list):
            # 过滤掉所有 type 为 image_url 的项，保留其他项
            content = [
                item for item in content 
                if not (isinstance(item, dict) and item.get("type") == "image_url")
            ]

        # ===== 原逻辑 =====
        self.full_history.append({"role": role, "content": content})
        if not self.use_history:
            return

        self.history.append({"role": role, "content": content})

        max_msgs = self.history_size * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def clear_history(self):
        self.history.clear()

    def extract_and_validate_json(self, input_string):
    # 1. 暴力提取包裹在花括号内的内容 (无视它用的是单引号还是双引号)
        json_match = re.search(r'\{.*\}', input_string, re.DOTALL)
        if json_match:
            clean_str = json_match.group(0)
        else:
            print("[Error] 无法在输出中找到合法的 {} 块")
            return {"error": "no_json_found", "raw_output": input_string}

        json_obj = None

        # 2. 双重解析引擎
        # 引擎 A：先尝试用最严格的 JSON 标准解析
        try:
            json_obj = json.loads(clean_str)
        except json.JSONDecodeError:
            # 引擎 B：一旦因为单双引号混用报错，立刻切换到 Python 字典解析器！
            try:
                json_obj = ast.literal_eval(clean_str)
                if not isinstance(json_obj, dict):
                    raise ValueError("解析出来的不是字典")
                print("[Info] 触发 ast 解析成功兜底了单双引号混用！")
            except (ValueError, SyntaxError) as e:
                print(f"[Error] 格式彻底损坏，JSON和AST解析双双阵亡: {e}")
                return {"error": "decode_error", "raw_output": input_string}

        # 3. 并发动作拦截 (拦截大模型同时输出多个动作的幻觉)
        try:
            action_keys = ["open", "point", "type", "press", "status"]
            found_actions = [k for k in action_keys if k in json_obj]

            if len(found_actions) == 0:
                print("[Error] 缺失动作字段！")
                return {"error": "missing_action", "raw_output": input_string}

            # 如果大模型输出了多个动作，尝试保留文本中出现的第一个动作（尽量按模型原始输出顺序）
            if len(found_actions) > 1:
                # 在原始提取字符串中查找各动作键第一次出现的位置，选择最先出现的动作
                first_action = None
                first_pos = None
                for k in found_actions:
                    # 匹配带引号或不带引号的键名后跟冒号的形式
                    m = re.search(r'["\']?%s["\']?\s*:' % re.escape(k), clean_str)
                    if m:
                        pos = m.start()
                        if first_pos is None or pos < first_pos:
                            first_pos = pos
                            first_action = k

                if first_action is None:
                    # 兜底：若无法从原始字符串判定，按 action_keys 的优先顺序取第一个
                    first_action = found_actions[0]

                print(f"[Warn] 检测到多个动作 {found_actions}，保留第一个动作: {first_action}")

                # 构造新的 json_obj，仅保留第一个动作键及其他非动作字段（如 thought）
                filtered = {}
                for kk, vv in json_obj.items():
                    if kk in action_keys:
                        continue
                    filtered[kk] = vv

                # 将第一个动作加入
                filtered[first_action] = json_obj.get(first_action)
                json_obj = filtered

            # 4. 最后一步：交给你的 new_extraction.json 进行数据类型和范围的验证
            validator.validate(json_obj, EXTRACT_SCHEMA)
            return json_obj

        except ValidationError as e:
            print(f"[Error] 字段内容验证失败: {e.message}")
            return {"error": "schema_invalid", "raw_output": input_string}
        except Exception as e:
            print(f"[Error] 未知错误: {e}")
            return {"error": "unknown_error", "raw_output": input_string}

    def predict_mm(
            self,
            text_prompt: str,
            images: str,  # base64
            all_std_apps_str: str = "[]",
            probable_apps_str: str = "[]",
            final_check: bool = False,
    ) -> tuple[str, Optional[bool], Any]:

        if all_std_apps_str and all_std_apps_str != "[]":
            app_constraint = f"""
## 应用名输出约束
在执行open动作时，应用名必须严格遵守以下规则：
1.优先关注：当前任务极有可能涉及这些应用：{probable_apps_str}。请优先从中选择。
2.允许扩展：如果任务确实需要其他应用，请从下方的【全量标准库】中查找。
3.强制标准：无论打开什么应用，输出的名称【必须】逐字匹配【全量标准库】中的名称，【严禁使用别名或造词】。
【全量标准库】：{all_std_apps_str}
"""
            full_system_prompt = SYSTEM_PROMPT + "\n" + app_constraint
        else:
            full_system_prompt = SYSTEM_PROMPT

        # ===== 1. system prompt（与 MiniCPMWrapper 完全一致）=====
        messages: list[dict] = [
            {
                "role": "system",
                "content": full_system_prompt
            }
        ]

        # ===== 2. history =====
        if self.use_history and self.history:
            messages.extend(self.history)

        # ===== 3. 当前 user =====
        user_content = [
            {
                "type": "text",
                "text": f"<Question>{text_prompt}</Question>\n当前屏幕截图：(<image>./</image>)",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{images}"
                },
            },
        ]
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model,  # 发送正确的模型名称
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 2048,
            "stream": False  # 显式声明非流式，匹配 qwen.py 的逻辑
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

        # headers = {"Content-Type": "application/json"}
        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS

        if final_check:
            assistant_text = "{\"thought\": \"任务已完成。\", \"status\": \"finish\"}"
            action = self.extract_and_validate_json(assistant_text)
            print("settings Extracted action:", action)
            self._push_history("user", user_content)
            self._push_history("assistant", assistant_text)
            return assistant_text, None, None, action

        while counter > 0:
            try:
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json=payload,
                )

                if response.ok and "choices" in response.json():
                    assistant_msg = response.json()["choices"][0]["message"]
                    assistant_text = assistant_msg["content"]
                    print(assistant_text)
                    action = self.extract_and_validate_json(assistant_text)
                    print("Extracted action:", action)
                    if not self.open_settings:
                        if "open" in action and action["open"] == "设置":
                            self.open_settings = True
                    else:
                        if "未" in action["thought"] or "滑" in action["thought"]:
                            assistant_text = "{\"thought\": \"当前在设置页面，应优先使用搜索框输入关键词进行查找，所以点击搜索框。\", \"point\": [220, 240]}"
                            action = self.extract_and_validate_json(assistant_text)
                            print("settings Extracted action:", action)
                        self.open_settings = False

                    # ===== 4. 写历史 =====
                    self._push_history("user", user_content)
                    self._push_history("assistant", assistant_text)

                    return assistant_text, None, response, action

                print("Error:", response.text)
                time.sleep(wait_seconds)
                wait_seconds *= 2

            except Exception as e:
                counter -= 1
                time.sleep(wait_seconds)
                wait_seconds *= 2
                print("Error calling Qwen3 agent:", e)

        return ERROR_CALLING_LLM, None, None

    def save_full_history_to_json(self, filepath: str = "full_history.json"):
        """
        将完整历史 full_history 保存为 JSON 文件
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.full_history, f, ensure_ascii=False, indent=2)

        print(f"Full history saved to {filepath}")


if __name__ == "__main__":
    # ========== 1. 初始化 Agent ==========
    agent = Qwen3AgentWrapper(
        temperature=1.0,
        use_history=True,
        history_size=2,
    )

    # ========== 2. 读取并编码图片 ==========
    image_path = "test_screen.jpeg"  # ← 改成你自己的截图路径
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_base64 = agent.encode_image(image_np)

    # ========== 3. 第一次调用 ==========
    question1 = "打开设置,然后再打开小红书，最后打开bilibili"

    print("\n===== Call 1 =====")
    response = agent.predict_mm(
        text_prompt=question1,
        images=image_base64,
    )
    action = response[3]

    print("Assistant:", action)
    thought_content = ""

    # 1. 情况一：如果大模型封装类已经帮我们把它解析成了 Python 字典 (Dictionary)
    if isinstance(action, dict):
        thought_content = action.get('thought', '没有找到思考过程')

    # 2. 情况二：如果它是一串长得像字典的字符串 (String)
    elif isinstance(action, str):
        try:
        # 优先尝试用标准 JSON 解析 (要求必须是双引号)
            action_dict = json.loads(action)
            thought_content = action_dict.get('thought', '没有找到思考过程')
        except json.JSONDecodeError:
            try:
                # 如果大模型不听话用了单引号 (像你上面发的例子就是单引号)
                # ast.literal_eval 可以安全地把字符串 "{'a': 1}" 变成真正的字典
                action_dict = ast.literal_eval(action)
                thought_content = action_dict.get('thought', '没有找到思考过程')
            except (ValueError, SyntaxError):
                print("[Warning] 无法解析动作格式")

    print(f"✅ 成功提取的思考过程是: {thought_content}")

    # ========== 6. 保存完整历史 ==========
    save_path = "qwen3_full_history.json"
    agent.save_full_history_to_json(save_path)

    print("\n===== Test Finished =====")
