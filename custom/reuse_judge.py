
from __future__ import annotations

import base64
import json
import re
import time
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

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
    RETRY_WAITING_SECONDS = 2
    MAX_RETRY = 3

    def __init__(self, api_key: str, model_name: str, api_url: str, check_way: str = "openai"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.check_way = check_way

    def predict(self, messages: List[dict], temperature: float = 0.0, max_tokens: int = 16) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
llm_wrapper = LlmWrapper(API_KEY, MODEL_NAME, API_URL, LLM_CHECK_WAY)


# -----------------------------
# Public function you can import
# -----------------------------

def can_reuse_action(
    pre_screenshot_path: str | Path,
    action: str,
    post_screenshot_path: str | Path,
    nx: str = "-1",
    ny: str = "-1"
) -> int:
    """
    Decide if an action can be reused/replayed directly.

    Args:
        pre_screenshot_path: file path to pre-action screenshot (png/jpg/webp)
        action: short action string (e.g., "tap 'Bluetooth' toggle", "swipe down", "drag slider to 70%")
        post_screenshot_path: file path to post-action screenshot

    Returns:
        0 or 1
    """
    '''
    override = _quick_rule_based_override(action)
    if override is not None:
        return override
    '''

    try:
        pre_b64 = _encode_image_to_base64(pre_screenshot_path)
        post_b64 = _encode_image_to_base64(post_screenshot_path)

        prompt = _build_prompt(action, nx, ny)

        # You can choose to pass images separately as "attachments" depending on your API.
        # Here we include them in a generic payload that you can adapt.
        payload = {
            "prompt": prompt,
            "inputs": {
                "action": action,
                "pre_screenshot_base64": pre_b64,
                "post_screenshot_base64": post_b64,
                "pre_screenshot_mime": _guess_mime(pre_screenshot_path),
                "post_screenshot_mime": _guess_mime(post_screenshot_path),
            },
        }

        raw = _call_llm_api(payload)
        print(_parse_binary_decision(raw))
        return 1
        #return _parse_binary_decision(raw)

    except Exception:
        # Non-conservative fallback: if anything goes wrong, ALLOW reuse.
        return 1


# -----------------------------
# Fast rule-based override
# -----------------------------
_VALUE_DRAG_KEYWORDS = [
    # CN
    "进度条", "滑块", "拖动", "拖拽", "轮盘", "滚轮", "滑杆", "调节", "调到", "调至", "拖到", "拖至",
    "时间轴", "音量条", "亮度条", "刻度", "缩放", "裁剪", "把手", "旋钮", "seek", "slider", "scrub",
    # Common apps
    "地图", "map", "zoom"
]
_SIMPLE_SCROLL_KEYWORDS = [
    "上滑", "下滑", "向上滑", "向下滑", "滑动一屏", "滚动一屏", "scroll up", "scroll down", "page down", "page up",
    "下拉", "上拉"
]


def _quick_rule_based_override(action: str) -> Optional[int]:
    """
    Returns:
        0 or 1 to override immediately, or None to fall back to LLM.

    We only hard-block obvious VALUE_DRAG gestures (case 2).
    Toggle (case 1) is NOT blocked here because you asked to require a visible switch in screenshots.
    """
    a = (action or "").lower()

    # If it's an explicit simple page scroll, allow reuse quickly.
    if any(k in a for k in _SIMPLE_SCROLL_KEYWORDS):
        return 1

    # Hard-block obvious value/drag gestures (progress bar / slider / wheel etc.)
    if any(k.lower() in a for k in _VALUE_DRAG_KEYWORDS):
        if any(k in a for k in _SIMPLE_SCROLL_KEYWORDS):
            return 1
        return 0

    return None

# -----------------------------
# Prompting
# -----------------------------


def _build_prompt(action: str, nx: str = "-1", ny: str = "-1") -> str:
    # Keep the model narrowly scoped. It must output ONLY "0" or "1".
    # Policy: ONLY output 0 when you are SURE it matches a disallowed category.
    return f"""你是一个遵循【输出约束】的移动端操作回放安全检查器。
你【不】规划操作。你【仅】决定给定的操作是否安全且可以直接回放。

==== 关键输出约束 ====
只能输出精确的一个字符，不要带有任何空格和换行：要么是0，要么是1。
绝对不要输出文字、标点符号、JSON 或任何解释说明。

含义：
0 = 回放不安全（交由 Agent 处理）
1 = 回放安全

决策策略：
只有当你确信该操作属于以下被禁用的类别时，才输出0。
如果你感到不确定，请输出1。

=== 禁用类别（输出0）：===
(1) 滑动/拖拽/滚动：任何涉及在屏幕上移动的手势（滚动、调整进度条、滑块、滚轮选择器、地图拖拽） -> 输出0。

=== 允许的类别（输出 1）===
除禁用类别外的所有其他操作都允许直接回放。

示例：
(1) action="点击‘矩阵分析’视频" -> 1
(2) action="点击列表中的第一个视频" -> 1
(3) action="点击‘设置’齿轮图标" -> 1
(4) action="点击蓝牙开关 (屏幕上可见蓝牙开关图标)" -> 1
(5) action="拖动进度条到 70%" -> 0
(6) action="向下滑动" -> 0 

你可用的输入信息：
- 操作字符串 (action string)
- 归一化到0-1000的坐标（当坐标为 -1 时，表示该操作既不是点击也不是长按。）
- 操作前的截图
- 操作后的截图

操作字符串：
{action}
点击或长按的坐标： 
({nx}, {ny})

如果发现截图缺失或者上述传入信息缺失，请直接判定为1（允许回放），不要输出0，因为我们不想因为环境问题误判为不安全。
"""

# -----------------------------
# Response parsing
# -----------------------------


_BINARY_RE = re.compile(r"[01]")


def _parse_binary_decision(raw: str | bytes | None) -> int:
    """
    Accepts typical LLM output and extracts the first 0/1.
    Any ambiguity => 1.
    """
    if raw is None:
        return 1
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", errors="ignore")
        except Exception:
            return 1

    s = raw.strip()

    # Common cases: exact "0" or "1"
    if s == "0":
        return 0
    if s == "1":
        return 1

    # Sometimes wrapped in JSON or other text; extract first digit.
    m = _BINARY_RE.search(s)
    if not m:
        return 1
    return 1 if m.group(0) == "1" else 0


# -----------------------------
# Image utilities
# -----------------------------

def _encode_image_to_base64(path: str | Path) -> str:
    """
    读取原图（不压缩）并转为 Base64
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Screenshot not found: {p}")
    data = p.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _guess_mime(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    # Default fallback
    return "application/octet-stream"


# -----------------------------
# API stub
# -----------------------------

def _call_llm_api(payload: dict) -> str:
    """
    使用本地 Wrapper 进行多模态调用
    """
    pre_b64 = payload["inputs"]["pre_screenshot_base64"]
    post_b64 = payload["inputs"]["post_screenshot_base64"]
    pre_mime = payload["inputs"].get("pre_screenshot_mime", "image/png")
    post_mime = payload["inputs"].get("post_screenshot_mime", "image/png")
    
    # DashScope's OpenAI-compatible interface accepts image_url items.
    # For local images, use a data URL.
    pre_data_url = f"data:{pre_mime};base64,{pre_b64}"
    post_data_url = f"data:{post_mime};base64,{post_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": pre_data_url}},
                {"type": "image_url", "image_url": {"url": post_data_url}},
                {"type": "text", "text": payload["prompt"]},
            ],
        }
    ]

    # 调用本地模型，temperature=0 保证判定稳定
    response = llm_wrapper.predict(messages, temperature=0.0, max_tokens=16)

    # 如果调用失败或返回空，默认返回 "1" (允许复用) 以保持保守策略，或者按需调整
    return response if response else "1"


# -----------------------------
# Optional: quick CLI smoke test
# -----------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Judge whether an action can be reused (0/1).")
    ap.add_argument("--pre", required=True,
                    help="Path to pre-action screenshot")
    ap.add_argument("--action", required=True, help="Action string")
    ap.add_argument("--post", required=True,
                    help="Path to post-action screenshot")
    args = ap.parse_args()

    print(can_reuse_action(args.pre, args.action, args.post))