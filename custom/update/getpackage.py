import json
import base64
import os
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, Set, Optional, Tuple

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

# ================== LlmWrapper (复用标准组件) ==================
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

    def predict(self, messages: List[dict], temperature: float = 0.01) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 128, # 包名判断不需要长输出
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


# ================== AppResolver 类 ==================
class AppResolver:
    def __init__(self, config_dir: Path):
        self.config_path = config_dir / "app_package_config.json"
        # 1. 加载映射表： pkg -> name
        self.pkg_to_name = self._load_config()
        # 2. 生成反向映射表： name -> pkg (用于模型返回结果后找回包名)
        self.name_to_pkg = {v: k for k, v in self.pkg_to_name.items()}

        # 初始化本地 LLM Wrapper
        self.llm = LlmWrapper(API_KEY, MODEL_NAME, API_URL, LLM_CHECK_WAY)

    def _load_config(self) -> Dict[str, str]:
        if not self.config_path.exists():
            return {}
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _extract_bundle_names(self, ui_tree: Any) -> Set[str]:
        """
        [暴力版] 递归遍历 UI 树中所有的 bundleName。
        不依赖特定的结构（如 'children'），只要 JSON 里有 bundleName 字段就能挖出来。
        """
        bundles = set()

        def _traverse(obj):
            if isinstance(obj, dict):
                # 1. 检查当前字典是否有目标字段
                if "bundleName" in obj:
                    bn = obj["bundleName"]
                    if isinstance(bn, str) and bn.strip():
                        bundles.add(bn)

                # 2. 【关键修改】不要只看 "children"，遍历所有 value 继续深挖
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        _traverse(value)

            elif isinstance(obj, list):
                # 如果是列表，遍历列表里的每个元素
                for item in obj:
                    _traverse(item)

        _traverse(ui_tree)
        return bundles

    def _encode_image(self, img_path: Path) -> str:
        if not img_path.exists():
            return ""
        data = img_path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        # 返回完整的数据 URI，方便直接塞入 messages
        return f"data:image/jpeg;base64,{b64}"

    def _call_llm_judge(self, screen_path: Path, thought: str, candidate_names: List[str]) -> str:
        """
        模型层：只处理自然语言（App名字）。
        输入：截图、意图、['淘宝', '微信', '系统桌面']
        输出：'淘宝'
        """
        img_data_url = self._encode_image(screen_path)
        if not img_data_url:
            return "system"

        # 构造选项字符串
        options_str = "、".join([f"【{name}】" for name in candidate_names])
        print(f"[Info] LLM Judge Options: {options_str}")
        
        prompt = f"""
你是一个手机操作意图识别助手。
当前用户正在执行一步操作，意图是：“{thought}”。
当前画面中检测到以下应用：
{options_str}
请根据截图和意图，判断执行该操作时主要在处于哪个应用内。
规则：
1. 请直接输出应用名称（必须从上述选项中选择一个）。
2. 如果当前状态不在任何一个应用内，请输出“系统/未知”。
3. 不要输出任何JSON或额外解释，只输出应用名。

举例：
当前动作为“在搜索框输入李子柒”，假设你的接收到的输入“哔哩哔哩、小红书”，那么如果你判断出此时处于哔哩哔哩，此时应该返回哔哩哔哩
"""
        try:
            # 构造多模态消息
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_data_url}},
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
            
            # 调用本地 LLM
            content = self.llm.predict(messages, temperature=0.01)
            
            if not content:
                return "系统/未知"

            # 清洗输出，防止模型带标点或Markdown
            result = content.strip().replace("【", "").replace("】", "")
            return result

        except Exception as e:
            print(f"[Error] LLM Judge failed: {e}")
            return "系统/未知"

    def resolve_package(self, ui_tree: Dict[str, Any], screen_path: Path, thought: str) -> str:
        """
        主逻辑：Code -> Model -> Code
        【严格模式】：只有在 app_package_config.json 里定义的包名才会被考虑。
        """
        # 1. [Code] 暴力提取所有包名
        raw_bundles = self._extract_bundle_names(ui_tree)

        # 2. [Code] 严格过滤：构建候选列表
        candidate_map = {}  # 映射：应用名 -> 包名
        candidate_names = []  # 列表：['淘宝', '微信']

        for pkg in raw_bundles:
            # 【核心修改】严格检查：必须在配置表中存在，才算有效候选
            if pkg in self.pkg_to_name:
                app_name = self.pkg_to_name[pkg]
                # 去重：防止同一个应用出现多次
                if app_name not in candidate_map:
                    candidate_map[app_name] = pkg
                    candidate_names.append(app_name)
            else:
                # 调试日志：可以看到哪些包名被丢弃了（通常是系统组件或未配置的App）
                # print(f"DEBUG: Ignored unknown package: {pkg}")
                pass

        # 3. 如果过滤后没有一个认识的 App，直接返回 system
        if not candidate_names:
            print("[Info] No known packages found in UI tree, defaulting to system.")
            return "system"
        
        print(f"[Info] Candidate apps: {candidate_names}")
        
        # 4. 如果只剩下一个认识的 App，且没有其他干扰项？
        #    为了严谨，这里还是建议问一下 LLM，因为“在淘宝界面点击状态栏”也是可能的。
        #    添加系统兜底选项
        candidate_names.append("系统/未知")

        # 5. [Model] 让模型选名字
        chosen_name = self._call_llm_judge(screen_path, thought, candidate_names)
        print(f"[Info] LLM chose: {chosen_name}")

        # 6. [Code] 应用名 -> 包名 (后处理)
        if chosen_name in candidate_map:
            return candidate_map[chosen_name]

        # 如果模型选了“系统/未知”或者其他的
        return "system"