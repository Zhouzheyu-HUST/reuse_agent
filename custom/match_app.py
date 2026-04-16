# -*- coding: utf-8 -*-
"""
根据用户指令匹配最合适应用名 + 应用名映射包名的工具函数
"""

import json
import os
import time
import requests
from typing import List, Optional
from pathlib import Path

# -------------------------- 配置项 --------------------------
# 1. 本地 LLM 配置


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
API_URL = _config.get("llm_endpoints")
API_KEY = _config.get("llm_api_key")
LLM_MODEL = _config.get("llm_model")
LLM_CHECK_WAY = _config.get("llm_check_way")

'''
#这里还有可以手动设置的部分，以防出现问题无法载入。
API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "sk-1234"
MODEL_NAME = "Qwen3-VL-8B-Instruct"
'''

# 2. 可选应用列表（可根据需求增删）
APP_LIST = [
    "音乐",
    "视频",
    "设置",
    "运动健康",
    "应用市场",
    "华为商城",
    "时钟",
    "58同城",
    "铁路12306",
    "航旅纵横",
    "中国国航",
    "淘宝",
    "闲鱼",
    "百度",
    "中国移动",
    "高德",
    "滴滴出行",
    "小红书",
    "微信",
    "QQ",
    "哔哩哔哩",
    "酷狗音乐",
    "喜马拉雅",
    "QQ音乐",
    "微博",
    "同花顺",
    "快手",
    "新浪新闻",
    "七猫免费小说",
    "今日头条",
    "番茄免费小说",
    "中国建设银行",
    "去哪儿旅行",
    "同程旅行",
    "美图秀秀",
    "央广网",
    "唯品会",
    "宝宝巴士",
    "UC浏览器",
    "懂车帝",
    "芒果TV",
    "知乎",
    "WPS Office",
    "爱奇艺",
    "优酷视频",
    "央视影音",
    "中国工商银行",
    "中国农业银行",
    "美团",
    "支付宝",
    "大众点评",
    "京东",
    "作业帮",
    "抖音",
    "携程旅行",
    "钉钉",
    "图库",
    "电话",
    "主题",
    "阅读",
    "钱包",
    "游戏中心",
    "智慧生活",
    "畅联",
    "日历",
    "备忘录",
    "我的华为",
    "小游戏",
    "天际通",
    "电子邮件",
    "玩机技巧",
    "地图",
    "天气",
    "计算器",
    "文件管理",
    "录音机",
    "查找设备",
    "信息",
    "浏览器",
    "相机",
    "智能遥控",
    "指南针"
]

# 3. JSON文件路径配置（请替换为你的实际JSON文件路径）
APP_PACKAGE_JSON_PATH = "configs/app_package_config.json"

# 尝试从配置文件动态加载 APP_LIST（若加载失败则保留上面定义的默认列表）


def load_app_list_from_json(path: str = APP_PACKAGE_JSON_PATH, fallback: Optional[list] = None) -> None:
    """
    尝试从给定 JSON 文件读取包名->应用名映射，并将唯一的应用名列表写回全局 APP_LIST。
    若读取或解析失败，则保持原有 APP_LIST（或使用 fallback 回退）。
    """
    global APP_LIST
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        apps = []
        seen = set()
        # 保持 JSON 中出现顺序，去重
        for v in raw.values():
            if isinstance(v, str) and v not in seen:
                apps.append(v)
                seen.add(v)

        if apps:
            APP_LIST = apps
    except Exception as e:
        print(f"Warning: 加载 APP_LIST 失败：{e}")
        if fallback is not None:
            APP_LIST = fallback


# 保存当前默认列表以便回退，然后尝试替换
DEFAULT_APP_LIST = APP_LIST.copy()
load_app_list_from_json(fallback=DEFAULT_APP_LIST)

# -------------------------- 全局缓存（避免重复读取JSON文件） --------------------------
_app_package_mapping = None  # 缓存应用名->包名的映射


# -------------------------- 新增：本地 LLM 调用包装器 --------------------------
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

    def predict(self, messages: List[dict], temperature: float = 0.0) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
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
                    timeout=20
                )

                if response.ok:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"]

                print(
                    f"Warning: LLM call failed with status {response.status_code}")
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1

            except Exception as e:
                print(f"Warning: Exception calling LLM: {e}")
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1

        return None


# 初始化全局 Wrapper
llm_wrapper = LlmWrapper(API_KEY, LLM_MODEL, API_URL, LLM_CHECK_WAY)


# -------------------------- 核心函数：匹配应用名 --------------------------

def get_matched_app(query: str) -> list[str]:
    """
    根据用户输入的指令，调用大语言模型返回一个应用名列表（list[str]）。

    示例：
    - 输入："打开QQ音乐播放薛之谦的刚刚好" → 返回：["QQ音乐"]
    - 输入："打开音乐帮我搜索并播放江南这首歌，然后进入设置打开WIFI，再打开音乐关闭歌曲" → 返回：["音乐", "设置", "音乐"]

    约束：
    - 仅选择 `APP_LIST` 中存在的应用名；不得返回 "无匹配应用"；必须至少选择一个最相关的应用。

    Args:
        query: 用户指令字符串

    Returns:
        一个包含所选应用名的列表（list[str]），至少包含一个元素。

    Raises:
        Exception: 调用LLM接口失败时抛出异常
    """
    # 构造 Prompt，要求严格的 JSON 数组输出，并强调仅选择主动操作的应用
    prompt = f"""
    你的任务是从下方用户指令中抽取需要操作的步骤，并为每个步骤从应用列表中选择一个最相关且“需要主动打开/操作”的应用名。只选择执行动作的主体应用，忽略仅作为对象/目标被提及的应用（例如被下载、被搜索、被购买的应用或内容名称）。输出一个严格的 JSON 数组（list[str]），数组中每个元素是你选择的应用名。

    应用列表：{APP_LIST}

    选择与输出规则：
    1. 只能选择列表中存在的应用名，禁止自创应用名称；
    2. 只输出严格的 JSON 数组，例如 ["音乐", "设置"]，不要添加任何解释、标点（除 JSON 必需符号）或额外文字；
    3. 若多步都在同一个应用中进行，列出一个应用即可；
    4. 不允许输出 "无匹配应用"；若难以判断，也必须选择一个最相关的应用；
    5. 安装/下载某应用时，选择应用分发/商店类应用（如“应用市场”），不要选择被安装/下载的目标应用名；
    6. 搜索/播放歌曲或视频等操作时，选择执行该操作所在的应用（如“音乐”、“QQ音乐”、“哔哩哔哩”等），不要选择仅被搜索/播放的对象名称；
    7. 只列出每一步“需要主动打开或操作”的应用，忽略不需要实际打开的目标对象名称。

    参考示例：
    - 用户指令：打开QQ音乐播放薛之谦的刚刚好 → ["QQ音乐"]
    - 用户指令：打开音乐帮我搜索并播放江南这首歌，然后进入设置打开WIFI，再打开音乐关闭歌曲 → ["音乐", "设置"]
    - 用户指令：打开应用市场下载微信 → ["应用市场"]
    - 用户指令：在应用市场搜索QQ音乐并安装 → ["应用市场"]

    用户指令：{query}
    """

    try:
        messages = [
            {"role": "system", "content": "你是一个精准的应用匹配助手。只选择执行操作时需要主动打开/操作的应用，忽略仅作为对象被提及的应用（例如被下载/被搜索的目标）。严格遵守输出规则，只输出严格的 JSON 数组。"},
            {"role": "user", "content": prompt}
        ]

        # 调用本地 Wrapper，温度设为0以保证输出稳定
        content = llm_wrapper.predict(messages, temperature=0.0)

        if content is None:
            raise Exception("调用大语言模型失败：API无响应或重试耗尽")

        content = content.strip()

        # 解析为列表
        apps: list[str] = []
        try:
            # 尝试清洗 Markdown
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            if content.startswith("```"):
                content = content.replace("```", "")

            parsed = json.loads(content)
            if isinstance(parsed, list):
                apps = [str(x).strip() for x in parsed if isinstance(x, str)]
        except Exception:
            apps = []

        # 仅保留在 APP_LIST 中的应用名
        apps = [a for a in apps if a in APP_LIST]

        # 兜底策略：若解析后为空，则根据 query 简单猜测一个最相关应用
        if not apps:
            # 直接包含某个应用名则选之（最长匹配优先）
            direct_matches = sorted(
                [a for a in APP_LIST if a in query], key=len, reverse=True)
            if direct_matches:
                apps = [direct_matches[0]]
            else:
                q = query.lower()
                if ("音乐" in query) or ("播放" in query) or ("歌曲" in query) or ("qq音乐" in q) or ("酷狗" in q):
                    apps = ["音乐"] if "音乐" in APP_LIST else [APP_LIST[0]]
                elif ("设置" in query) or ("wifi" in q) or ("wi-fi" in q) or ("蓝牙" in query):
                    apps = ["设置"] if "设置" in APP_LIST else [APP_LIST[0]]
                elif ("视频" in query) or ("电影" in query) or ("看剧" in query) or ("优酷" in q) or ("爱奇艺" in q) or ("哔哩哔哩" in query):
                    apps = ["视频"] if "视频" in APP_LIST else [APP_LIST[0]]
                else:
                    # 最后兜底，保证至少返回一个
                    apps = [APP_LIST[0]]

        return apps

    except Exception as e:
        raise Exception(f"调用大语言模型失败：{str(e)}")

# -------------------------- 新增函数：应用名映射包名 --------------------------


def get_app_package_name(app_name: str) -> str:
    """
    根据应用名，从JSON文件中映射获取对应的应用包名

    Args:
        app_name: 应用名（如"运动健康"），若为"无匹配应用"则直接返回

    Returns:
        对应的应用包名（如"com.huawei.hmos.health"），无匹配则返回"无匹配包名"

    Raises:
        Exception: JSON文件读取/解析失败时抛出异常
    """
    # 若应用名为无匹配，直接返回
    if app_name == "无匹配应用":
        return "无匹配包名"

    global _app_package_mapping

    # 首次调用时读取JSON文件并构建应用名->包名的映射
    if _app_package_mapping is None:
        try:
            # 检查JSON文件是否存在
            if not os.path.exists(APP_PACKAGE_JSON_PATH):
                raise FileNotFoundError(f"应用包名映射文件不存在：{APP_PACKAGE_JSON_PATH}")

            # 读取JSON文件
            with open(APP_PACKAGE_JSON_PATH, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)

            # 反转映射（原JSON是包名->应用名，需要转为应用名->包名）
            _app_package_mapping = {v: k for k, v in raw_mapping.items()}

        except json.JSONDecodeError as e:
            raise Exception(f"JSON文件解析错误：{str(e)}")
        except Exception as e:
            raise Exception(f"读取应用包名映射文件失败：{str(e)}")

    # 根据应用名查找包名
    return _app_package_mapping.get(app_name, "无匹配包名")


# -------------------------- 测试示例 --------------------------
if __name__ == "__main__":
    # 第一步：先创建示例JSON文件（用户可替换为自己的文件，这里仅做演示）
    demo_package_mapping = {
        "com.huawei.hmos.health": "运动健康",
        "com.huawei.hmsapp.himovie": "视频",
        "com.huawei.hmsapp.music": "音乐",
        "com.huawei.hmos.settings": "设置",
        "com.huawei.hmos.vmall": "华为商城",
        "com.huawei.hmsapp.appgallery": "应用市场"
    }
    # 写入示例JSON文件（实际使用时删除这部分，替换为自己的JSON文件）
    with open(APP_PACKAGE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(demo_package_mapping, f, ensure_ascii=False, indent=4)

    # 测试用例
    test_queries = [
        "打开QQ音乐播放薛之谦的刚刚好",
        "用高德地图导航到天安门",
        "打开运动健康记录跑步数据",
        "在华为商城买一部手机",
        "打开应用市场下载微信",
        "打开音乐帮我搜索并播放江南这首歌，然后进入设置打开WIFI，再打开音乐关闭歌曲"
    ]

    # 执行测试
    print(APP_LIST)
    for query in test_queries:
        try:
            # 第一步：匹配应用名（返回列表）
            matched_apps = get_matched_app(query)
            # 第二步：根据应用名获取包名（逐个映射）
            package_names = [get_app_package_name(app) for app in matched_apps]

            print(f"用户指令：{query}")
            print(f"匹配应用：{matched_apps}")
            print(f"应用包名：{package_names}\n")
            time.sleep(1)  # 避免接口调用频率过高
        except Exception as e:
            print(f"处理指令'{query}'时出错：{e}\n")

    # 清理示例JSON文件（可选）
    if os.path.exists(APP_PACKAGE_JSON_PATH):
        os.remove(APP_PACKAGE_JSON_PATH)
