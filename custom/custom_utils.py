import json
import os
from typing import Any, Dict

import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

HISTORY_EXAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "history_example.json")


def check_click_and_press(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    act_obj = data.get("act_obj", {})
    act_type = act_obj.get("act_type")
    point = act_obj.get("point")

    if act_type in ("click", "longpress") and point is not None:
        return True, tuple(point)
    else:
        return False, None


def load_history_points(file_path: str) -> list[dict]:
    """Loads cached user/assistant message pairs for history replay."""
    if not os.path.exists(file_path):
        logger.warning("History example file not found: %s", file_path)
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        raw_history = json.load(f)

    if len(raw_history) % 2 != 0:
        logger.warning(
            "History example entries should be in user/assistant pairs.")

    points: list[dict] = []
    for i in range(0, len(raw_history) - 1, 2):
        user_msg = raw_history[i]
        assistant_msg = raw_history[i + 1]
        if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
            logger.warning("Unexpected role ordering at history index %s", i)
            continue
        points.append({"user": user_msg, "assistant": assistant_msg})

    return points


def _ensure_action_dict(action_raw: Any) -> Dict[str, Any]:
    """Normalize the raw action payload into a dict for device.step."""
    if isinstance(action_raw, dict):
        return action_raw
    if isinstance(action_raw, str):
        try:
            parsed = json.loads(action_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Action string is not valid JSON: {action_raw}") from exc
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Action JSON did not produce a dict: {parsed}")
    raise TypeError(f"Unsupported action type: {type(action_raw).__name__}")


def build_user_content(query: str, screenshot_np: np.ndarray) -> list[dict]:
    """Create a user message payload with current query and screenshot."""
    return [
        {
            "type": "text",
            "text": f"<Question>{query}</Question>\n当前屏幕截图：(<image>./</image>)",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{screenshot_np}"
            },
        },
    ]


def make_history_points(base_workflow: list[str], action_list: list[int], app_list: list[int], app_id_dict: dict,
                        database_dir: str) -> list[dict]:
    """
    将 base_workflow 和对应的 action_list 转换为用于 replay 的 history points。

    返回值为 list[dict]，每项格式为 {"user": {...}, "assistant": {...}}，
    其中 assistant.content 为 JSON 字符串，内容包含 "thought" 和 "action" 字段，
    action 字段来自 abstract_json_path 中的 action_map（例如 "point"）。
    """
    points: list[dict] = []

    # 缓存每个 app_package_name 对应的 action map，避免重复读取文件
    abstract_cache: dict = {}

    # 将 action_list 与 base_workflow 对齐
    for idx, aid in enumerate(action_list or []):
        thought_idx = idx
        thought = base_workflow[thought_idx] if thought_idx < len(
            base_workflow) else ""

        # 从 app_list 获取对应的 app_id（可能不存在）
        app_id = None
        if app_list and idx < len(app_list):
            app_id = app_list[idx]

        # 通过 app_id 在 app_id_dict 中查找 app_package_name（兼容 str/int 键）
        app_package_name = None
        if app_id is not None and isinstance(app_id, (str, int)):
            # 尝试多种 key 形式
            app_package_name = app_id_dict.get(
                app_id) if app_id in app_id_dict else app_id_dict.get(str(app_id))

        # 读取或从缓存中获取 action_map
        action_map: dict = {}
        if app_package_name:
            if app_package_name in abstract_cache:
                action_map = abstract_cache[app_package_name]
            else:
                abstract_json_path = os.path.join(
                    database_dir, app_package_name, "actions", "abstracts.json")
                if os.path.exists(abstract_json_path):
                    try:
                        with open(abstract_json_path, 'r', encoding='utf-8') as f:
                            abstract_data = json.load(f)
                        action_map = abstract_data.get('act_map', {}) if isinstance(
                            abstract_data, dict) else {}
                    except Exception:
                        action_map = {}
                else:
                    action_map = {}
                abstract_cache[app_package_name] = action_map

        action_type = action_map.get(str(aid)) if aid is not None else None

        combined = {"thought": thought}
        if action_type == "pointto":
            combined['point'] = {}
        elif action_type is not None:
            combined[action_type] = {}

        assistant_msg = {
            "role": "assistant",
            "content": json.dumps(combined, ensure_ascii=False)
        }

        user_msg = {
            "role": "user",
            "content": [thought]
        }

        points.append({"user": user_msg, "assistant": assistant_msg})

    return points


def fill_history_point_content(assistant_content: Dict[str, Any], json_path: str) -> Dict[str, Any]:
    """
    补全单条 assistant.content 中缺失的 action 细节。

    参数：
        assistant_content: assistant 的 content（可能是 JSON 字符串或已解析的 dict），形如 {"thought":..., "POINT": {}}
        json_path: 指向包含实际动作细节的 JSON 文件（例如 act.json）的路径

    返回：补全后的 dict（未序列化）。
    支持从 act.json 的 act_obj 中提取 `point` -> 填充到 `point`，以及 `to` 等字段。
    """
    # 解析 assistant_content
    if isinstance(assistant_content, str):
        try:
            content = json.loads(assistant_content)
        except Exception:
            # 返回原始包装
            return {"thought": assistant_content}
    elif isinstance(assistant_content, dict):
        content = dict(assistant_content)
    else:
        return {"thought": str(assistant_content)}

    # 找到 action key（排除 thought）
    action_key = None
    for k in content.keys():
        if k != "thought":
            action_key = k
            break

    # 如果没有 action key，直接返回原始 content
    if action_key is None:
        return content

    # 若文件不存在，返回原始 content
    if not os.path.exists(json_path):
        return content

    # 读取 act.json
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            act_data = json.load(f)
    except Exception:
        return content

    # 支持不同命名：优先 act_obj
    act_obj = act_data.get('act_obj') if isinstance(act_data, dict) else None
    if act_obj is None:
        # 尝试兼容老格式
        act_obj = act_data.get('action') if isinstance(
            act_data, dict) else None

    # 如果仍然没有 act_obj，返回原始
    if not isinstance(act_obj, dict):
        return content

    # 填充点坐标
    if 'point' in act_obj and (not content.get(action_key)):
        pt = act_obj.get('point')
        # 保留原样（float 或 int），但确保为列表
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            # 将数值转换为 int
            try:
                content[action_key] = [
                    int(round(float(pt[0]))), int(round(float(pt[1])))]
            except Exception:
                content[action_key] = pt

    # 填充滑动方向等额外字段
    if 'to' in act_obj and 'to' not in content:
        content['to'] = act_obj.get('to')

    # 填充长按时长
    if 'duration' in act_obj and 'duration' not in content:
        content['duration'] = act_obj.get('duration')

    # 填充打开应用的名称
    if 'open' in act_obj:
        content['open'] = act_obj.get('open')

    # 填充系统指令
    if 'press' in act_obj:
        content['press'] = act_obj.get('press')

    # 填充文本输入（若存在）
    if 'text' in act_obj and 'type' in content:
        content['type'] = act_obj.get('text')

    return content
