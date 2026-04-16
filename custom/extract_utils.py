from typing import Any, Dict
import urllib.parse
from typing import Optional

def extract_thought(data: Dict[str, Any]) -> Optional[str]:
    """
    從動作字典中提取思考過程。
    與 extract_action 保持相同的極簡處理邏輯，依賴於上游已傳入合法的 Dict。
    """
    if isinstance(data, dict) and "thought" in data:
        return data["thought"]
    return None


def extract_action(data: Dict[str, Any], width: int, height: int):
    """Execute a control step (tap/swipe/key/text/clear). Returns True if STATUS=finish/impossible."""
    if "point" in data:
        return handle_point(data, width, height)
    if "press" in data:
        return handle_press(data["press"])
    if "type" in data:
        return handle_type(data["type"])
    if "open" in data:
        return handle_open(data["open"])

    status = data.get("status")
    if status in {"finish", "impossible"}:
        return [{
            "type": "done",
            "params": {}
        }]
    if status == "continue":
        return [{
            "type": "retry",
            "params": {}
        }]

    return False


def handle_point(data: Dict[str, Any], width: int, height: int):
    x, y = data["point"]
    x = int(x / 1000 * width)
    y = int(y / 1000 * height)
    if "to" in data:
        x2, y2 = compute_swipe_target(data["to"], x, y, width, height)
        return [{
            "type": "scroll",
            "params": {
                "points": [x, y, x2, y2]
            }
        }]
    elif "duration" in data:
        return [{
            "type": "longclick",
            "params": {
                "points": [
                    x,
                    y
                ]
            }
        }]
    else:
        return [{
            "type": "click",
            "params": {
                "points": [
                    x,
                    y
                ]
            }
        }]


def compute_swipe_target(target: Any, x: int, y: int, width: int, height: int) -> tuple[int, int]:
    if isinstance(target, list):
        x2, y2 = target
        x2 = int(x2 / 1000 * width)
        y2 = int(y2 / 1000 * height)
        return x2, y2
    dirs = {
        "up": (0, -0.15),
        "down": (0, 0.15),
        "left": (-0.15, 0),
        "right": (0.15, 0),
    }
    if target not in dirs:
        raise ValueError(f"Invalid swipe direction: {target}")
    dx_ratio, dy_ratio = dirs[target]
    x2 = int(max(min(x + dx_ratio * width, width), 0))
    y2 = int(max(min(y + dy_ratio * height, height), 0))
    return x2, y2


def handle_press(key: str):
    return [{
        "type": key,
        "params": {}
    }]


def handle_type(raw: str):
    text = urllib.parse.unquote(raw)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return [{
        "type": "set_text",
        "params": {
            "text": escaped,
            "enter": False  # 是否回车
        }
    }]


def handle_open(app: str):
    return [{
        "type": "open",
        "params": {
            "app_name": app
        }
    }]
