import json
import os.path
from typing import Dict, Any, List
from pathlib import Path


def action_seq_to_assistant_action(action: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    """
    将 record 中的 action_seq 单步转换成 history 中 assistant 的 action json
    """
    a_type = action["type"]
    params = action.get("params", {})

    if a_type == "open":
        return {
            "open": params["app_name"]
        }

    if a_type == "click":
        x, y = params["points"]
        return {
            "point": [
                int(x / width * 1000),
                int(y / height * 1000)
            ]
        }

    if a_type == "scroll":
        x1, y1, x2, y2 = params["points"]
        return {
            "point": [
                int(x1 / width * 1000),
                int(y1 / height * 1000)
            ],
            "to": [
                int(x2 / width * 1000),
                int(y2 / height * 1000)
            ]
        }

    if a_type == "longclick":
        x, y = params["points"]
        return {
            "point": [
                int(x / width * 1000),
                int(y / height * 1000)
            ],
            "duration": True
        }

    if a_type == "set_text":
        return {
            "type": params["text"]
        }

    if a_type == "done":
        return {
            "status": "finish"
        }

    # press/back/home
    if a_type in {"back", "home", "enter"}:
        return {
            "press": a_type
        }

    raise ValueError(f"Unknown action type: {a_type}")


def build_history_step(
        query: str,
        step: Dict[str, Any],
        img_root: str,
        ui_root: str,
        width: int,
        height: int
) -> List[Dict[str, Any]]:
    screenshot = step["screenshot"]
    # 将 ImageInfo 替换为 tap_only
    #screenshot = screenshot.replace("ImageInfo", "tap_only")
    ui_path = step["ui_info"][0]
    user_item = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"<Question>{query}</Question>\n当前屏幕截图：(<image>./</image>)"
            },
            {
                "type": "image_path",
                "image_path": {
                    "path": f"{img_root}\\{screenshot}"
                }
            },
            {
                "type": "ui_path",
                "ui_path": {
                    "path": f"{ui_root}\\{ui_path}"
                }
            }
        ]
    }

    # record 中 action_seq 理论上只有 1 个
    action = step["action_seq"][0]
    assistant_action = action_seq_to_assistant_action(action, width, height)

    assistant_item = {
        "role": "assistant",
        "content": json.dumps({
            "thought": "",
            "abstract": "",
            **assistant_action
        }, ensure_ascii=False)
    }

    return [user_item, assistant_item]


def save_history(
        history: List[Dict[str, Any]],
        save_path: str
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"[SAVED] history -> {save_path}")


def record_to_history(record: List[Dict[str, Any]], img_root, ui_root, width, height):
    history = []

    query = record[0]["query"]

    for item in record:
        if "step_id" not in item:
            continue

        history.extend(
            build_history_step(
                query=query,
                step=item,
                img_root=img_root,
                ui_root=ui_root,
                width=width,
                height=height
            )
        )

    return history


def load_record(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_to_history_and_save(root_path: Path, width: int, height: int, save_path: str):
    record_json = os.path.join(root_path, "record.json")
    record = load_record(record_json)

    history = record_to_history(
        record=record,
        img_root=root_path,
        ui_root=root_path,
        width=width,
        height=height
    )
    save_history(
        history,
        save_path
    )
