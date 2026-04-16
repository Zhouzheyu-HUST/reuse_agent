import json
import os
import re
import shutil
import argparse  # ⭐ 新增：用于解析命令行参数
from pathlib import Path
from typing import List, Dict, Any
import subprocess  # ⭐ 新增：用于调用外部命令获取分辨率
from utils import read_json
from custom.custom_utils import check_click_and_press
from custom.update.update import update
from custom.reuse_judge import can_reuse_action
from custom.sim.query_to_npy import encode_queries_from_json
from convert_history import record_to_history_and_save
from custom.refine_all import compress_workflow
from custom.workflow_summarizer import summarize_workflow


RESULT_PATH = os.environ.get("RESULT_PATH") or "results/GUI Agent"
SAVE_DIR = "database"
GTE_MODEL_NAME = "gte"
APP_JSON_PATH = "configs/app_package_config.json"
DATABASE_PATH = "database"


def first_step_is_open(subdir):
    record_file = subdir / "record.json"
    with open(record_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if item.get("step_id") == 0:
            action = item["action_seq"][0]["type"]
            return action == "open"

    return False


# ⭐ 修改函数签名，接收 width 和 height
def save_workflows(screen_width: int, screen_height: int):
    database_path = Path(DATABASE_PATH)

    # 1. 清理旧数据
    if database_path.exists():
        print(f"[INFO] database 文件夹已存在，将清空所有内容")
        for item in database_path.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"[WARN] 删除 {item} 时出错: {e}")
        print(f"[INFO] 已清空 database 文件夹中的所有内容")
    else:
        print(f"[INFO] 创建 database 文件夹")
        database_path.mkdir(parents=True, exist_ok=True)

    # 2. 创建基础目录
    tasks_path = database_path / "tasks"
    tasks_path.mkdir(parents=True, exist_ok=True)
    part_reuse_tasks_path = database_path / "part_reuse_tasks"
    part_reuse_tasks_path.mkdir(parents=True, exist_ok=True)

    apps_json_path = database_path / "apps.json"
    with open(apps_json_path, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)

    result_path = Path(RESULT_PATH)
    if not result_path.exists():
        raise FileNotFoundError(f"RESULT_PATH 不存在: {RESULT_PATH}")

    # ==================== 去重逻辑 ====================
    # 字典结构调整为：{ unique_query: {"path": subdir, "steps": step_count} }
    candidate_tasks = {}

    print("[INFO] 开始扫描并筛选任务（去重模式：保留步数最少的成功记录）...")

    # 3. 第一遍循环：扫描结果，进行去重
    for subdir in result_path.iterdir():
        if not subdir.is_dir():
            continue

        # --- A. 检查是否成功 ---
        record_file = subdir / "record.json"
        if not record_file.exists():
            continue

        try:
            with open(record_file, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            continue

        eval_result = None
        for item in records:
            if isinstance(item, dict) and "eval_result" in item:
                eval_result = item["eval_result"]
                break

        if eval_result is None:  # or str(eval_result).lower() != "success":
            continue

        eval_success = str(eval_result).lower() == "success"

        # --- B. 正则提取任务 Query ---
        dir_name = subdir.name
        match = re.search(r'\d{8}_\d{6}_(.+)_pass@\d+', dir_name)

        if match:
            unique_query = match.group(1)
        else:
            unique_query = dir_name

        # --- C. 计算任务步数（统计 ImageInfo 文件夹下的 jpeg 数量） ---
        image_info_dir = subdir / "ImageInfo"
        if image_info_dir.exists() and image_info_dir.is_dir():
            # 使用 glob 匹配所有的 .jpeg 文件并计算数量
            step_count = len(list(image_info_dir.glob("*.jpeg")))
        else:
            # 如果没有找到文件夹或图片，设定为正无穷大，避免它成为最优解
            step_count = float('inf')

        # --- D. 筛选逻辑：保留步数最少的记录 ---
        if unique_query not in candidate_tasks:
            # 首次遇到该任务，直接收录，并保存 eval_success 标记
            candidate_tasks[unique_query] = {
                "path": subdir, "steps": step_count, "eval_success": eval_success}
            print(
                f"  [收录] {unique_query} (当前步数: {step_count}) success={eval_success}")
        else:
            # 遇到重复任务，比较并选择：优先收录成功记录，其次步数最少。
            current = candidate_tasks[unique_query]
            current_min_steps = current["steps"]
            current_success = bool(current.get("eval_success", False))

            # 如果已有记录是成功的，而且当前遍历到的是失败的，保留已有成功记录
            if current_success and not eval_success:
                # 保持当前已收录的成功记录，不更新
                continue

            # 如果已有记录是失败的，而当前遍历到的是成功的，则优先替换为成功记录
            if not current_success and eval_success:
                print(f"  [更新] {unique_query} 发现更优记录 (失败->成功) steps={step_count} success={eval_success}")
                candidate_tasks[unique_query] = {"path": subdir, "steps": step_count, "eval_success": eval_success}
                continue

            # 到这里说明两者成功状态相同（都成功或都失败），按步数最少选择
            if step_count < current_min_steps:
                print(f"  [更新] {unique_query} 发现更优记录 (步数: {current_min_steps} -> {step_count}) success={eval_success}")
                candidate_tasks[unique_query] = {"path": subdir, "steps": step_count, "eval_success": eval_success}
            elif step_count == current_min_steps:
                # 步数相同时，优先首步为 open 的记录
                if not first_step_is_open(current["path"]) and first_step_is_open(subdir):
                    print(f"  [更新] {unique_query} 发现更优记录 (第一步为open更优) success={eval_success}")
                    candidate_tasks[unique_query] = {"path": subdir, "steps": step_count, "eval_success": eval_success}

    # 4. 第二遍循环：只处理筛选出的任务
    print(f"\n[INFO] 筛选完成，共有 {len(candidate_tasks)} 个唯一任务准备入库。")
    print(f"[INFO] 当前使用的分辨率配置: {screen_width} x {screen_height}")

    for query, info in candidate_tasks.items():
        path = info["path"]
        steps = info["steps"]
        print(f"[正在入库] {query} (总步数: {steps}) ...")

        # ⭐ 将分辨率参数和 eval_success 标记传递下去
        save_one_workflow(path, screen_width, screen_height,
                          info.get("eval_success", False))


def inject_thought_by_file(
    full_history_path: str,
    modified_history_path: str
):
    full_history_path = Path(full_history_path)
    modified_history_path = Path(modified_history_path)

    with open(full_history_path, "r", encoding="utf-8") as f:
        full_history: List[Dict[str, Any]] = json.load(f)

    with open(modified_history_path, "r", encoding="utf-8") as f:
        modified_history: List[Dict[str, Any]] = json.load(f)

    thoughts = []
    for item in full_history:
        if item.get("role") != "assistant":
            continue
        try:
            content = json.loads(item["content"])
        except Exception:
            thoughts.append("")
            continue
        thoughts.append(content.get("thought", ""))

    thought_idx = 0
    for item in modified_history:
        if item.get("role") != "assistant":
            continue

        if thought_idx >= len(thoughts):
            break

        content = json.loads(item["content"])
        thought = thoughts[thought_idx]
        content["thought"] = thought
        content["abstract"] = thought

        item["content"] = json.dumps(content, ensure_ascii=False)
        thought_idx += 1

    with open(modified_history_path, "w", encoding="utf-8") as f:
        json.dump(modified_history, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 已写入 thought -> {modified_history_path}")


# ⭐ 修改函数签名，接收 width 和 height
def save_one_workflow(root_path: Path, screen_width: int, screen_height: int, eval_success: bool):
    full_history_path = root_path / "full_history.json"
    workflow_compressed_path = root_path / "workflow_compressed.json"
    modified_full_history_path = "modified_history.json"
    print(
        f"[INFO] save_one_workflow called for {root_path} eval_success={eval_success}")

    # is_compress_success = compress_workflow(full_history_path, workflow_compressed_path)
    # if not is_compress_success:
    # print(f"[WARN] Compress failed for {root_path}")
    # return
    summarize_workflow(root_path, screen_width, screen_height)

    # ⭐ 将参数传给 record_to_history_and_save
    record_to_history_and_save(
        root_path, screen_width, screen_height, str(modified_full_history_path))

    inject_thought_by_file(str(workflow_compressed_path),
                           str(modified_full_history_path))

    task_id = update(
        DATABASE_PATH, str(modified_full_history_path), eval_success)

    if not task_id:
        return

    if eval_success:
        task_dir = os.path.join(SAVE_DIR, "tasks", str(task_id))
    else:
        task_dir = os.path.join(SAVE_DIR, "part_reuse_tasks", str(task_id))

    apps_json_path = os.path.join(DATABASE_PATH, "apps.json")

    if os.path.exists(task_dir):
        for filename in sorted(os.listdir(task_dir), key=lambda x: int(x.split(".")[0])):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(task_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            thought = data.get("thought")
            action_id = data.get("action_id")
            app_id = data.get("app_id")
            app_id_dict = read_json(apps_json_path)
            app_package_name = app_id_dict.get(str(app_id))

            if action_id == 0:
                continue

            variant_num = data.get("variant_num")
            action_variant_dir = os.path.join(
                SAVE_DIR, app_package_name, "actions", str(
                    action_id), str(variant_num)
            )

            act_dir = os.path.join(action_variant_dir, "act.json")
            have_coord, coord = check_click_and_press(act_dir)
            screen_jpeg_dir = os.path.join(action_variant_dir, "screen.jpeg")
            next_screen_jpeg_dir = os.path.join(
                action_variant_dir, "next_screen.jpeg")

            if os.path.exists(screen_jpeg_dir):
                if have_coord:
                    reuse_flag = can_reuse_action(
                        screen_jpeg_dir, thought, next_screen_jpeg_dir, coord[0], coord[1]
                    )
                else:
                    reuse_flag = can_reuse_action(
                        screen_jpeg_dir, thought, next_screen_jpeg_dir
                    )

                if reuse_flag == 0:
                    data["action_id"] = 0
                    print(f"  [Reuse] Step not reusable (Set action_id=0)")
                else:
                    print(f"  [Reuse] Step is reusable")

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

    if eval_success:
        old_query_json_path = os.path.join(
            SAVE_DIR, "tasks", "querys.json")
        old_query_npy_path = os.path.join(SAVE_DIR, "tasks", "querys.npy")

        encode_queries_from_json(
            old_query_json_path, old_query_npy_path, GTE_MODEL_NAME)
        
    else:
        old_query_json_path = os.path.join(
            SAVE_DIR, "part_reuse_tasks", "querys.json")
        old_query_npy_path = os.path.join(SAVE_DIR, "part_reuse_tasks", "querys.npy")

        encode_queries_from_json(
            old_query_json_path, old_query_npy_path, GTE_MODEL_NAME)


def get_screen_scale() -> tuple[int, int]:
    """
    独立调用 hdc 获取设备屏幕分辨率，不依赖 device_api。
    """
    # 1. 精准定位你的 hdc.exe 路径 (当前脚本目录下的 hdc 文件夹内)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    hdc_path = os.path.join(base_dir, "hdc", "hdc.exe")

    # 2. 拼接命令：调用 hdc shell snapshot_display
    command = [hdc_path, "shell", "snapshot_display", "/data/local/tmp/"]

    try:
        # 3. 运行命令 (隐藏黑窗口)
        kwargs = {}
        if os.name == 'nt':  # 如果是 Windows 系统
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore',
            **kwargs
        )

        # 4. 用正则提取输出结果中的宽和高 (完全复用原有的正则逻辑)
        ret = result.stdout
        match = re.search(r'width.*\s(\d+)\s*,\s*height.*\s(\d+)', ret)

        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width, height
        else:
            print(f"[WARN] 解析分辨率失败，hdc 输出: {ret}")
            return 1276, 2848  # 解析失败时给个默认兜底值

    except Exception as e:
        print(f"[ERROR] 执行 hdc 命令获取分辨率失败: {e}")
        return 1276, 2848  # 报错时给个默认兜底值


if __name__ == "__main__":
    width, height = get_screen_scale()
    print(f"[INFO] Detected screen resolution: {width} x {height}")
    # 传入参数
    save_workflows(width, height)
