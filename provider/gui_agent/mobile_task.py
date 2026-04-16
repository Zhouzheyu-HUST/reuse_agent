# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"

from tasks import BaseTask
import base64
import io
import json
import time
import os
import re
import ast
from abc import ABC
from tasks.base_task import ExecuteStepReturn
import Path
from colorama import Fore
from PIL import Image

from utils import (
    read_json,
    write_json,
    Operate,
    print_out
)
from custom.agent_wrapper import Qwen3AgentWrapper
from typing import Any, Dict, List, Optional, Union
import urllib.parse
import subprocess
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from custom.sim.user_to_index import find_best_match
from custom.custom_utils import _ensure_action_dict, build_user_content, make_history_points, \
    fill_history_point_content, check_click_and_press
from custom.predict_dif import get_base_query_by_index, load_base_workflow_by_index, predict_dif
from custom.validity_check import validity_check, action_code, board_check
from custom.update.update import update
from custom.refine_all import compress_workflow
from custom.reuse_judge import can_reuse_action
from custom.sim.query_to_npy import encode_queries_from_json
from custom.match_app import get_matched_app, get_app_package_name, APP_LIST
from custom.extract_utils import extract_action, extract_thought
from custom.reflect_wrapper import ReflectWrapper
from custom.ui_check import calculate_ui_similarity_ordered

APIKEY = "sk-70d4e7e320a740a1862c10a4e7715d71"


class ConsecutiveFailureError(Exception):
    """当Agent在同一步骤连续失败达到阈值时抛出此异常。"""
    pass


GTE_MODEL_NAME = "gte"

IMG_DIR = "img_tmp"


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
REUSE_ENABLE = _config.get("reuse_enable")


class GuiAgentMobileTask(BaseTask):
    def __init__(self,
                 query: str,
                 bundle_name_dict: dict,
                 hdc_command: str = "hdc.exe",
                 max_retries: int = 3,
                 factor: float = 0.5,
                 max_operate_steps: int = 30
                 ) -> None:

        super().__init__(query, bundle_name_dict, hdc_command,
                         max_retries, factor, max_operate_steps)
        self.match_id = 0  # reuse_point下标
        self.agent = Qwen3AgentWrapper(
            temperature=0.1, use_history=True, history_size=5)
        self.reflector = ReflectWrapper()
        self.width, self.height = self.operate_ins.get_screen_scale()
        self.history_points = []
        self.stable_json_paths = []
        self.stable_img_paths = []
        self.save_dir = "database"
        self.enforce_agent = False
        self.task_id = time.strftime('%Y%m%d_%H%M%S') + '_' + self.query
        self.enforce_end = False

        # Ensure the save directory exists and that apps.json is present.
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            pass
        self.apps_json_path = os.path.join(self.save_dir, "apps.json")
        if not os.path.exists(self.apps_json_path):
            try:
                with open(self.apps_json_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False)
            except Exception:
                pass

        self.app_name_list = get_matched_app(self.query)
        self.all_app_list = APP_LIST
        self.app_id_dict = None
        self.reuse_point: list[int] = []
        self.base_workflow: list = []
        self.action_list: list = []
        self.app_list: list = []
        self.index = 0
        self.is_reusable_task = False
        self.current_action_seq = []
        # 【新增 1】用于暂存反思模块的错误反馈
        self.step_feedback_buffer = None
        self.current_thought = None  # 用于存储当前步骤的 thought，供反思模块使用
        # 【新增 2】记录当前这一步是否是复用的（用于反思失败时的回滚判断）
        self.is_current_step_reuse = False

        # 【新增 3】连续失败熔断机制
        self.consecutive_failure_count = 0
        self.max_consecutive_failures = 3  # 连续出现3次死循环则熔断

        # 【新增 4】用于打破回滚振荡的失败记忆
        self.last_failed_reuse_point_index = -1

        # ========== 【新增：死循环检测机制变量】 ==========
        self.repeat = 0
        self.prev_action_seq = []
        self.prev_thought = None
        self.action_history = []  # 记录历史动作序列的滑动窗口

        # 振荡参数控制
        self.oscillation_threshold = 2

        # 动态计算最大历史长度：如果阈值是3，最长需要保存 3*3=9 步，这里多加点冗余设为 12
        self.max_history_len = max(10, self.oscillation_threshold * 3 + 3)
        # ==================================================

        # ========== 【新增：前瞻终态校验机制变量】 ==========
        self.last_base_action_idx = -1        # 原任务的最后一步物理动作索引
        self.should_check_final_state = False # 是否满足条件开启终态靶向校验 (历史相似度 < 90%)
        self.check_final_state_next = False   # 下一步是否触发拦截
        self.target_ui_path = None            # 历史任务的预期终态 UI 路径
        self.changed_ui_threshold = 0.90            # 历史UI相似度阈值，低于此值认为动作破坏力大，开启校验
        self.old_new_ui_similarity_threshold = 1.0       # 当前UI与预期终态UI的相似度阈值，低于此值交给大模型兜底，高于此值直接终结任务
        # ==================================================
        self.should_meltdown_now = False  # 熔断开关
        self.from_part_reuse_tasks = False    # 是否来自 part_reuse 库
        # ========== 【新增：加载敏感词库】 ==========
        self.sensitive_words = []
        words_path = os.path.join("custom", "sensitive_words.txt")
        if os.path.exists(words_path):
            try:
                with open(words_path, 'r', encoding='utf-8') as f:
                    self.sensitive_words = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            except Exception as e:
                print_out(f"读取 sensitive_words.txt 失败: {e}", stdout=True, stdout_color=Fore.YELLOW)
        # ==========================================
        if REUSE_ENABLE:
            self._prepare_reuse_plan()

        current_record = {
            'query': self.query
        }
        write_json(self.record_path, current_record, "list", "a")

    def _prepare_reuse_plan(self) -> None:
        """预加载复用相关数据，供 execute_step 使用。"""
        # 清除之前的应用数据
        self.operate_ins.kill_all_app_process()
        part_reuse = False
        querys_json_path = os.path.join(self.save_dir, "tasks", "querys.json")
        querys_npy_path = os.path.join(self.save_dir, "tasks", "querys.npy")
        part_reuse_querys_json_path = os.path.join(
            self.save_dir, "part_reuse_tasks", "querys.json")
        part_reuse_querys_npy_path = os.path.join(
            self.save_dir, "part_reuse_tasks", "querys.npy")

        if not (os.path.exists(querys_json_path) and os.path.exists(querys_npy_path)):
            self.index = 0
            return

        # 支持 part_reuse 模式：可从两个历史 query 文件中匹配
        matched_json_path = querys_json_path  
        if not part_reuse:
            self.index = find_best_match(
                self.query, GTE_MODEL_NAME, querys_json_path, querys_npy_path)

            if self.index == 0:
                print("\n没有匹配到可复用任务")
                return

            print(f"\n匹配到历史任务，全局序号：{self.index}")
            self.is_reusable_task = True  # 新增：打上复用任务的永久烙印
            base_query = get_base_query_by_index(self.index, querys_json_path)
            tasks_dir = os.path.join(self.save_dir, "tasks")
            self.base_workflow, self.action_list, self.app_list = load_base_workflow_by_index(
                self.index, tasks_dir)
        else:
            # part_reuse=True 时，从两个文件中匹配（主库 + 部分复用库）
            try:
                from custom.sim.user_to_index import find_best_match_multi
            except Exception:
                print("无法导入 find_best_match_multi，回退到单文件匹配")
                self.index = find_best_match(
                    self.query, GTE_MODEL_NAME, querys_json_path, querys_npy_path)

                if self.index == 0:
                    print("\n没有匹配到可复用任务")
                    return

                matched_json_path = querys_json_path
                print(f"\n匹配到历史任务，全局序号：{self.index} (来自主库)")
                self.is_reusable_task = True
                base_query = get_base_query_by_index(
                    self.index, querys_json_path)
                tasks_dir = os.path.join(self.save_dir, "tasks")
                self.base_workflow, self.action_list, self.app_list = load_base_workflow_by_index(
                    self.index, tasks_dir)
            else:
                local_idx_1b, source_json = find_best_match_multi(
                    self.query, GTE_MODEL_NAME,
                    [querys_json_path, part_reuse_querys_json_path],
                    [querys_npy_path, part_reuse_querys_npy_path]
                )

                if local_idx_1b == 0:
                    print("\n没有匹配到可复用任务（跨库）")
                    return

                # 将 index 设为在对应 source_json 中的局部 1-based 索引
                self.index = local_idx_1b
                matched_json_path = source_json or querys_json_path
                print(f"\n匹配到历史任务，全局序号：{self.index} (来自: {matched_json_path})")
                self.is_reusable_task = True
                base_query = get_base_query_by_index(
                    self.index, matched_json_path)
                # 根据 matched_json_path 决定使用哪个 tasks 目录
                if matched_json_path == part_reuse_querys_json_path:
                    self.from_part_reuse_tasks = True
                    tasks_dir = os.path.join(self.save_dir, "part_reuse_tasks")
                else:
                    tasks_dir = os.path.join(self.save_dir, "tasks")
                self.base_workflow, self.action_list, self.app_list = load_base_workflow_by_index(
                    self.index, tasks_dir)
        print("选中的 base_query:", base_query)
        print(self.base_workflow)
        # ========== 【新增：解析具体每一步对应的 variant_id】 ==========
        self.variant_list = []
        try:
            querys_list = read_json(matched_json_path)
            # self.index 是从 1 开始的
            matched_task_id = querys_list[self.index - 1]["task_id"]

            # 遍历 base_workflow，读取 tasks/task_id/x.json 中的 variant_id
            for i in range(len(self.base_workflow)):
                try:
                    tasks_dir
                except NameError:
                    tasks_dir = os.path.join(self.save_dir, "tasks")

                step_file = os.path.join(tasks_dir, str(matched_task_id), f"{i+1}.json")
                if os.path.exists(step_file):
                    step_data = read_json(step_file)
                    # 如果老数据没有 variant_id，则默认 fallback 到 '1'
                    self.variant_list.append(str(step_data.get("variant_id", "1")))
                else:
                    self.variant_list.append("1")
        except Exception as e:
            print_out(
                f"解析 variant_id 失败，将默认使用 1 号变体。原因: {e}", stdout=True, stdout_color=Fore.YELLOW)
            self.variant_list = ["1"] * len(self.base_workflow)
        # =========================================================
        break_point = predict_dif(
            self.base_workflow, base_query, self.query, APIKEY)
        print("差异步骤:", break_point)

        self.app_id_dict = read_json(self.apps_json_path)
        self.history_points = make_history_points(
            self.base_workflow, self.action_list, self.app_list, self.app_id_dict, self.save_dir)

        for i in range(len(self.base_workflow)):
            if i + 1 not in break_point and self.action_list[i]:
                self.reuse_point.append(i + 1)

        print(self.reuse_point)

    def _check_single_folder(self, folder_name, checkdata_dir, stable_img_path, stable_json_path):
        """
        新增单个变体文件夹的视觉与UI检查函数。
        """
        folder_path = os.path.join(checkdata_dir, folder_name)
        try:
            # 1. 构建历史文件路径
            act_json_path = os.path.join(folder_path, "act.json")
            saved_img_path = os.path.join(folder_path, "screen.jpeg")
            saved_uitree_path = os.path.join(folder_path, "UI.json")

            # 2. 检查文件完整性
            if not all(os.path.exists(p) for p in [act_json_path, saved_img_path, saved_uitree_path]):
                return False, folder_name, f"文件缺失: {folder_name}"

            # 3. 从 act.json 读取历史点击坐标
            with open(act_json_path, 'r', encoding='utf-8') as f:
                act_data = json.load(f)
                point = act_data.get("act_obj", {}).get("point", [])
                if len(point) != 2:
                    return False, folder_name, f"坐标数据错误: {folder_name}"
                nx, ny = point

            # 4. 调用底层的 validity_check 进行当前 UI 匹配
            result = validity_check(saved_img_path, stable_img_path,
                                    saved_uitree_path, stable_json_path, nx, ny)

            return result, folder_name, f"检查完成: {folder_name}"

        except Exception as e:
            return False, folder_name, f"检查异常: {folder_name} - {str(e)}"

    def whole_validity_check(self, action, checkdata_dir, stable_img_path, stable_json_path, expected_variant_id='1'):
        """
        各种动作的比较方式
        """
        is_match = False
        success_index = None
        success_folder = None

        # action detect
        action_type = action_code(action)
        if action_type == 1:  # 有坐标 - 进入并行比对流程
            try:
                # 获取checkdata_dir下的所有数字文件夹
                subfolders = [f for f in os.listdir(checkdata_dir)
                              if os.path.isdir(os.path.join(checkdata_dir, f)) and f.isdigit()]
                subfolders.sort(key=int)  # 按数字大小排序

                if not subfolders:
                    print_out(f'步骤{self.step_id + 1} checkdata_dir下没有找到数字文件夹',
                              stdout=True, stdout_color=Fore.YELLOW)
                    return False

                # 并行检查所有文件夹，调用刚刚抽离的 self._check_single_folder
                with ThreadPoolExecutor(max_workers=min(len(subfolders), 8)) as executor:
                    future_to_folder = {
                        executor.submit(self._check_single_folder, folder, checkdata_dir, stable_img_path, stable_json_path): folder
                        for folder in subfolders
                    }

                    # 等待结果，一旦有成功就立即返回
                    for future in as_completed(future_to_folder):
                        folder_name = future_to_folder[future]
                        try:
                            result, returned_folder, message = future.result()
                            print(f"并行检查结果: {message}")

                            if result:  # 找到匹配的文件夹
                                is_match = True
                                success_index = int(returned_folder)
                                success_folder = os.path.join(
                                    checkdata_dir, returned_folder)

                                # 取消剩余的任务
                                for remaining_future in future_to_folder:
                                    if not remaining_future.done():
                                        remaining_future.cancel()

                                print_out(
                                    f'步骤{self.step_id + 1} 检测到关键UI匹配（文件夹{returned_folder}），进入复用流程',
                                    stdout=True,
                                    stdout_color=Fore.CYAN
                                )
                                break
                        except Exception as e:
                            print(f"并行检查异常: 文件夹{folder_name} - {str(e)}")

            except Exception as e:
                print_out(f'步骤{self.step_id + 1} 并行比对过程出错: {str(e)}',
                          stdout=True, stdout_color=Fore.RED)
                return False

        elif action_type == 2:  # 有键盘
            is_match = board_check(stable_img_path, stable_json_path)
            if is_match:
                print_out(
                    f'步骤{self.step_id + 1} 检测到键盘，进入复用流程，预期变体: {expected_variant_id}',
                    stdout=True,
                    stdout_color=Fore.CYAN
                )
        else:  # 其他动作（系统动作、滑动、打开应用）
            is_match = True
            print_out(
                f'步骤{self.step_id + 1} 上一步正确，进入复用流程，预期变体: {expected_variant_id}',
                stdout=True,
                stdout_color=Fore.CYAN
            )

        # 最终返回逻辑
        if action_type == 1:
            return is_match, success_index, success_folder
        else:
            target_folder = str(expected_variant_id)
            target_path = os.path.join(checkdata_dir, target_folder)

            # 增加存在性校验，如果找不到，再 fallback 回 '1'
            if not os.path.exists(target_path):
                print_out(f"警告：找不到变体文件夹 {target_folder}，回退到 1 号文件夹",
                          stdout=True, stdout_color=Fore.YELLOW)
                target_folder = '1'
                target_path = os.path.join(checkdata_dir, target_folder)

            return is_match, target_path

    def modify_history_json(self, history_path, longchain_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history_data = json.load(f)

        longchain_data = []

        # 处理每一条对话记录
        for i, item in enumerate(history_data):
            converted_item = {}
            converted_item["role"] = item["role"]

            if item["role"] == "user":
                # 用户消息需要添加图像和UI路径
                content_list = []

                # 添加文本部分
                text_content = item["content"][0]  # 第一个元素是文本
                content_list.append(text_content)

                # 添加图像路径部分
                step_num = i // 2  # 计算步骤序号（每2条记录为一步）
                print(step_num)
                image_content = {
                    "type": "image_path",
                    "image_path": {
                        "path": self.stable_img_paths[step_num]
                    }
                }
                content_list.append(image_content)

                # 添加UI路径部分
                ui_content = {
                    "type": "ui_path",
                    "ui_path": {
                        "path": self.stable_json_paths[step_num]
                    }
                }
                content_list.append(ui_content)

                converted_item["content"] = content_list

            else:
                # assistant消息需要转换格式
                try:
                    # 解析原始JSON字符串
                    original_content = json.loads(item["content"])

                    # 创建一个有序字典，确保字段顺序
                    converted_content = collections.OrderedDict()

                    # 处理每个字段，确保thought后面紧跟abstract
                    processed_thought = False

                    for key, value in original_content.items():
                        # 先添加thought字段
                        if key == "thought":
                            converted_content[key] = value
                            processed_thought = True

                            # 如果有thought但没有abstract，立即在thought后面添加abstract
                            if "abstract" not in original_content:
                                converted_content["abstract"] = value
                        else:
                            converted_content[key] = value

                    converted_item["content"] = json.dumps(
                        converted_content, ensure_ascii=False)

                except (json.JSONDecodeError, TypeError):
                    # 如果不是JSON字符串，直接复制
                    converted_item["content"] = item["content"]

            longchain_data.append(converted_item)

        # 写入转换后的文件
        with open(longchain_path, 'w', encoding='utf-8') as f:
            json.dump(longchain_data, f, ensure_ascii=False, indent=2)

        print(f"转换完成！已生成: {longchain_path}")
        print(f"共转换了 {len(longchain_data)} 条记录")

        return longchain_data

    def to_compact_json(self, data: list[str]):
        return json.dumps(
            data,
            indent=None,  # 移除缩进
            ensure_ascii=False,  # 中文不转义
            separators=(',', ':')  # 移除空格（最紧凑）
        )

    def _record_time_to_json(self, step_id, key, value):
        """
        将时间数据保存到与 full_history.json 相同的目录下
        """
        # 获取与保存 history 相同的目录
        data_dir = os.environ.get('DATA_DIR', '.')
        time_json_path = os.path.join(data_dir, "time.json")

        data = {}
        # 如果文件已存在，先读取旧数据以便追加
        if os.path.exists(time_json_path):
            try:
                with open(time_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                data = {}

        # 确保以步骤 ID 为键
        step_key = str(step_id)
        if step_key not in data:
            data[step_key] = {}

        # 记录时间，保留三位小数
        data[step_key][key] = round(value, 3)

        # 写回文件
        with open(time_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def execute_step(self,
                     encoded_image: str,
                     fmt: str,
                     ui_tree: Union[list, dict]) -> ExecuteStepReturn:
        if not REUSE_ENABLE:
            self.enforce_agent = True

        start_time = time.time()
        # 记录当前 UI 树路径与截图，供后续复用与存档
        stable_json_path = os.path.join(
            os.environ['DATA_DIR'], "JsonInfo", f'frame_{self.step_id}', 'tree_origin.json')
        self.stable_json_paths.append(stable_json_path)

        os.makedirs(os.path.join(IMG_DIR, self.task_id), exist_ok=True)
        stable_img_path = os.path.join(
            IMG_DIR, self.task_id, f'frame_{self.step_id}.jpeg')
        image_bytes = base64.b64decode(encoded_image)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((int(image.width*self.factor),
                             int(image.height*self.factor)), Image.Resampling.LANCZOS)
        image.save(stable_img_path, 'jpeg')
        self.stable_img_paths.append(stable_img_path)

        # ========== 【修改 1：開頭只打標，不提前 return】 ==========
        intercepted_done = False
        if getattr(self, 'check_final_state_next', False):
            self.check_final_state_next = False
            
            if hasattr(self, 'target_ui_path') and self.target_ui_path and os.path.exists(self.target_ui_path):
                try:
                    current_ui = self.stable_json_paths[-1]
                    target_sim = calculate_ui_similarity_ordered(current_ui, self.target_ui_path)
                    print_out(f"【智能分流】与预期终态的相似度: {target_sim:.3f} (阈值: {self.old_new_ui_similarity_threshold})", stdout=True, stdout_color=Fore.CYAN)
                    
                    if target_sim >= self.old_new_ui_similarity_threshold:
                        # ========== 【极简敏感词拦截】 ==========
                        is_sensitive = any(word in self.query for word in getattr(self, 'sensitive_words', []))
                        
                        if is_sensitive:
                            print_out("【智能分流】检测到状态切换类敏感词，UI树无法确信，直接交接给大模型做视觉确认！", stdout=True, stdout_color=Fore.YELLOW)
                            intercepted_done = False # 拒绝拦截，大模型兜底
                        else:
                            print_out("【智能分流】普通任务终态校验通过！准备注入完成指令...", stdout=True, stdout_color=Fore.GREEN)
                            if not self.from_part_reuse_tasks:
                                intercepted_done = True 
                        # ========================================
                        print_out("【智能分流】终态校验完美通过！准备注入完成指令...", stdout=True, stdout_color=Fore.GREEN)
                        if self.from_part_reuse_tasks == False:
                            intercepted_done = True  # <--- 核心：只做打標，不強行 return
                    else:
                        print_out("【智能分流】终态不匹配 (跳错页或有弹窗)，交接给大模型兜底。", stdout=True, stdout_color=Fore.YELLOW)
                except Exception as e:
                    print_out(f"【智能分流】UI 相似度计算异常，回退至大模型: {e}", stdout=True, stdout_color=Fore.RED)
        # =============================================================

        is_match = False
        success_folder = None

        # 在每一步开始时，如果不是被强制要求Agent执行，就清除上一步的失败记忆。
        # 这样可以确保“失败记忆”只在紧邻的下一步生效，避免过度禁止。
        # 当enforce_agent为True时，说明上一步失败了，此时需要保留失败记忆。
        if not self.enforce_agent:
            self.last_failed_reuse_point_index = -1

        if not self.enforce_agent and self.index and len(self.reuse_point):
            reuse_point_index = self.reuse_point[self.match_id]
            app_package_name = self.app_id_dict.get(
                str(self.app_list[reuse_point_index - 1]))
            actions_data_dir = os.path.join(
                self.save_dir, app_package_name, "actions")
            checkdata_dir = os.path.join(
                actions_data_dir, str(self.action_list[reuse_point_index - 1]))

            # 【修改】在检查前，确认此复用点不是刚刚失败过的点
            if reuse_point_index != self.last_failed_reuse_point_index:
                current_point = self.history_points[self.reuse_point[self.match_id] - 1]
                candidate_action = _ensure_action_dict(
                    current_point["assistant"].get("content"))
                # 增加传参 expected_variant_id
                expected_variant_id = self.variant_list[reuse_point_index - 1] if hasattr(
                    self, 'variant_list') else '1'
                check_result = self.whole_validity_check(
                    candidate_action, checkdata_dir, stable_img_path, stable_json_path, expected_variant_id)
                if isinstance(check_result, tuple) and len(check_result) == 3:
                    is_match, success_index, success_folder = check_result
                    if is_match:
                        print(f"匹配成功，使用文件夹索引: {success_index}")
                else:
                    is_match, success_folder = check_result
            else:
                print_out(f"跳过刚刚失败的复用点: {reuse_point_index}",
                          stdout=True, stdout_color=Fore.YELLOW)

            # 【修改】前瞻机制同样需要检查“失败记忆”
            if not is_match and self.match_id < len(self.reuse_point) - 1:
                next_reuse_point_index = self.reuse_point[self.match_id + 1]
                if next_reuse_point_index != self.last_failed_reuse_point_index:
                    next_checkdata_dir = os.path.join(
                        actions_data_dir, str(self.action_list[next_reuse_point_index - 1]))
                    next_current_point = self.history_points[next_reuse_point_index - 1]
                    next_action = _ensure_action_dict(
                        next_current_point["assistant"].get("content"))

                    # 增加传参 next_expected_variant_id
                    next_expected_variant_id = self.variant_list[next_reuse_point_index - 1] if hasattr(
                        self, 'variant_list') else '1'
                    next_check_result = self.whole_validity_check(
                        next_action, next_checkdata_dir, stable_img_path, stable_json_path, next_expected_variant_id)

                    if isinstance(next_check_result, tuple) and len(next_check_result) == 3:
                        is_match, success_index, success_folder = next_check_result
                        if is_match:
                            print(f"下一个复用点匹配成功，使用文件夹索引: {success_index}")
                    else:
                        is_match, success_folder = next_check_result

                    if is_match:
                        self.match_id += 1
                else:
                    print_out(
                        f"前瞻机制跳过刚刚失败的复用点: {next_reuse_point_index}", stdout=True, stdout_color=Fore.YELLOW)

        action_flag = False
        # 【新增】解析失败重试计数器，防止在当前步骤无限循环
        parsing_retry_count = 0
        max_parsing_retries = 2

        while (action_flag == False):
            # 【新增】在每次循环开始时检查重试次数
            if parsing_retry_count >= max_parsing_retries:
                is_match = False
                intercepted_done = False
                self.enforce_end = True

            if is_match:
                current_point = self.history_points[self.reuse_point[self.match_id] - 1]

                action = _ensure_action_dict(
                    current_point["assistant"].get("content"))

                filled = fill_history_point_content(
                    current_point["assistant"].get(
                        "content"), os.path.join(success_folder, "act.json")
                )
                current_point["assistant"]["content"] = json.dumps(
                    filled, ensure_ascii=False)

                action = _ensure_action_dict(
                    current_point["assistant"].get("content"))
            else:
                if intercepted_done:
                    current_step_prompt = self.query
                    print_out("拦截成功，直接输出 finish 状态", stdout=True, stdout_color=Fore.GREEN)
                    response = self.agent.predict_mm(
                        current_step_prompt, #<--- 这里改成新的 prompt
                        encoded_image,
                        self.to_compact_json(self.all_app_list),
                        self.to_compact_json(self.app_name_list),
                        True
                    )
                    action = response[3]
                    intercepted_done = False # 清除標誌位 
                else:
                    print("agent")

                    current_step_prompt = self.query
                
                    # 2. 如果 Buffer 里有反思的反馈，拼接到 Prompt 后面
                    if self.step_feedback_buffer:
                        print_out(f"Injecting Feedback to Agent: {self.step_feedback_buffer}", stdout=True, stdout_color=Fore.YELLOW)
                        current_step_prompt = f"{self.query}\n\n[上一步操作后给出的反馈]: {self.step_feedback_buffer}"
                    
                        # 3. 使用一次后清空，避免这条错误一直跟着
                        self.step_feedback_buffer = None
                    start_time_agent = time.time()
                    final_check = False
                    if self.enforce_end:
                        final_check = True
                    response = self.agent.predict_mm(
                        current_step_prompt, #<--- 这里改成新的 prompt
                        encoded_image,
                        self.to_compact_json(self.all_app_list),
                        self.to_compact_json(self.app_name_list),
                        final_check
                    )
                    agent_time = time.time() - start_time_agent
                    self._record_time_to_json(self.step_id, "agent_time", agent_time)
                    action = response[3]
                    self.enforce_agent = False

            action_seq = extract_action(action, self.width, self.height)
            thought = extract_thought(action)
            self.current_thought = thought
            print(f"Extracted thought: {self.current_thought}")

            if action_seq != False:
                action_flag = True
                # ========== 【修改：死循环检测逻辑 (参数化版本)】 ==========
                # 1. 将当前动作转为字符串格式并存入历史窗口
                current_action_str = str(action_seq)
                self.action_history.append(current_action_str)

                # 2. 保持滑动窗口大小
                if len(self.action_history) > self.max_history_len:
                    self.action_history.pop(0)

                L = len(self.action_history)
                self.repeat = 0
                loop_type = ""
                self.should_meltdown_now = False

                # ========== 【新增：计算当前步与上一步的 UI 相似度】 ==========
                ui_unchanged = False
                if len(self.stable_json_paths) >= 2:
                    try:
                        # 计算当前屏幕与上一步屏幕的 UI 相似度
                        ui_sim = calculate_ui_similarity_ordered(self.stable_json_paths[-2], self.stable_json_paths[-1])
                        # UI_check 返回 1 (或极高相似度) 判定为 UI 树未变
                        if ui_sim >= 0.99: 
                            ui_unchanged = True
                    except Exception as e:
                        print_out(f"UI 相似度计算异常，默认视为 UI 已改变: {e}", stdout=True, stdout_color=Fore.RED)
                # ==============================================================

                # 优先检测情况 (1): 周期 1 (A-A...) - 双轨制判定
                # 轨道 A：UI 没变 (严格模式：2次警告，3次熔断)
                if ui_unchanged and L >= 3 and self.action_history[-1] == self.action_history[-2] == self.action_history[-3]:
                    self.repeat = 1
                    loop_type = "类型(1) UI未变且连续死磕 (单步重复3次，触发最终熔断)"
                    self.should_meltdown_now = True
                elif ui_unchanged and L >= 2 and self.action_history[-1] == self.action_history[-2]:
                    self.repeat = 1
                    loop_type = "类型(1) UI未变且连续死磕 (单步重复2次，提出警告)"
                    self.should_meltdown_now = False
                    
                # 轨道 B：UI 变了 (宽松模式：4次警告，5次熔断)
                elif not ui_unchanged and L >= 5 and self.action_history[-1] == self.action_history[-2] == self.action_history[-3] == self.action_history[-4] == self.action_history[-5]:
                    self.repeat = 1
                    loop_type = "类型(1) UI变了但重复操作过多 (单步重复5次，触发最终熔断)"
                    self.should_meltdown_now = True
                elif not ui_unchanged and L >= 4 and self.action_history[-1] == self.action_history[-2] == self.action_history[-3] == self.action_history[-4]:
                    self.repeat = 1
                    loop_type = "类型(1) UI变了但重复操作过多 (单步重复4次，提出警告)"
                    self.should_meltdown_now = False
                
                # 检测情况 (2): 周期 2 振荡 (A-B...)
                elif L >= 5 and self.action_history[-5] == self.action_history[-3] == self.action_history[-1] and self.action_history[-4] == self.action_history[-2]:
                    self.repeat = 2
                    loop_type = "类型(2) 进退失措振荡 (已执行5步，触发最终熔断)"
                    self.should_meltdown_now = True
                elif L >= 4 and self.action_history[-4] == self.action_history[-2] and self.action_history[-3] == self.action_history[-1]:
                    self.repeat = 2
                    loop_type = "类型(2) 进退失措振荡 (已执行4步，提出警告)"
                    self.should_meltdown_now = False

                # 检测情况 (3): 周期 3 振荡 (A-B-C...)
                elif L >= 7 and self.action_history[-7] == self.action_history[-4] == self.action_history[-1] and self.action_history[-6] == self.action_history[-3] and self.action_history[-5] == self.action_history[-2]:
                    self.repeat = 3
                    loop_type = "类型(3) 复杂循环振荡 (已执行7步，触发最终熔断)"
                    self.should_meltdown_now = True
                elif L >= 6 and self.action_history[-6] == self.action_history[-3] and self.action_history[-5] == self.action_history[-2] and self.action_history[-4] == self.action_history[-1]:
                    self.repeat = 3
                    loop_type = "类型(3) 复杂循环振荡 (已执行6步，提出警告)"
                    self.should_meltdown_now = False
                elif L >= 5 and self.action_history[-5] == self.action_history[-2] and self.action_history[-4] == self.action_history[-1]:
                    self.repeat = 3
                    loop_type = "类型(3) 复杂循环振荡 (已执行5步，提出警告)"
                    self.should_meltdown_now = False

                # 3. 触发警告打印 (真正的熔断拦截动作在 reflect_action 中处理)
                if self.repeat > 0:
                    print_out(
                        f"警告：检测到 Agent 陷入死循环！循环模式: {loop_type}！", stdout=True, stdout_color=Fore.RED)
                

                # 4. 更新历史思考
                self.prev_thought = self.current_thought
                # ============================================
            else:
                is_match = False
                # 【新增】解析失败，增加重试计数
                parsing_retry_count += 1
                print_out(
                    f"Action parsing failed. Retry attempt {parsing_retry_count}/{max_parsing_retries}.", stdout=True, stdout_color=Fore.RED)

            if is_match == True:
                self.agent._push_history(
                    "user", build_user_content(self.query, encoded_image))
                self.agent._push_history(
                    "assistant", current_point["assistant"].get("content"))
                self.match_id += 1
                # ========== 【新增：精准触发下一步的终态校验】 ==========
                just_executed_base_step = self.reuse_point[self.match_id - 1]
                if getattr(self, 'should_check_final_state', False) and just_executed_base_step == getattr(self, 'last_base_action_idx', -1):
                    print_out(f"【触发前瞻】成功复用原任务最后一步 ({just_executed_base_step})，下一步准备进行终态靶向校验！", stdout=True, stdout_color=Fore.GREEN)
                    #不想要这个机制就把这里注释掉
                    #self.check_final_state_next = True
                # ===============================================================
                if self.match_id >= len(self.reuse_point):
                    self.index = 0
                elif self.reuse_point[self.match_id] - self.reuse_point[self.match_id - 1] > 1:
                    self.enforce_agent = True

        if any(item.get("type") == "done" for item in action_seq):
            self.agent.save_full_history_to_json(os.path.join(
                os.environ['DATA_DIR'], "full_history.json"))

        # 如果 action_seq 有效，则更新当前的 action_seq；否则保持之前的 action_seq 不变（可能是空列表）
        if action_seq is not False:
            self.current_action_seq = action_seq
        else:
            self.current_action_seq = []

        # ========== 【新增：记录当前步状态】 ==========
        # 在返回前，记录这一步到底是复用的还是 Agent 生成的，这对 reflect_action 的回滚判断至关重要
        self.is_current_step_reuse = is_match
        # ==========================================

        ret = ExecuteStepReturn(is_match, action_seq)
        execute_time = time.time() - start_time
        self._record_time_to_json(self.step_id, "execute_time", execute_time)
        return ret

    def reflect_action(self,
                       pre_encoded_image: str,
                       pre_fmt: str,
                       next_encoded_image: str,
                       next_fmt: str,
                       pre_ui_tree: Union[list, dict],
                       next_ui_tree: Union[list, dict],
                       ) -> None:
        start_time = time.time()
        # 1. 安全檢查：如果當前沒有動作序列，跳過反思
        if not hasattr(self, 'current_action_seq') or not self.current_action_seq:
            reflect_time = time.time() - start_time
            print_out(f"reflect_time: {reflect_time:.2f} seconds. No action sequence to reflect on, skipping reflection.",
                      stdout=True, stdout_color=Fore.YELLOW)
            self._record_time_to_json(
                self.step_id, "reflect_time", reflect_time)
            return

        # 2. 如果agent出现死循环，直接返回固定的feedback，强行干预，如果这里是复用的，则不需要强行干预
        if self.repeat > 0 and not getattr(self, 'is_current_step_reuse', False):
            print_out(
                f"拦截请求：检测到死循环(类型 {self.repeat})，直接生成强制纠错反馈！", stdout=True, stdout_color=Fore.RED)

            if self.repeat == 1:
                feedback = "思路错误，陷入死循环：Agent 连续执行了完全相同的动作。请重新思考，必须尝试点击不同的UI元素或改变操作策略！"
            elif self.repeat == 2:
                feedback = "思路错误，陷入进退振荡死循环：你在反复执行'进入某页面又立即退出'的操作。说明该方向完全错误，请停止这种反复横跳，务必尝试进入其他全新的页面或执行完全不同的操作！"
            elif self.repeat == 3:
                feedback = "思路错误，陷入复杂循环死循环：你正在重复相同的错误动作序列(例如:进入页面->无效操作->返回)。请回顾最近的历史动作，打破执念，必须尝试与之前完全不同的交互策略！"

        # 3. 只要匹配到了可复用的历史任务。
        elif getattr(self, 'is_reusable_task', False):
            feedback = "1"

        # 4. 調用云端小模型進行反思
        else:
            feedback = self.reflector.predict_reflection(
                query=self.query,
                pre_image_base64=pre_encoded_image,
                next_image_base64=next_encoded_image,
                action_seq=self.current_action_seq,
                thought=self.current_thought,
                width=self.width,
                height=self.height
            )

        # 5. 處理反饋結果
        if feedback.strip().startswith("1"):
            print_out("Reflect: Action Validated (Pass)",
                      stdout=True, stdout_color=Fore.GREEN)
            # 如果成功，清空 Buffer 以防萬一
            self.step_feedback_buffer = None
            # 【修改】成功时也要清除失败记忆
            self.last_failed_reuse_point_index = -1
            # 【修改】成功时重置连续失败计数器
            self.consecutive_failure_count = 0
            reflect_time = time.time() - start_time
            print_out(f"reflect_time: {reflect_time:.2f} seconds.",
                      stdout=True, stdout_color=Fore.GREEN)
            self._record_time_to_json(
                self.step_id, "reflect_time", reflect_time)
            return
        else:
            print_out(f"Reflect Feedback: {feedback}",
                      stdout=True, stdout_color=Fore.MAGENTA)

            self.consecutive_failure_count += 1

            # 检查是否达到连续失败阈值，或者【当前步骤触发了严格的死循环最终熔断】
            if self.consecutive_failure_count >= self.max_consecutive_failures or getattr(self, 'should_meltdown_now', False):
                print_out("触发硬性熔断机制！交接给大模型，下一步将强制输出 finish 动作结束任务。", stdout=True, stdout_color=Fore.RED)
                self.enforce_agent = True
                self.enforce_end = True

            # 1. 構造錯誤信息 (存入 Buffer，留給下一次 execute_step 讀取)
            error_message = (
                f"出现错误! 错误反馈: {feedback} "
            )
            self.step_feedback_buffer = error_message

            # 2. 強制 Agent 接管下一步 (無論是否回滾，下一步都必須由 Agent 思考)
            self.enforce_agent = True

            # 3. 【帶 Gap 保護的智能回滾】
            # 只有當剛剛這一步是「復用步驟」時，我們才考慮回滾。
            if self.is_current_step_reuse:
                should_rollback = True

                # --- Gap 檢測邏輯 ---
                # 檢查當前 match_id (已指向下一步) 和上一步之間是否存在空隙 (action_id=0)
                if self.match_id < len(self.reuse_point) and self.match_id > 0:
                    next_step_idx = self.reuse_point[self.match_id]
                    prev_step_idx = self.reuse_point[self.match_id - 1]

                    # 如果索引差大於 1，說明中間有 action_id=0 的步驟 (Gap)
                    if next_step_idx - prev_step_idx > 1:
                        print_out("Reflect Failed: 檢測到後續有 Gap (斷層)，暫停回滾，交由 Agent 穿越。",
                                  stdout=True, stdout_color=Fore.YELLOW)
                        should_rollback = False

                # 只有在「應該回滾」且不是第一步時，才執行指針後退
                if should_rollback and self.match_id > 0:
                    # 【新增】在回滚前，记录下刚刚失败的这个复用点的ID
                    # self.match_id 此时指向下一个点，所以失败的是 self.match_id - 1
                    failed_reuse_point_index = self.reuse_point[self.match_id - 1]
                    self.last_failed_reuse_point_index = failed_reuse_point_index
                    print_out(
                        f"Reflect Failed: 将复用点 {failed_reuse_point_index} 加入临时黑名单", stdout=True, stdout_color=Fore.YELLOW)
                    self.match_id -= 1
                    print_out(
                        f"Reflect Failed: 回滾 match_id 至 {self.match_id}", stdout=True, stdout_color=Fore.RED)

            # ====================================
            reflect_time = time.time() - start_time
            self._record_time_to_json(
                self.step_id, "reflect_time", reflect_time)
            return
