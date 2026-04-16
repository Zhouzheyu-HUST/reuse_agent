# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from __future__ import annotations

__author__ = "Zhipeng Hou"

from dataclasses import dataclass
import base64
import io
import json
import time
import os
from abc import ABC, abstractmethod
from typing import Union

from colorama import Fore
from PIL import Image

from utils import (
    write_json,
    Operate,
    print_out
)


@dataclass(frozen=True)
class ExecuteStepReturn(object):
    use_cache_flag: bool
    action_list: list[dict]


class BaseTask(ABC):
    def __init__(self, 
                 query: str,
                 bundle_name_dict: dict,
                 hdc_command: str = "hdc.exe",
                 max_retries: int = 3,
                 factor: float = 0.5,
                 max_operate_steps: int = 30) -> None:
        self.query = query
        self.bundle_name_dict = bundle_name_dict
        self.max_retries = max_retries
        self.factor = factor

        self.operate_ins = Operate(bundle_name_dict, hdc_command, factor)

        self.screenshot_width, self.screenshot_height = self.operate_ins.get_screen_scale()

        self.retry_count = 0
        self.step_id = -1
        self.record_path = os.path.join(os.environ['DATA_DIR'], 'record.json')

        self.total_elapsed_time = 0.0

        self.task_finished = False
        self.max_operate_steps = max_operate_steps

        self.use_cache_steps = 0

    def execute(self) -> None:
        current_record = {
            'query': self.query
        }
        write_json(self.record_path, current_record, "list", "a")

        while True:
            self.step_id += 1

            # get ui tree
            pre_ui_tree = self.operate_ins.dump_ui_tree(self.step_id, False)
            # get img and fmt
            pre_encoded_image, pre_fmt = self.get_env_data()

            start_time = time.time()
            # get operation
            execute_step_return = self.execute_step(pre_encoded_image, pre_fmt, pre_ui_tree)
            elapsed_time = time.time() - start_time

            action_seq = execute_step_return.action_list
            if execute_step_return.use_cache_flag:
                self.use_cache_steps += 1

            action_seq_str = json.dumps(action_seq, ensure_ascii=False)
            print_out(
                f'步骤{self.step_id + 1} 输出动作：{action_seq_str}',
                stdout=True,
                stdout_color=Fore.CYAN
            )

            if not self._execute_action_seq(action_seq):
                self.total_elapsed_time += elapsed_time
                self._record(elapsed_time, pre_encoded_image, action_seq)

                current_record = {
                    'eval_elapsed_time': self.total_elapsed_time / (self.step_id + 1)
                }
                write_json(self.record_path, current_record, "list", "a")
                self.operate_ins.kill_all_app_process()
                break
            
            # Wait for 2 seconds to ensure stability
            time.sleep(2)
            
            # get ui tree
            next_ui_tree = self.operate_ins.dump_ui_tree(self.step_id, True)
            # get img and fmt
            next_encoded_image, next_fmt = self.get_env_data()
            
            start_time = time.time()
            # get reflection
            self.reflect_action(pre_encoded_image, pre_fmt, next_encoded_image, next_fmt, pre_ui_tree, next_ui_tree)
            elapsed_time += time.time() - start_time
            self.total_elapsed_time += elapsed_time

            self._record(elapsed_time, pre_encoded_image, action_seq)

            if self.step_id + 1 == self.max_operate_steps:
                current_record = {
                    'eval_elapsed_time': self.total_elapsed_time / (self.step_id + 1)
                }
                write_json(self.record_path, current_record, "list", "a")
                self.operate_ins.kill_all_app_process()
                break

    def _execute_action_seq(self, 
                            action_seq: list[dict]) -> bool:
        for action in action_seq:
            action_type = action.get('type')
            if action_type == 'retry':
                self.retry_count += 1
                if self.retry_count >= self.max_retries:
                    print_out(
                        f'Task {self.query} failed after {self.max_retries} retries',
                        log_level="error"
                    )
                    return False
            else:
                self.retry_count = 0

            if action_type == 'done':
                print_out(f'Task {self.query} execution finished')
                self.task_finished = True
                return False

            self._do_action(action)
        return True

    def _do_action(self, 
                   action: dict) -> None:
        print_out(f'execute action {action}')

        trace_json = {
            "step_number": self.step_id
        }

        action_type = action.get('type')
        params = action.get('params')
        if action_type == 'open':
            trace_json["action"] = "open_app"
            trace_json["bounds"] = "[0,0][0,0]"
            trace_json["original_item"] = {
                "id": "",
                "text": params.get('app_name')
            }
            self._perform_open(params)
        elif action_type == 'click':
            x, y = params.get('points')

            trace_json["action"] = action_type
            trace_json["bounds"] = "[0,0][0,0]"
            trace_json["original_item"] = {
                "id": "",
                "text": [
                    x,
                    y
                ]
            }

            self.operate_ins.perform_click(x, y)
        elif action_type == 'longclick':
            x, y = params.get('points')

            trace_json["action"] = action_type
            trace_json["bounds"] = "[0,0][0,0]"
            trace_json["original_item"] = {
                "id": "",
                "text": [
                    x,
                    y
                ]
            }
            self.operate_ins.perform_longclick(x, y)
        elif action_type == 'scroll':
            x1, y1, x2, y2 = params.get('points')
            trace_json["action"] = "scroll"
            trace_json["bounds"] = "[0,0][0,0]"
            trace_json["original_item"] = {
                "id": "",
                "text": f"[{x1},{y1}][{x2},{y2}]"
            }
            self.operate_ins.perform_scroll(x1, y1, x2, y2)
        elif action_type == 'set_text':
            text = params.get('text')
            enter = params.get('enter')
            trace_json["action"] = "input_text"
            trace_json["bounds"] = "[0,0][0,0]"
            trace_json["original_item"] = {
                "id": "",
                "text": text
            }
            self.operate_ins.perform_settext(text, enter)
        elif action_type == "back":
            trace_json["action"] = "back"
            trace_json["bounds"] = "[0,0][0,0]"
            trace_json["original_item"] = {
                "id": "",
                "text": ""
            }
            self.operate_ins.perform_back()
        elif action_type == "home":
            trace_json["action"] = "home"
            trace_json["bounds"] = "[0,0][0,0]"
            trace_json["original_item"] = {
                "id": "",
                "text": ""
            }
            self.operate_ins.perform_home()
        elif action_type == "retry":
            trace_json["action"] = "retry"
            trace_json["bounds"] = "[0,0][0,0]"
            trace_json["original_item"] = {
                "id": "",
                "text": ""
            }
        else:
            trace_json["action"] = "unknown_action"
            trace_json["bounds"] = "[0,0][0,0]"
            trace_json["original_item"] = {
                "id": "",
                "text": action_type
            }
            print_out(
                f'Unknown action: {action_type}',
                log_level="error"
            )
        
        trace_path = os.path.join(os.environ['DATA_DIR'], 'trace.json')
        write_json(trace_path, trace_json, "list", "a")

    def _perform_open(self, 
                      params: dict) -> None:
        app_name = params.get('app_name')

        app_name_dict = {}
        for com_package, app in self.bundle_name_dict.items():
            app_name_dict[app] = com_package
        
        package_name = app_name_dict.get(app_name)
        self.operate_ins.start_app(package_name)

    def _record(self, 
                elapsed_time: float, 
                encoded_image: str, 
                action_seq: list[dict]) -> None:
        image_dir = os.path.join(os.environ['DATA_DIR'], "ImageInfo")
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f'frame_{self.step_id}.jpeg')
        image_bytes = base64.b64decode(encoded_image)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((int(image.width / self.factor), int(image.height / self.factor)), Image.Resampling.LANCZOS)
        image.save(image_path, 'jpeg')

        ui_info_dir = os.path.join("JsonInfo", f'frame_{self.step_id}')

        current_record = {
            'step_id': self.step_id,
            'screenshot': os.path.join("ImageInfo", f'frame_{self.step_id}.jpeg'),
            'action_seq': action_seq,
            'elapsed_time': elapsed_time,
            'ui_info': [
                os.path.join(ui_info_dir, f'tree_origin.json')
            ]
        }
        write_json(self.record_path, current_record, "list", "a")

    def get_env_data(self) -> tuple[str, str]:
        encoded_image, fmt = self.operate_ins.get_screenshot_data()
        return encoded_image, fmt

    @abstractmethod
    def execute_step(self,
                     encoded_image: str, 
                     fmt: str,
                     ui_tree: Union[list, dict]) -> ExecuteStepReturn:
        """
        智能体结合query，当前截图的base64编码，以及截图的格式，推理出当前操作系列

        Args:
            encoded_image: 当前截图的base64编码
            fmt: 截图的格式，比如jpg
            ui_tree: 当前截图的控件树
        
        Returns:
            ExecuteStepReturn:
                - use_cache_flag => 当前输出的操作序列是否用到了图缓存
                - action_list => 智能体决策出的操作序列，是一个list[dict]，每一个dict是操控动作
        """
        pass

    @abstractmethod
    def reflect_action(self,
                       pre_encoded_image: str, 
                       pre_fmt: str,
                       next_encoded_image: str, 
                       next_fmt: str,
                       pre_ui_tree: Union[list, dict],
                       next_ui_tree: Union[list, dict]) -> None:
        """
        智能体结合query，操控前截图的base64编码，以及操控前截图的格式，操控后截图的base64编码，以及操控后截图的格式，推理出当前操作系列的反思结果

        Args:
            pre_encoded_image: 操控前截图的base64编码
            pre_fmt: 操控前截图的格式，比如jpg
            next_encoded_image: 操控后截图的base64编码
            next_fmt: 操控后截图的格式，比如jpg
            pre_ui_tree: 操控前截图的控件树
            next_ui_tree: 操控后截图的控件树
        Returns:
            None: 不返回值
        """
        pass
