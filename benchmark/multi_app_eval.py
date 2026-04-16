# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import io
import json
import os
import re
from typing import Optional

from PIL import Image

from benchmark.prompts import (
    APP_DECOMPOSITION_SYS_PROMPT, 
    APP_DECOMPOSITION_USER_PROMPT,
    EVAL_AGENT_SYS_PROMPT,
    EVAL_AGENT_USER_PROMPT,
    get_action_mode_prompt,
    MEMORY_SUMMARY_SYS_PROMPT,
    MEMORY_SUMMARY_USER_PROMPT    
)
from benchmark.utils import (
    LlmInterface,
    add_action_2_screenshot,
    get_screenshot_file_names,
    extract_json_format_string,
    extract_action_details    
)
from utils import (
    read_json,
    encode_image    
)


class MultiAppEval(object):
    multi_dataset_path = "benchmark/data/multi_app.json"

    def __init__(self,
                 llm_model: str, 
                 llm_endpoints: str, 
                 llm_api_key: str,
                 llm_check_way: str,
                 usage_tracking_path: Optional[str] = None) -> None:
        self.multi_dataset_dict = self._get_dict_multi_datasets()
        self.llm_ins = LlmInterface(llm_model, llm_endpoints, llm_api_key, usage_tracking_path=usage_tracking_path)
        self.llm_check_way = llm_check_way
    
    def _get_dict_multi_datasets(self) -> dict[dict]:
        multi_dataset = read_json(self.multi_dataset_path)
        multi_dataset_dict = {}
        for sub_dataset in multi_dataset:
            multi_dataset_dict[sub_dataset["query"]] = sub_dataset
        return multi_dataset_dict

    def _llm_call_stage1(self,
                         task_desc: str,
                         screenshot_list: list,
                         task_app: list[str],
                         stage_1_crop_w_ratio: float = 1.0,
                         stage_1_crop_h_ratio: float = 1.0) -> dict:
        history = self.llm_ins.add_prompt(APP_DECOMPOSITION_SYS_PROMPT, [], [], role="system")

        img_base64_list = []
        img_fmt_list = []
        for screenshot in screenshot_list:
            img = Image.open(screenshot)

            orig_w, orig_h = img.size
            crop_w = max(1, int(orig_w * stage_1_crop_w_ratio))
            crop_h = max(1, int(orig_h * stage_1_crop_h_ratio))
            left = (orig_w - crop_w) // 2
            top = (orig_h - crop_h) // 2
            right = left + crop_w
            bottom = top + crop_h
            cropped_img = img.crop((left, top, right, bottom))

            buf = io.BytesIO()
            cropped_img.save(buf, format='JPEG')

            img_base64_list.append(encode_image(byte_stream=buf.getvalue()))
            img_fmt_list.append("jpg")
        
        user_prompt = APP_DECOMPOSITION_USER_PROMPT.format(
            task_description=task_desc,
            task_app=task_app
        )

        history = self.llm_ins.add_prompt(user_prompt, img_base64_list, img_fmt_list, "user", history)

        if self.llm_check_way == "openai":
            llm_res = self.llm_ins.infer(history)
        elif self.llm_check_way == "csb":
            llm_res = self.llm_ins.infer(history, "csb-token")
        else:
            raise ValueError(f"unsupported check way {self.llm_check_way}, ensure use [openai, csb]")

        llm_res = extract_json_format_string(llm_res)
        try:
            llm_dict = json.loads(llm_res)
        except json.JSONDecodeError as e:
            print(e)
            llm_dict = {}
        
        return llm_dict

    def _eval_stage1(self,
                     task_desc: str,
                     task_result_dir: str,
                     include_text: str,
                     max_retries: int = 3,
                     stage_1_crop_w_ratio: float = 1.0,
                     stage_1_crop_h_ratio: float = 1.0) -> tuple[bool, dict]:
        current_dataset = self.multi_dataset_dict[task_desc]
        if include_text == "true":
            add_action_2_screenshot(task_result_dir, True)
            screenshot_dir = "tap_and_text"
        else:
            add_action_2_screenshot(task_result_dir, False)
            screenshot_dir = "tap_only"
        
        target_dir = os.path.join(task_result_dir, screenshot_dir)
        screenshot_list = get_screenshot_file_names(target_dir)

        app_order = []
        for idx in range(len(current_dataset["subtasks"])):
            app_order.append(current_dataset["subtasks"][f"subtask_{idx + 1}"]["app"])

        counter = 0
        while counter < max_retries:
            try:
                print(f"Start to evaluate task: {task_desc}, attempt {counter + 1}")
                llm_dict = self._llm_call_stage1(task_desc, screenshot_list, app_order, stage_1_crop_w_ratio, stage_1_crop_h_ratio)
                print(f"LLM response: {llm_dict}")
                break
            except Exception as e:
                print(f"Error occurred during evaluation: {e}")
                counter += 1
                continue
        else:
            return False, {}
        
        prev_end_idx = None
        for _, screen_idx in llm_dict.items():
            start_idx = screen_idx["start_screen"]
            end_idx = screen_idx["end_screen"]

            if start_idx == -1 or end_idx == -1:
                return False, llm_dict
            
            if start_idx > end_idx:
                return False, llm_dict
            
            if start_idx > len(screenshot_list) or end_idx > len(screenshot_list):
                return False, llm_dict
            
            if prev_end_idx is not None and prev_end_idx >= start_idx:
                return False, llm_dict
            
            prev_end_idx = end_idx
        
        return True, llm_dict

    def _stage2_data_prep(self,
                        current_dataset: dict) -> dict:
        app_count = {}

        for _, subtask_detail in current_dataset["subtasks"].items():
            app_name = subtask_detail["app"]
            if app_name not in app_count:
                app_count[app_name] = 0
            app_count[app_name] += 1
        
        result_app_details = {}

        suffix_count = {}
        for _, subtask_detail in current_dataset["subtasks"].items():
            app_name = subtask_detail["app"]

            if app_count[app_name] == 1:
                new_app_name = app_name
            else:
                if app_name not in suffix_count:
                    suffix_count[app_name] = 0
                suffix_count[app_name] += 1
                new_app_name = f"{app_name}_{suffix_count[app_name]}"
            
            result_app_details[new_app_name] = {
                "task": subtask_detail["task"],
                "history": subtask_detail["history"],
                "memory": subtask_detail["memory"]
            }
        return result_app_details

    def _llm_call_stage2(self,
                         task_desc: str,
                         screenshot_list: list,
                         include_text: str,
                         task_result_dir: str,
                         task_history: list,
                         slices: tuple[int, int],
                         stage_2_crop_w_ratio: float = 1.0,
                         stage_2_crop_h_ratio: float = 1.0) -> tuple[str, str]:
        history = self.llm_ins.add_prompt(EVAL_AGENT_SYS_PROMPT, [], [], role="system")

        img_base64_list = []
        img_fmt_list = []
        for screenshot in screenshot_list[slices[0] - 1: slices[1]]:
            img = Image.open(screenshot)

            orig_w, orig_h = img.size
            crop_w = max(1, int(orig_w * stage_2_crop_w_ratio))
            crop_h = max(1, int(orig_h * stage_2_crop_h_ratio))
            left = (orig_w - crop_w) // 2
            top = (orig_h - crop_h) // 2
            right = left + crop_w
            bottom = top + crop_h
            cropped_img = img.crop((left, top, right, bottom))

            buf = io.BytesIO()
            cropped_img.save(buf, format='JPEG')

            img_base64_list.append(encode_image(byte_stream=buf.getvalue()))
            img_fmt_list.append("jpg")
        
        if include_text == "false":
            extra_action = extract_action_details(task_result_dir)
        else:
            extra_action = ""
        
        action_info, action_attn = get_action_mode_prompt(include_text, extra_action)

        if task_history:
            history_info = "Please take the following historical information into consideration during your evaluation:"
            for history_class, history_content in task_history:
                history_info += f"\nThe historical information about '{history_class}' is: {history_content}"
        else:
            history_info = ""
        
        user_prompt = EVAL_AGENT_USER_PROMPT.format(
            history_info=history_info,
            task_description=task_desc,
            text_on_screenshot=action_info,
            text_on_screenshot_attn=action_attn
        )

        history = self.llm_ins.add_prompt(user_prompt, img_base64_list, img_fmt_list, "user", history)

        if self.llm_check_way == "openai":
            llm_res = self.llm_ins.infer(history)
        elif self.llm_check_way == "csb":
            llm_res = self.llm_ins.infer(history, "csb-token")
        else:
            raise ValueError(f"unsupported check way {self.llm_check_way}, ensure use [openai, csb]")

        llm_res = extract_json_format_string(llm_res)
        try:
            llm_dict = json.loads(llm_res)
        except json.JSONDecodeError as e:
            print(e)
            llm_dict = {}
        
        return llm_dict.get("reason", ""), llm_dict.get("result", "0")

    def _summary_stage2_memory(self,
                               screenshot_list: list,
                               slices: tuple[int, int],
                               memory_desc: str,
                               stage_2_crop_w_ratio: float = 1.0,
                               stage_2_crop_h_ratio: float = 1.0) -> str:
        history = self.llm_ins.add_prompt(MEMORY_SUMMARY_SYS_PROMPT, [], [], role="system")

        img_base64_list = []
        img_fmt_list = []
        for screenshot in screenshot_list[slices[0] - 1: slices[1]]:
            img = Image.open(screenshot)

            orig_w, orig_h = img.size
            crop_w = max(1, int(orig_w * stage_2_crop_w_ratio))
            crop_h = max(1, int(orig_h * stage_2_crop_h_ratio))
            left = (orig_w - crop_w) // 2
            top = (orig_h - crop_h) // 2
            right = left + crop_w
            bottom = top + crop_h
            cropped_img = img.crop((left, top, right, bottom))

            buf = io.BytesIO()
            cropped_img.save(buf, format='JPEG')

            img_base64_list.append(encode_image(byte_stream=buf.getvalue()))
            img_fmt_list.append("jpg")
        
        user_prompt = MEMORY_SUMMARY_USER_PROMPT.format(
            task_description=memory_desc
        )

        history = self.llm_ins.add_prompt(user_prompt, img_base64_list, img_fmt_list, "user", history)

        if self.llm_check_way == "openai":
            llm_res = self.llm_ins.infer(history)
        elif self.llm_check_way == "csb":
            llm_res = self.llm_ins.infer(history, "csb-token")
        else:
            raise ValueError(f"unsupported check way {self.llm_check_way}, ensure use [openai, csb]")

        llm_res = extract_json_format_string(llm_res)
        try:
            llm_dict = json.loads(llm_res)
        except json.JSONDecodeError as e:
            print(e)
            llm_dict = {}
        
        return llm_dict.get("summary", "")

    def _eval_stage2(self,
                     task_desc: str,
                     task_result_dir: str,
                     include_text: str,
                     stage_1_screen_dict: dict,
                     stage_2_crop_w_ratio: float = 1.0,
                     stage_2_crop_h_ratio: float = 1.0) -> tuple[bool, dict]:
        evaluation_detail = {}
        evaluation_detail["stage_1_screen_dict"] = stage_1_screen_dict

        current_dataset = self.multi_dataset_dict[task_desc]

        if include_text == "true":
            screenshot_dir = "tap_and_text"
        else:
            screenshot_dir = "tap_only"
        
        target_dir = os.path.join(task_result_dir, screenshot_dir)
        screenshot_list = get_screenshot_file_names(target_dir)

        subtasks = self._stage2_data_prep(current_dataset)
        evaluation_detail["subtasks"] = subtasks

        memory_dict = {}
        for app_name, screen_idx in stage_1_screen_dict.items():
            print(f">>>\n{app_name}")

            start_idx = screen_idx["start_screen"]
            end_idx = screen_idx["end_screen"]
            slices = (start_idx, end_idx)
            evaluation_detail[f"slices_{app_name}"] = slices

            task_desc = subtasks[app_name]["task"]

            history = []
            if subtasks[app_name]["history"]:
                pattern = r"\{(.*?)\}"
                matches = re.findall(pattern, task_desc)
                for match in matches:
                    key = match.strip()
                    history.append((key, memory_dict[key]))
            
            fine_detect_reason, fine_detect_result = self._llm_call_stage2(
                task_desc,
                screenshot_list,
                include_text,
                task_result_dir,
                history,
                slices,
                stage_2_crop_w_ratio,
                stage_2_crop_h_ratio
            )
            fine_detect_result = int(fine_detect_result)

            evaluation_detail[f"fine_detect_{app_name}"] = fine_detect_result
            evaluation_detail[f"fine_detect_reason_{app_name}"] = (
                fine_detect_reason.replace("\n", "").replace('"', "").replace("'", "")
            )

            if fine_detect_result == 0:
                return False, evaluation_detail

            if subtasks[app_name]["memory"].lower() != "none":
                print(f"Update memory for {app_name}: {subtasks[app_name]['memory']}")

                memory_summary = self._summary_stage2_memory(
                    screenshot_list,
                    slices,
                    subtasks[app_name]["memory"],
                    stage_2_crop_w_ratio,
                    stage_2_crop_h_ratio
                )

                memory_dict[subtasks[app_name]["memory"]] = memory_summary
                print(f"Memory summary: {memory_summary}")
        
        evaluation_detail["memory_dict"] = memory_dict
        return True, evaluation_detail

    def eval(self,
             task_desc: str,
             task_result_dir: str,
             include_text: str,
             max_retries: int = 3,
             stage_1_crop_w_ratio: float = 1.0,
             stage_1_crop_h_ratio: float = 1.0,
             stage_2_crop_w_ratio: float = 1.0,
             stage_2_crop_h_ratio: float = 1.0) -> dict:
        stage_1_check_flag, stage_1_screen_dict = self._eval_stage1(
            task_desc,
            task_result_dir,
            include_text,
            max_retries,
            stage_1_crop_w_ratio,
            stage_1_crop_h_ratio
        )

        if not stage_1_check_flag:
            print(f"{task_desc} evaluation failed at stage 1.")
            final_result = False
            evaluation_detail = {}
        else:
            print(f"{task_desc} evaluation passed at stage 1.")
            final_result, evaluation_detail = self._eval_stage2(
                task_desc,
                task_result_dir,
                include_text,
                stage_1_screen_dict,
                stage_2_crop_w_ratio,
                stage_2_crop_h_ratio
            )
        
        if final_result:
            print(f"{task_desc} is successful")
            evaluation_detail["eval_result"] = "success"
        elif not final_result:
            print(f"{task_desc} is failed")
            evaluation_detail["eval_result"] = "fail"
        
        return evaluation_detail
