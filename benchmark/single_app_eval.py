# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import json
import os
from typing import Optional

from PIL import Image
import io

from utils import (
    read_json,
    encode_image    
)
from benchmark.utils import (
    PaddleOcrClient,
    LlmInterface,
    extract_json_format_string,
    add_action_2_screenshot,
    get_screenshot_file_names,
    extract_action_details,
)
from benchmark.prompts import (
    EVAL_AGENT_SYS_PROMPT,
    EVAL_AGENT_USER_PROMPT,
    get_action_mode_prompt,
    SUMMARY_IMG_AGENT_SYS_PROMPT,
    SUMMARY_IMG_AGENT_USER_PROMPT
)


class SingleAppEval(object):

    single_dataset_path = "benchmark/data/single_app.json"

    def __init__(self,
                 ocr_model: str,
                 ocr_endpoint: str,
                 ocr_api_key: str,
                 llm_model: str, 
                 llm_endpoints: str, 
                 llm_api_key: str,
                 llm_check_way: str,
                 usage_tracking_path: Optional[str] = None) -> None:
        self.single_dataset_dict = self._get_dict_single_datasets()
        self.ocr_ins = PaddleOcrClient(ocr_model, ocr_endpoint, ocr_api_key)
        self.llm_ins = LlmInterface(llm_model, llm_endpoints, llm_api_key, usage_tracking_path=usage_tracking_path)
        self.llm_check_way = llm_check_way
    
    def _get_dict_single_datasets(self) -> dict[dict]:
        single_dataset = read_json(self.single_dataset_path)
        single_dataset_dict = {}
        for sub_dataset in single_dataset:
            single_dataset_dict[sub_dataset["query"]] = sub_dataset
        return single_dataset_dict
    
    def _key_component_match(self,
                             current_key_components: list[str],
                             screenshot_list: list) -> tuple[str, int]:
        key_components = [component.lower() for component in current_key_components]
        if len(screenshot_list) > 0:
            # start from the last screenshot
            for file_path in reversed(screenshot_list):
                img_base64 = encode_image(file_path)
                screen_info = self.ocr_ins.infer(img_base64, "base64")

                screen_info = [component.lower() for component in screen_info]
                print(f"ocr recognize:\n{screen_info}")

                success_pool = []
                for component in key_components:
                    for each_screen_info in screen_info:
                        if component in each_screen_info:
                            success_pool.append(True)
                            break
                    else:
                        success_pool.append(False)

                # check whether all key component exist in the final screenshot
                success = all(success_pool)
                if success:
                    return file_path, len(screenshot_list)
            # doesn't match any screenshots
            return "", len(screenshot_list)
        else:
            return "", 0

    def _summary_screenshots(self,
                             task_desc: str,
                             img_base64_list: list,
                             img_fmt_list: list) -> tuple[str, int]:
        img_len = len(img_base64_list)
        if img_len != len(img_fmt_list):
            raise ValueError(f"img_base64_list and img_fmt_list length mismatch: "
                            f"{img_len} vs {len(img_fmt_list)}")

        round_num = max(0, (img_len - 1) // 10)

        if round_num == 0:
            return "", 0

        summaries = []
        for i in range(round_num):
            print(f"Summarizing screenshots batch {i + 1}/{round_num}...")
            start = i * 10
            end = start + 10
            sub_img_base64_list = img_base64_list[start:end]
            sub_img_fmt_list = img_fmt_list[start:end]

            batch_id = [f"b{i + 1}_i{j + 1}" for j in range(len(sub_img_base64_list))]

            history = self.llm_ins.add_prompt(SUMMARY_IMG_AGENT_SYS_PROMPT, [], [], role="system")
            user_prompt = SUMMARY_IMG_AGENT_USER_PROMPT.format(
                task_description=task_desc,
                batch_id=batch_id
            )
            history = self.llm_ins.add_prompt(user_prompt, sub_img_base64_list, sub_img_fmt_list, "user", history)

            if self.llm_check_way == "openai":
                llm_res = self.llm_ins.infer(history)
            elif self.llm_check_way == "csb":
                llm_res = self.llm_ins.infer(history, "csb-token")
            else:
                raise ValueError(f"unsupported check way {self.llm_check_way}, ensure use [openai, csb]")

            llm_res = extract_json_format_string(llm_res)
            summaries.append(llm_res)

        return "\n".join(summaries).strip(), round_num

    def _llm_eval(self,
                  task_desc: str,
                  screenshot_list: list,
                  include_text: str,
                  task_result_dir: str,
                  crop_w_ratio: float = 1.0,
                  crop_h_ratio: float = 1.0) -> tuple[str, str]:
        history = self.llm_ins.add_prompt(EVAL_AGENT_SYS_PROMPT, [], [], role="system")

        img_base64_list = []
        img_fmt_list = []
        for screenshot in screenshot_list:
            img = Image.open(screenshot)

            orig_w, orig_h = img.size
            crop_w = max(1, int(orig_w * crop_w_ratio))
            crop_h = max(1, int(orig_h * crop_h_ratio))
            left = (orig_w - crop_w) // 2
            top = (orig_h - crop_h) // 2
            right = left + crop_w
            bottom = top + crop_h
            cropped_img = img.crop((left, top, right, bottom))

            buf = io.BytesIO()
            cropped_img.save(buf, format='JPEG')

            img_base64_list.append(encode_image(byte_stream=buf.getvalue()))
            img_fmt_list.append("jpg")
        
        img_summary, round_num = self._summary_screenshots(task_desc, img_base64_list, img_fmt_list)

        if include_text == "false":
            extra_action = extract_action_details(task_result_dir)
        else:
            extra_action = ""
        
        action_info, action_attn = get_action_mode_prompt(include_text, extra_action)
        
        user_prompt = EVAL_AGENT_USER_PROMPT.format(
            history_info="",
            task_description=task_desc,
            text_on_screenshot=action_info,
            text_on_screenshot_attn=action_attn
        )

        if img_summary:
            user_prompt += f"\n\nHere is the summary of {round_num} batches of screenshots:\n```\n{img_summary}\n```"
        
        history = self.llm_ins.add_prompt(user_prompt, img_base64_list[10 * round_num:], img_fmt_list[10 * round_num:], "user", history)

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

    def eval(self,
             task_desc: str,
             task_result_dir: str,
             include_text: str,
             crop_w_ratio: float = 1.0,
             crop_h_ratio: float = 1.0) -> dict:
        current_dataset = self.single_dataset_dict[task_desc]
        if include_text == "true":
            add_action_2_screenshot(task_result_dir, True)
            screenshot_dir = "tap_and_text"
        else:
            add_action_2_screenshot(task_result_dir, False)
            screenshot_dir = "tap_only"

        screenshot_list = get_screenshot_file_names(os.path.join(task_result_dir, screenshot_dir))

        matched_screenshot, total_num_screenshot = self._key_component_match(current_dataset["key_component"], screenshot_list)

        if total_num_screenshot == 0:
            evaluation_detail = {}
            result = -1
        else:
            evaluation_detail = {
                "coarse_detect": 1 if matched_screenshot else 0,
            }
            fine_detect_result = 0
            fine_detect_reason = ""
            result = 0

            if matched_screenshot:
                fine_detect_reason, fine_detect_result = self._llm_eval(task_desc, screenshot_list, include_text, task_result_dir, crop_w_ratio, crop_h_ratio)
                fine_detect_result = int(fine_detect_result)

                if fine_detect_result:
                    result = 1
                
                evaluation_detail["matched"] = matched_screenshot
                evaluation_detail["total_num"] = total_num_screenshot
                evaluation_detail["fine_detect"] = fine_detect_result

                evaluation_detail["fine_detect_reason"] = (
                    fine_detect_reason.replace("\n", "").replace('"', "").replace("'", "")
                )

        if result == 1:
            print(f"{task_desc} is successful")
            evaluation_detail["eval_result"] = "success"
        elif result == 0:
            print(f"{task_desc} is failed")
            evaluation_detail["eval_result"] = "fail"
        elif result == -1:
            print(f"{task_desc} does not have any screenshots")
            evaluation_detail["eval_result"] = "error"

        return evaluation_detail
