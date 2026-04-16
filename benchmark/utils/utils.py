# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import json
import os
import re

from benchmark.utils.draw_text import load_trace


def fix_common_json_issues(json_str: str) -> str:
    # 'key': 'value' → "key": "value"
    # json_str = re.sub(r"'([^']+)'\s*:\s*'([^']*)'", r'"\1": "\2"', json_str)

    json_str = re.sub(r'\\(?!["\\/bfnrt])', r'\\', json_str)

    if r'\\"""' in json_str:
        # \\""" -> \"\"\"
        json_str = json_str.replace(r'\\"""', r'\\"\\"\\"')
    if '"""' in json_str:
        # """ -> \"\"\"
        json_str = json_str.replace('"""', r'\\"\\"\\"')

    if r"\\'''" in json_str:
        # \\''' -> \\\'\'\'
        json_str = json_str.replace(r"\\'''", r"\\'\\'\\'")
    if "'''" in json_str:
        # ''' -> \'\'\'
        json_str = json_str.replace("'''", r"\\'\\'\\'")

    return json_str


def extract_json_format_string(text: str) -> str:
    # print(f"Original text: \n{text}")

    start = text.find('{')
    if start == -1:
        return ""
    
    in_string = False
    escape = False
    brace_count = 0

    for i in range(start, len(text)):
        char = text[i]
        
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and not escape:
            escape = True
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_string = text[start: i + 1]

                    try:
                        json.loads(json_string)
                        # print(f"Extracted JSON string: \n{json_string}")
                        return json_string
                    except json.JSONDecodeError:
                    
                        json_string = fix_common_json_issues(json_string)
                        # print(f"Extracted JSON string: \n{json_string}")
                        return json_string
        escape = False
    else:
        return ""


def get_screenshot_file_names(target_dir: str) -> list:
    try:
        files = os.listdir(target_dir)
    except FileNotFoundError:
        return []
    
    screenshot_list = []
    for file in files:
        screenshot_list.append(os.path.join(target_dir, file))

    screenshot_list.sort(key=lambda x: os.path.splitext(x)[0])

    return screenshot_list


def extract_action_details(task_result_dir: str) -> str:
    trace_path = os.path.join(task_result_dir, "trace.json")
    descriptions = load_trace(trace_path)

    extra_action = ""
    for desc in descriptions:
        extra_action += f"The action that changes from screenshot No.{desc.step} to screenshot No.{desc.step + 1} is: **{desc.name}**, with details: **{desc.detail}**.\n"

    return extra_action.strip()
