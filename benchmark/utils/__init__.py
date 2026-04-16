# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


__all__ = [
    "PaddleOcrClient",

    "extract_json_format_string",
    "get_screenshot_file_names",
    "extract_action_details",

    "LlmInterface",

    "add_action_2_screenshot"
]


from benchmark.utils.call_paddle_ocr import PaddleOcrClient
from benchmark.utils.utils import (
    extract_json_format_string,
    get_screenshot_file_names,
    extract_action_details    
)
from benchmark.utils.call_llm_api import LlmInterface
from benchmark.utils.draw_text import add_action_2_screenshot
