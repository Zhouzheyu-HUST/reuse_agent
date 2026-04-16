# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional
import json

from benchmark.utils import (
    LlmInterface,
    extract_json_format_string    
)
from benchmark.prompts import (
    SPLIT_DATA_SYS_PROMPT,
    SPLIT_DATA_USER_PROMPT
)
from utils import (
    read_json,
    write_json    
)


class SplitMultiApp(object):

    multi_app_dataset_path = "data/multi_app.json"

    def __init__(self,
                 llm_model: str, 
                 llm_endpoints: str, 
                 llm_api_key: str,
                 llm_check_way: str,
                 usage_tracking_path: Optional[str] = None) -> None:
        self.llm_ins = LlmInterface(llm_model, llm_endpoints, llm_api_key, usage_tracking_path=usage_tracking_path)
        self.llm_check_way = llm_check_way
        self.data = read_json(self.multi_app_dataset_path)
    
    def _call_llm(self,
                  query: str,
                  app_list: list[str]) -> dict:
        history = self.llm_ins.add_prompt(SPLIT_DATA_SYS_PROMPT, [], [], "system")

        user_prompt = SPLIT_DATA_USER_PROMPT.format(
            task_description=query,
            task_app=app_list
        )
        history = self.llm_ins.add_prompt(user_prompt, [], [], "user", history)

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

    def split_data(self) -> None:
        split_data_list = []
        length = len(self.data)

        for idx, item in enumerate(self.data):
            query = item["query"]
            app_list = item["app"]

            print(f"Processing {idx + 1}/{length} task: {query} with apps {app_list}")

            llm_dict = self._call_llm(query, app_list)
            item["subtasks"] = llm_dict
            split_data_list.append(item)
        
        write_json(self.multi_app_dataset_path, split_data_list, "list", "w")


def _main():
    api_path = "../configs/api_settings.json"
    api_data = read_json(api_path)
    split_ins = SplitMultiApp(
        llm_model=api_data["llm_model"],
        llm_endpoints=api_data["llm_endpoints"],
        llm_api_key=api_data["llm_api_key"],
        llm_check_way=api_data["llm_check_way"]
    )
    split_ins.split_data()


if __name__ == "__main__":
    _main()
