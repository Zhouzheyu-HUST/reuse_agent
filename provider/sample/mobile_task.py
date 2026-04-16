# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


from typing import Union
from tasks import BaseTask, ExecuteStepReturn


class SampleMobileTask(BaseTask):
    def __init__(self, 
                 query: str,
                 bundle_name_dict: dict,
                 hdc_command: str = "hdc.exe",
                 max_retries: int = 3,
                 factor: float = 0.5,
                 max_execute_steps: int = 35) -> None:
        super().__init__(query, bundle_name_dict, hdc_command, max_retries, factor, max_execute_steps)
    
    def execute_step(self,
                     encoded_image: str, 
                     fmt: str,
                     ui_tree: Union[list, dict]) -> ExecuteStepReturn:
        print(f"current query is {self.query}")
        # 此示例为华为Mate 70 pro在设置里搜索麦克风权限的任务操作序列
        if self.step_id == 0:
            action_list = [{
                "type": "click",
                "params": {
                    "points": [
                        1144,
                        2113
                    ]
                }
            }]
            return ExecuteStepReturn(use_cache_flag=False, action_list=action_list)
        elif self.step_id == 1:
            action_list = [{
                "type": "click",
                "params": {
                    "points": [
                        613,
                        629
                    ]
                }
            }]
            return ExecuteStepReturn(use_cache_flag=False, action_list=action_list)
        elif self.step_id == 2:
            action_list = [{
                "type": "set_text",
                "params": {
                    "text": "麦克风权限",
                    "enter": True
                }
            }]
            return ExecuteStepReturn(use_cache_flag=False, action_list=action_list)
        elif self.step_id == 3:
            action_list = [{
                "type": "done"
            }]
            return ExecuteStepReturn(use_cache_flag=False, action_list=action_list)

    def reflect_action(self,
                       pre_encoded_image: str, 
                       pre_fmt: str,
                       next_encoded_image: str, 
                       next_fmt: str,
                       pre_ui_tree: Union[list, dict],
                       next_ui_tree: Union[list, dict]) -> None:
        pass
