# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"

from provider import (
    SampleMobileTask,
    GuiAgentMobileTask
)
from utils import print_out


class TaskManager(object):
    def __init__(self, 
                 query: str, 
                 provider: str,
                 bundle_name_dict: dict,
                 hdc_command: str = "hdc.exe",
                 max_retries: int = 3,
                 factor: float = 1.0,
                 max_execute_steps: int = 35) -> None:
        task_map = {
            "sample": SampleMobileTask,
            "华科何强组": GuiAgentMobileTask
        }
        print_out(
            f"now provider is {provider}",
            stdout=True
        )
        
        self.task_mgr = task_map.get(provider)(query, bundle_name_dict, hdc_command, max_retries, factor, max_execute_steps)

    def execute(self) -> None:
        self.task_mgr.execute()
