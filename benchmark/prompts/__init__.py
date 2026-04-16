# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


__all__ = [
    "EVAL_AGENT_SYS_PROMPT",
    "APP_DECOMPOSITION_SYS_PROMPT",
    "SPLIT_DATA_SYS_PROMPT",
    "MEMORY_SUMMARY_SYS_PROMPT",
    "SUMMARY_IMG_AGENT_SYS_PROMPT",

    "EVAL_AGENT_USER_PROMPT",
    "get_action_mode_prompt",
    "APP_DECOMPOSITION_USER_PROMPT",
    "SPLIT_DATA_USER_PROMPT",
    "MEMORY_SUMMARY_USER_PROMPT",
    "SUMMARY_IMG_AGENT_USER_PROMPT"
]


from benchmark.prompts.system_prompts import (
    EVAL_AGENT_SYS_PROMPT,
    APP_DECOMPOSITION_SYS_PROMPT,
    SPLIT_DATA_SYS_PROMPT,
    MEMORY_SUMMARY_SYS_PROMPT,
    SUMMARY_IMG_AGENT_SYS_PROMPT
)
from benchmark.prompts.user_prompts import (
    EVAL_AGENT_USER_PROMPT, 
    get_action_mode_prompt,
    APP_DECOMPOSITION_USER_PROMPT,
    SPLIT_DATA_USER_PROMPT,
    MEMORY_SUMMARY_USER_PROMPT,
    SUMMARY_IMG_AGENT_USER_PROMPT
)
