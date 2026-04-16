# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"

import argparse


def parse_cli_args_from_init() -> dict:
    parser = argparse.ArgumentParser(description="run GUI Agent test framework")
    parser.add_argument(
        "--file-setting-path", 
        default="configs/file_path_config.json", 
        type=str, 
        help="file setting path"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO", 
        type=str, 
        help="logger record level"
    )
    parser.add_argument(
        "--provider", 
        default="sample", 
        type=str, 
        help="which agent will be used"
    )
    parser.add_argument(
        "--hdc-command", 
        default="hdc/mac/hdc", 
        type=str, 
        help="hdc cli"
    )
    parser.add_argument(
        "--max-retries", 
        default=5, 
        type=int, 
        help="each step will be tried for maximum time"
    )
    parser.add_argument(
        "--factor", 
        default=0.5, 
        type=float, 
        help="resize img proportion"
    )
    parser.add_argument(
        "--max_execute_steps", 
        default=35, 
        type=int, 
        help="agent maximum steps in executing tasks"
    )
    parser.add_argument(
        "--eval", 
        default="true", 
        type=str, 
        choices=["true", "false"],
        help="whether to eval the task"
    )
    parser.add_argument(
        "--include_text",
        default="true", 
        type=str, 
        choices=["true", "false"],
        help="whether to add text to the screenshot"
    )
    parser.add_argument(
        "--eval_type", 
        default="single", 
        type=str, 
        choices=["single", "multi"],
        help="eval app type"
    )
    parser.add_argument(
        "--single_eval_w_crop", 
        default=1.0, 
        type=float, 
        help="evaluation for single eval with weight crop"
    )
    parser.add_argument(
        "--single_eval_h_crop", 
        default=1.0, 
        type=float, 
        help="evaluation for single eval with height crop"
    )
    parser.add_argument(
        "--multi_eval_w_stage1_crop", 
        default=1.0, 
        type=float, 
        help="evaluation for multi eval stage1 with weight crop"
    )
    parser.add_argument(
        "--multi_eval_h_stage1_crop", 
        default=1.0, 
        type=float, 
        help="evaluation for multi eval stage1 with height crop"
    )
    parser.add_argument(
        "--multi_eval_w_stage2_crop", 
        default=1.0, 
        type=float, 
        help="evaluation for multi eval stage2 with weight crop"
    )
    parser.add_argument(
        "--multi_eval_h_stage2_crop", 
        default=1.0, 
        type=float, 
        help="evaluation for multi eval stage2 with height crop"
    )
    parser.add_argument(
        "--agent_llm", 
        default="Qwen3-VL-8B-Instruct", 
        type=str, 
        help="call base agent llm"
    )
    parser.add_argument(
        "--cache_llm", 
        default="Qwen3-VL-8B-Instruct", 
        type=str, 
        help="call cache llm"
    )
    parser.add_argument(
        "--agent_deployment_env", 
        default="昇腾910B", 
        type=str, 
        help="call agent llm env"
    )
    parser.add_argument(
        "--cache_deployment_env", 
        default="昇腾910B", 
        type=str, 
        help="call cache llm env"
    )
    parser.add_argument(
        "--exector_times", 
        default=3, 
        type=int, 
        help="evaluation for pass@k"
    )
    parser.add_argument(
        "--enable_cache", 
        default="false", 
        type=str, 
        choices=["true", "false"],
        help="whether to use cache mechanism"
    )
    args = vars(parser.parse_args())
    return args
