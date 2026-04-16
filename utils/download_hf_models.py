# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import argparse
from huggingface_hub import snapshot_download


def _main(args: dict) -> None:
    if args["pattern"]:
        snapshot_download(
            repo_id=args["repo_name"],
            local_dir_use_symlinks=False,
            allow_patterns=[args["pattern"]]
        )
    else:
        snapshot_download(
            repo_id=args["repo_name"],
            local_dir_use_symlinks=False
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="download hf model")
    parser.add_argument(
        "--repo_name", 
        default="Qwen/Qwen3-VL-8B-Instruct", 
        type=str
    )
    parser.add_argument(
        "--pattern", 
        default="", 
        type=str
    )
    args = vars(parser.parse_args())

    _main(args)
