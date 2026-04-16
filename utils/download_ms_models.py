# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import argparse
from modelscope import snapshot_download


def _main(args: dict) -> None:
    model_dir = snapshot_download(args["repo_name"])
    print(f"模型下载至{model_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="download modelscope model")
    parser.add_argument(
        "--repo_name", 
        default="stepfun-ai/GELab-Zero-4B-preview", 
        type=str
    )
    args = vars(parser.parse_args())

    _main(args)
