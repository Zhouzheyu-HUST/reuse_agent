# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"

import base64
import hashlib
import math
import os
import time
import json
import locale
import sys
from typing import (
    Optional,
    Union
)

from loguru import logger
from colorama import Style


def build_task_dir_name(query: str,
                        pass_idx: Optional[int] = None) -> str:
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    task_dir_name = f"{timestamp}_{query_hash}"
    if pass_idx is not None:
        task_dir_name += f"_pass_{pass_idx + 1}"
    return task_dir_name


def _count_cjk_chars(value: str) -> int:
    return sum(1 for char in value if '\u4e00' <= char <= '\u9fff')


def _count_common_cjk_chars(value: str) -> int:
    common_chars = set(
        "的一是在不了有人这中大来上个国到说们为子和你地出道也时年得就那要下以生会自着去之过家"
        "后可里天开关打启闭设置信网蓝牙麦克风权限搜查看图册照相片音乐歌应用设备手机系统结果目录"
        "任务进入退出选择当前搜索连接命令文本输入点击返回首页"
    )
    return sum(1 for char in value if char in common_chars)


def _count_mojibake_chars(value: str) -> int:
    suspicious_chars = {
        '�', 'Ã', 'Â', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë',
        'Ì', 'Í', 'Î', 'Ï', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö',
        '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß', 'æ', 'ø',
        'ð', 'þ', 'ã', 'å', 'ä', 'ö', 'ü', 'é'
    }
    control_count = sum(1 for char in value if ord(char) < 32 and char not in '\r\n\t')
    return sum(1 for char in value if char in suspicious_chars) + control_count


def _text_quality_score(value: str) -> int:
    cjk_count = _count_cjk_chars(value)
    common_cjk_count = _count_common_cjk_chars(value)
    mojibake_count = _count_mojibake_chars(value)
    replacement_count = value.count('�')
    return (common_cjk_count * 4) + (cjk_count * 2) - (mojibake_count * 2) - (replacement_count * 4)


def normalize_console_text(value: str) -> str:
    if not value:
        return value

    original_score = _text_quality_score(value)
    best_value = value
    best_score = original_score
    candidates = []

    for source_encoding, target_encoding in (
        ("gbk", "utf-8"),
        ("cp936", "utf-8"),
        ("utf-8", "gbk"),
        ("utf-8", "cp936"),
    ):
        try:
            candidate = value.encode(source_encoding).decode(target_encoding)
        except (UnicodeError, LookupError):
            continue
        candidates.append(candidate)

    for candidate in candidates:
        candidate_score = _text_quality_score(candidate)
        if (
            candidate_score > best_score
            and _count_common_cjk_chars(candidate) >= _count_common_cjk_chars(best_value)
        ):
            best_value = candidate
            best_score = candidate_score

    if (
        best_score >= original_score + 3
        and _count_common_cjk_chars(best_value) >= _count_common_cjk_chars(value)
        and _count_mojibake_chars(best_value) <= _count_mojibake_chars(value)
    ):
        return best_value
    return value


def decode_command_output(raw: Optional[bytes]) -> str:
    if not raw:
        return ""

    encodings = []
    for encoding in (
        "utf-8",
        locale.getpreferredencoding(False),
        sys.getfilesystemencoding(),
        "gbk",
        "cp936",
    ):
        if encoding and encoding.lower() not in {item.lower() for item in encodings}:
            encodings.append(encoding)

    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    return raw.decode("utf-8", errors="replace")


def read_json(file_path: str) -> Union[dict, list]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def write_json(file_path: str,
               input_data: Union[dict, list],
               json_type: str = "dict",
               mode: str = 'w') -> None:
    if mode not in ('w', 'a'):
        raise ValueError("[mode] must be 'w' or 'a'")

    if json_type not in ('dict', 'list'):
        raise ValueError("[json_type] must be 'dict' or 'list'")
    
    if json_type == "dict":
        # input_data must be dict
        if not isinstance(input_data, dict):
            raise ValueError("For dict json_type, input_data must be a dict")
        
        # overwrite
        if mode == "w":
            data = input_data
        # append
        else:
            try:
                data = read_json(file_path)
                if not isinstance(data, dict):
                    raise ValueError("Existing JSON is not a dict, by default a initial dict")
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                data = {}

            dup_keys = set(data.keys()) & set(input_data.keys())
            if dup_keys:
                raise KeyError(f"append failed：the field is existed -> {dup_keys}")
            data.update(input_data)

    else:
        if mode == "w":
            if not isinstance(input_data, list):
                data = [input_data]
            else:
                data = input_data
        else:
            try:
                data = read_json(file_path)
                if not isinstance(data, list):
                    raise ValueError("Existing JSON is not a list, by default a initial list")
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                data = []
            if isinstance(input_data, list):
                data.extend(input_data)
            else:
                data.append(input_data)
    
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def encode_image(image_path: Optional[str] = None,
                 byte_stream: Optional[bytes] = None) -> str:
    if image_path is None and byte_stream is None:
        raise ValueError("args [image_path] and [byte_stream] should not empty for all.")
    
    if image_path is not None and byte_stream is not None:
        raise ValueError("args [image_path] and [byte_stream] should have values for one.")
    
    if image_path:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        return base64.b64encode(byte_stream).decode('utf-8')


def setup_logging(log_level: str = "INFO") -> None:
    log_file = os.path.join(os.environ['DATA_DIR'], f'appagent_test.log')

    logger.remove()
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level=log_level
    )


def load_config(root_dir: str,
                results_dir: str,
                temp_dir: str) -> None:
    os.environ['ROOT_DIR'] = root_dir
    os.environ['TEMP_DIR'] = os.path.join(root_dir, temp_dir)
    os.environ['RESULTS_DIR'] = os.path.join(root_dir, results_dir)


def track_usage(res_json: dict) -> dict:
    usage = res_json['usage']
    prompt_tokens, completion_tokens, total_tokens = usage['prompt_tokens'], usage['completion_tokens'], usage['total_tokens']

    if "gpt-4o" in res_json['model']:
        prompt_token_price = (2.5 / 1000000) * prompt_tokens
        completion_token_price = (10 / 1000000) * completion_tokens
        return {
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "id": res_json.get('id', 'unknown'),
            "model": res_json.get('model', 'unknown'),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_token_price": prompt_token_price,
            "completion_token_price": completion_token_price,
            "total_price": prompt_token_price + completion_token_price
        }
    else:    
        return {
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "id": res_json.get('id', 'unknown'),
            "model": res_json.get('model', 'unknown'),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }


# apikey out of quota
class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, 
                 key: str, 
                 cause: Optional[str] = None) -> None:
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


# api key with no permission
class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, 
                 key: str, 
                 cause: Optional[str] = None) -> None:
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


def print_out(mesg: str,
              logout: bool = True,
              stdout: bool = False,
              stdout_color: str = "",
              log_level: str = "info") -> None:
    if logout:
        if log_level.lower() == "info":
            logger.info(mesg)
        elif log_level.lower() == "error":
            logger.error(mesg)
        elif log_level.lower() == "debug":
            logger.debug(mesg)
        elif log_level.lower() == "warning":
            logger.warning(mesg)
        
    if stdout:
        if stdout_color:
            print(stdout_color + mesg + Style.RESET_ALL)
        else:
            print(mesg)


def to_float(v: any) -> Optional[float]:
    """尽量把单元格值转成 float。失败返回 None。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        # 过滤 NaN
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    s = str(v).strip()
    if s == "":
        return None

    s = s.replace("s", "").replace("S", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def to_bool(v: any) -> Optional[bool]:
    """尽量把单元格值转成 bool。失败返回 None。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        # 0/1
        if v == 0:
            return False
        if v == 1:
            return True
    s = str(v).strip().lower()
    if s in {"true", "t", "yes", "y", "是", "成功", "pass", "ok"}:
        return True
    if s in {"false", "f", "no", "n", "否", "失败", "fail"}:
        return False
    return None


def mean_metric(xs: list[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def ratio_true(bs: list[Optional[bool]]) -> Optional[float]:
    """True 占比（忽略 None）。返回 0~1。"""
    valid = [b for b in bs if b is not None]
    if not valid:
        return None
    return sum(1 for b in valid if b) / len(valid)
