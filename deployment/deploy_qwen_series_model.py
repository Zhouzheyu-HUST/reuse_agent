# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import asyncio
import json
import argparse
from collections import OrderedDict
from contextlib import contextmanager
import math
import os
import re
from threading import Thread, BoundedSemaphore, Lock, Event
import time
import uuid
from typing import List, Literal, Optional, Union, Dict, Any, Iterator

from fastapi.responses import StreamingResponse
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from transformers import (
    AutoConfig,
    AutoProcessor,
    TextIteratorStreamer,
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from qwen_vl_utils import process_vision_info
import uvicorn

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVL
    _VL_MODEL_CLASS_NAME = "AutoModelForImageTextToText"
except ImportError:
    # 兼容旧版 transformers（新类不存在时回退旧类）。
    from transformers import AutoModelForVision2Seq as AutoModelForVL
    _VL_MODEL_CLASS_NAME = "AutoModelForVision2Seq"


DEFAULT_MAX_IMAGE_DATA_URL_BYTES = 8 * 1024 * 1024  # 8 MiB decoded payload
MAX_STREAM_ERROR_MESSAGE_CHARS = 512


class ChatMessagePartText(BaseModel):
    type_: Literal["text"] = Field(default="text", alias="type")
    text: str = Field(...)


class ChatMessagePartImageURL(BaseModel):
    type_: Literal["image_url"] = Field(default="image_url", alias="type")
    image_url: Dict[str, Any] = Field(...)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(...)
    content: Union[str, List[Union[ChatMessagePartText, ChatMessagePartImageURL]]] = Field(...)


class ChatCompletionRequest(BaseModel):
    model: str = Field(...)
    messages: List[ChatMessage] = Field(...)
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=1.0)
    stream: bool = Field(default=False)
    # 文本模型可用，用于会话级 KV cache 复用。
    session_id: Optional[str] = Field(default=None)


class ChatCompletionChoiceMessage(BaseModel):
    role: str = Field(...)
    content: str = Field(...)


class ChatCompletionChoice(BaseModel):
    index: int = Field(...)
    message: ChatCompletionChoiceMessage = Field(...)
    finish_reason: str = Field(...)


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = Field(...)
    completion_tokens: int = Field(...)
    total_tokens: int = Field(...)


class ChatCompletionResponse(BaseModel):
    id: str = Field(...)
    object: str = Field(...)
    created: int = Field(...)
    model: str = Field(...)
    choices: List[ChatCompletionChoice] = Field(...)
    usage: ChatCompletionUsage = Field(...)


MODEL_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

SUPPORTED_KV_CACHE_IMPLEMENTATIONS = {
    "auto",
    "dynamic",
    "static",
    "offloaded",
    "offloaded_static",
}

KV_CACHE_CLI_CHOICES = ["recommended"] + sorted(SUPPORTED_KV_CACHE_IMPLEMENTATIONS)
SUPPORTED_MODEL_SIZE_TIERS = ["0.5b", "0.6b", "1.5b", "1.7b", "2b", "3b", "4b", "7b", "8b", "14b"]
MODEL_SIZE_TIER_CLI_CHOICES = ["auto"] + SUPPORTED_MODEL_SIZE_TIERS
SUPPORTED_MODEL_SIZE_TIERS_BY_MODE: Dict[str, List[str]] = {
    "text": ["0.5b", "0.6b", "1.5b", "1.7b", "3b", "4b", "7b", "8b", "14b"],
    "vl": ["2b", "3b", "4b", "7b", "8b", "14b"],
}
ANCHOR_MODEL_SIZE_TIER_BY_MODE: Dict[str, Dict[str, str]] = {
    "text": {
        "0.5b": "3b",
        "0.6b": "3b",
        "1.5b": "3b",
        "1.7b": "3b",
        "3b": "3b",
        "4b": "7b",
        "7b": "7b",
        "8b": "14b",
        "14b": "14b",
    },
    "vl": {
        "2b": "3b",
        "3b": "3b",
        "4b": "7b",
        "7b": "7b",
        "8b": "14b",
        "14b": "14b",
    },
}

# 非锚点档位（0.5/0.6/1.5/1.7/2/4/8B）对锚点配置做细粒度覆盖。
# 这样既能保留锚点（3/7/14B）的稳定基线，又能给出更贴合模型规模的推荐值。
MODEL_SIZE_TIER_PROFILE_OVERRIDES: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "cuda": {
        "text": {
            "0.5b": {"max_concurrent_generations": 8, "concurrency_timeout": 45.0, "session_kv_cache_max_entries": 1024},
            "0.6b": {"max_concurrent_generations": 8, "concurrency_timeout": 45.0, "session_kv_cache_max_entries": 1024},
            "1.5b": {"max_concurrent_generations": 6, "concurrency_timeout": 45.0, "session_kv_cache_max_entries": 768},
            "1.7b": {"max_concurrent_generations": 6, "concurrency_timeout": 60.0, "session_kv_cache_max_entries": 640},
            "4b": {"max_concurrent_generations": 3, "concurrency_timeout": 60.0, "session_kv_cache_max_entries": 384},
            "8b": {"max_concurrent_generations": 1, "concurrency_timeout": 75.0, "session_kv_cache_max_entries": 192},
        },
        "vl": {
            "2b": {"max_concurrent_generations": 2, "concurrency_timeout": 45.0},
            "4b": {"max_concurrent_generations": 1, "concurrency_timeout": 75.0},
            "8b": {"max_concurrent_generations": 1, "concurrency_timeout": 105.0},
        },
    },
    "mps": {
        "text": {
            "0.5b": {"concurrency_timeout": 45.0, "session_kv_cache_max_entries": 256, "session_kv_cache_ttl_seconds": 3600},
            "0.6b": {"concurrency_timeout": 45.0, "session_kv_cache_max_entries": 256, "session_kv_cache_ttl_seconds": 3600},
            "1.5b": {"concurrency_timeout": 60.0, "session_kv_cache_max_entries": 192},
            "1.7b": {"concurrency_timeout": 60.0, "session_kv_cache_max_entries": 160},
            "4b": {"concurrency_timeout": 75.0, "session_kv_cache_max_entries": 112},
            "8b": {"concurrency_timeout": 105.0, "session_kv_cache_max_entries": 80},
        },
        "vl": {
            "2b": {"concurrency_timeout": 75.0},
            "4b": {"concurrency_timeout": 105.0},
            "8b": {"concurrency_timeout": 150.0},
        },
    },
    "cpu": {
        "text": {
            "0.5b": {"concurrency_timeout": 60.0, "session_kv_cache_max_entries": 128},
            "0.6b": {"concurrency_timeout": 60.0, "session_kv_cache_max_entries": 128},
            "1.5b": {"concurrency_timeout": 90.0, "session_kv_cache_max_entries": 96},
            "1.7b": {"concurrency_timeout": 90.0, "session_kv_cache_max_entries": 96},
            "4b": {"concurrency_timeout": 150.0, "session_kv_cache_max_entries": 48},
            "8b": {"concurrency_timeout": 240.0, "session_kv_cache_max_entries": 24, "session_kv_cache_ttl_seconds": 720},
        },
        "vl": {
            "2b": {"concurrency_timeout": 150.0},
            "4b": {"concurrency_timeout": 240.0},
            "8b": {"concurrency_timeout": 420.0},
        },
    },
}


# 面向本地服务的保守生产默认值（可通过 CLI 覆盖）。
# 维度：device -> mode -> anchor_model_size_tier(3b/7b/14b)
# 非锚点档位通过 MODEL_SIZE_TIER_PROFILE_OVERRIDES 覆盖生成。
# 说明：
# - CUDA 默认按单机单卡场景给值；多卡场景建议按压测手动调大并发。
# - text/vl 分开给推荐值，VL 一般占用更大显存，因此并发更低。
RECOMMENDED_RUNTIME_PROFILES: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "cuda": {
        "text": {
            "3b": {
                "model_dtype": "bf16",
                "max_concurrent_generations": 4,
                "concurrency_timeout": 60.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
                "session_kv_cache_max_entries": 512,
                "session_kv_cache_ttl_seconds": 3600,
                "session_kv_cache_min_prefix_tokens": 64,
            },
            "7b": {
                "model_dtype": "bf16",
                "max_concurrent_generations": 2,
                "concurrency_timeout": 60.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
                "session_kv_cache_max_entries": 256,
                "session_kv_cache_ttl_seconds": 3600,
                "session_kv_cache_min_prefix_tokens": 64,
            },
            "14b": {
                "model_dtype": "bf16",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 90.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
                "session_kv_cache_max_entries": 128,
                "session_kv_cache_ttl_seconds": 1800,
                "session_kv_cache_min_prefix_tokens": 64,
            },
        },
        "vl": {
            "3b": {
                "model_dtype": "bf16",
                "max_concurrent_generations": 2,
                "concurrency_timeout": 60.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
            },
            "7b": {
                "model_dtype": "bf16",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 90.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
            },
            "14b": {
                "model_dtype": "bf16",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 120.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
            },
        },
    },
    "mps": {
        "text": {
            "3b": {
                "model_dtype": "fp16",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 60.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
                "session_kv_cache_max_entries": 128,
                "session_kv_cache_ttl_seconds": 1800,
                "session_kv_cache_min_prefix_tokens": 64,
            },
            "7b": {
                "model_dtype": "fp16",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 90.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
                "session_kv_cache_max_entries": 96,
                "session_kv_cache_ttl_seconds": 1800,
                "session_kv_cache_min_prefix_tokens": 64,
            },
            "14b": {
                "model_dtype": "fp16",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 120.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
                "session_kv_cache_max_entries": 64,
                "session_kv_cache_ttl_seconds": 900,
                "session_kv_cache_min_prefix_tokens": 64,
            },
        },
        "vl": {
            "3b": {
                "model_dtype": "fp16",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 90.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
            },
            "7b": {
                "model_dtype": "fp16",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 120.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
            },
            "14b": {
                "model_dtype": "fp16",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 180.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
            },
        },
    },
    "cpu": {
        "text": {
            "3b": {
                "model_dtype": "fp32",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 120.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
                "session_kv_cache_max_entries": 64,
                "session_kv_cache_ttl_seconds": 900,
                "session_kv_cache_min_prefix_tokens": 64,
            },
            "7b": {
                "model_dtype": "fp32",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 180.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
                "session_kv_cache_max_entries": 32,
                "session_kv_cache_ttl_seconds": 900,
                "session_kv_cache_min_prefix_tokens": 64,
            },
            "14b": {
                "model_dtype": "fp32",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 300.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
                "session_kv_cache_max_entries": 16,
                "session_kv_cache_ttl_seconds": 600,
                "session_kv_cache_min_prefix_tokens": 64,
            },
        },
        "vl": {
            "3b": {
                "model_dtype": "fp32",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 180.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
            },
            "7b": {
                "model_dtype": "fp32",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 300.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
            },
            "14b": {
                "model_dtype": "fp32",
                "max_concurrent_generations": 1,
                "concurrency_timeout": 600.0,
                "use_kv_cache": True,
                "kv_cache_implementation": "dynamic",
            },
        },
    },
}


def parse_model_dtype(model_dtype: str) -> torch.dtype:
    normalized = model_dtype.strip().lower()
    if normalized not in MODEL_DTYPE_MAP:
        raise ValueError(f"unsupported type: {model_dtype}, expected one of {list(MODEL_DTYPE_MAP.keys())}")
    return MODEL_DTYPE_MAP[normalized]


def detect_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_built() and mps_backend.is_available():
        return "mps"

    return "cpu"


def _load_model_config_dict_local(model_path: str) -> Optional[Dict[str, Any]]:
    if not model_path:
        return None
    if not os.path.isdir(model_path):
        return None

    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"Failed to read local config.json for mode detection: {e}")
    return None


def load_model_config_dict_for_mode_detection(model_path: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    local_config = _load_model_config_dict_local(model_path)
    if local_config is not None:
        return local_config, "local_config_json"

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config_dict = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        if isinstance(config_dict, dict):
            return config_dict, "autoconfig"
    except Exception as e:
        print(f"Cannot load model config for mode detection from '{model_path}': {e}")

    return None, None


def infer_model_mode_from_config_dict(config_dict: Dict[str, Any]) -> Optional[str]:
    if not isinstance(config_dict, dict):
        return None

    model_type = str(config_dict.get("model_type", "")).lower()
    architectures = [str(x).lower() for x in (config_dict.get("architectures") or [])]

    if "vision_config" in config_dict:
        return "vl"

    if any(("imagetexttotext" in arch) or ("vision2seq" in arch) for arch in architectures):
        return "vl"
    if any(("vl" in arch) and ("conditionalgeneration" in arch or "generation" in arch) for arch in architectures):
        return "vl"
    if any("causallm" in arch for arch in architectures):
        return "text"

    if "vl" in model_type or "vision" in model_type:
        return "vl"
    if model_type.startswith("qwen"):
        return "text"

    return None


def infer_model_mode_from_name_or_path(model_name: str, model_path: str) -> str:
    combined = f"{model_name} {model_path}".lower()
    compact = re.sub(r"[^a-z0-9]+", "", combined)
    if "qwen" in compact and "vl" in compact:
        return "vl"
    if any(token in combined for token in ["-vl", "_vl", " vl", "vision-language", "image-text"]):
        return "vl"
    return "text"


def resolve_mode_arg(args: Dict[str, Any]) -> tuple[str, str]:
    requested_mode = str(args.get("mode", "auto")).strip().lower()
    if requested_mode not in {"auto", "text", "vl"}:
        raise ValueError(f"unsupported mode: {requested_mode}, expected one of ['auto', 'text', 'vl']")

    model_path = str(args.get("model_path", ""))
    model_name = str(args.get("model_name", ""))

    config_dict, config_source = load_model_config_dict_for_mode_detection(model_path)
    inferred_mode_from_config = (
        infer_model_mode_from_config_dict(config_dict)
        if config_dict is not None
        else None
    )

    if requested_mode in {"text", "vl"}:
        if inferred_mode_from_config and requested_mode != inferred_mode_from_config:
            raise ValueError(
                "Mode mismatch detected before model loading: "
                f"--mode={requested_mode}, but model config ({config_source}) suggests mode={inferred_mode_from_config}. "
                "Please use the correct --mode or set --mode auto."
            )
        if inferred_mode_from_config:
            print(f"Validated explicit mode={requested_mode} with model config ({config_source}).")
            return requested_mode, f"explicit(validated_by_{config_source})"
        else:
            print(f"Using explicit mode={requested_mode}; model config auto-detection unavailable.")
            return requested_mode, "explicit(config_detection_unavailable)"

    if inferred_mode_from_config:
        print(f"Resolved mode from model config ({config_source}): {inferred_mode_from_config}")
        return inferred_mode_from_config, str(config_source)

    fallback_mode = infer_model_mode_from_name_or_path(model_name=model_name, model_path=model_path)
    print(f"Resolved mode by name/path heuristic (config unavailable): {fallback_mode}")
    return fallback_mode, "name_path_heuristic"


def normalize_model_size_tier(model_size_tier: str) -> str:
    normalized = model_size_tier.strip().lower()
    if normalized not in SUPPORTED_MODEL_SIZE_TIERS:
        raise ValueError(
            f"unsupported model_size_tier: {model_size_tier}, expected one of {SUPPORTED_MODEL_SIZE_TIERS}"
        )
    return normalized


def infer_model_size_tier(model_name: str, model_path: str) -> str:
    text = f"{model_name} {model_path}"
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*[bB]\b", text)

    if not matches:
        print("Cannot infer model size tier from model_name/model_path, fallback to '7b'.")
        return "7b"

    size_b = max(float(x) for x in matches)
    if size_b > 14.0:
        print(f"Detected model size {size_b}B > 14B, cap recommended profile tier to '14b'.")
        return "14b"

    numeric_tiers = [float(t[:-1]) for t in SUPPORTED_MODEL_SIZE_TIERS]
    nearest_idx = min(range(len(numeric_tiers)), key=lambda i: abs(numeric_tiers[i] - size_b))
    return SUPPORTED_MODEL_SIZE_TIERS[nearest_idx]


def resolve_model_size_tier_arg(args: Dict[str, Any]) -> str:
    model_size_tier = args.get("model_size_tier", "auto")
    if model_size_tier in (None, "auto"):
        return infer_model_size_tier(
            model_name=str(args.get("model_name", "")),
            model_path=str(args.get("model_path", "")),
        )
    return normalize_model_size_tier(str(model_size_tier))


def get_recommended_runtime_profile(device_type: str, mode: str, model_size_tier: str) -> Dict[str, Any]:
    normalized_device_type = device_type.strip().lower()
    normalized_mode = mode.strip().lower()
    normalized_model_size_tier = normalize_model_size_tier(model_size_tier)

    if normalized_device_type not in RECOMMENDED_RUNTIME_PROFILES:
        raise ValueError(
            f"unsupported device_type for recommendations: {device_type}, "
            f"expected one of {sorted(RECOMMENDED_RUNTIME_PROFILES.keys())}"
        )
    if normalized_mode not in {"text", "vl"}:
        raise ValueError(f"unsupported mode for recommendations: {mode}, expected one of ['text', 'vl']")
    if normalized_model_size_tier not in SUPPORTED_MODEL_SIZE_TIERS_BY_MODE[normalized_mode]:
        raise ValueError(
            f"model_size_tier={normalized_model_size_tier} is not supported for mode={normalized_mode}. "
            f"Supported tiers: {SUPPORTED_MODEL_SIZE_TIERS_BY_MODE[normalized_mode]}"
        )

    anchor_tier = ANCHOR_MODEL_SIZE_TIER_BY_MODE[normalized_mode][normalized_model_size_tier]
    profile = dict(RECOMMENDED_RUNTIME_PROFILES[normalized_device_type][normalized_mode][anchor_tier])
    tier_overrides = (
        MODEL_SIZE_TIER_PROFILE_OVERRIDES
        .get(normalized_device_type, {})
        .get(normalized_mode, {})
        .get(normalized_model_size_tier, {})
    )
    if tier_overrides:
        profile.update(tier_overrides)
    return profile


def apply_recommended_runtime_args(args: Dict[str, Any]) -> Dict[str, Any]:
    resolved_args = dict(args)
    device_type = detect_device_type()
    mode, mode_source = resolve_mode_arg(resolved_args)
    resolved_args["mode"] = mode
    resolved_args["mode_source"] = mode_source
    model_size_tier = resolve_model_size_tier_arg(resolved_args)
    resolved_args["model_size_tier"] = model_size_tier
    profile = get_recommended_runtime_profile(device_type, mode, model_size_tier)

    if resolved_args.get("model_dtype") in (None, "auto"):
        resolved_args["model_dtype"] = profile["model_dtype"]
    if resolved_args.get("max_concurrent_generations") is None:
        resolved_args["max_concurrent_generations"] = profile["max_concurrent_generations"]
    if resolved_args.get("concurrency_timeout") is None:
        resolved_args["concurrency_timeout"] = profile["concurrency_timeout"]
    if resolved_args.get("use_kv_cache") is None:
        resolved_args["use_kv_cache"] = profile["use_kv_cache"]
    if resolved_args.get("kv_cache_implementation") in (None, "recommended"):
        resolved_args["kv_cache_implementation"] = profile["kv_cache_implementation"]

    if mode == "text":
        if resolved_args.get("session_kv_cache_max_entries") is None:
            resolved_args["session_kv_cache_max_entries"] = profile["session_kv_cache_max_entries"]
        if resolved_args.get("session_kv_cache_ttl_seconds") is None:
            resolved_args["session_kv_cache_ttl_seconds"] = profile["session_kv_cache_ttl_seconds"]
        if resolved_args.get("session_kv_cache_min_prefix_tokens") is None:
            resolved_args["session_kv_cache_min_prefix_tokens"] = profile["session_kv_cache_min_prefix_tokens"]

    print(
        "Applied recommended runtime profile => "
        "device_type="
        f"{device_type}, mode={mode}, model_size_tier={model_size_tier}, "
        f"profile={json.dumps(profile, ensure_ascii=False)}"
    )
    return resolved_args


def print_effective_startup_config(args: Dict[str, Any]) -> None:
    """
    打印最终生效的启动参数明细（经过推荐值填充 + CLI 覆盖后）。
    """
    api_key = str(args.get("api_key", ""))
    if len(api_key) <= 6:
        masked_api_key = "***"
    else:
        masked_api_key = f"{api_key[:4]}***{api_key[-2:]}"

    lines = [
        "",
        "=" * 72,
        "最终生效参数明细 (startup effective config)",
        "=" * 72,
        f"mode: {args['mode']}",
        f"mode_source: {args.get('mode_source', '-')}",
        f"model_name: {args['model_name']}",
        f"model_path: {args['model_path']}",
        f"model_size_tier: {args.get('model_size_tier')}",
        f"port: {args['port']}",
        f"api_key(masked): {masked_api_key}",
        f"model_dtype: {args['model_dtype']}",
        f"max_concurrent_generations: {args['max_concurrent_generations']}",
        f"concurrency_timeout: {args['concurrency_timeout']}",
        f"use_kv_cache: {args['use_kv_cache']}",
        f"kv_cache_implementation: {args['kv_cache_implementation']}",
    ]

    if args["mode"] == "text":
        lines.extend(
            [
                f"session_kv_cache_max_entries: {args['session_kv_cache_max_entries']}",
                f"session_kv_cache_ttl_seconds: {args['session_kv_cache_ttl_seconds']}",
                f"session_kv_cache_min_prefix_tokens: {args['session_kv_cache_min_prefix_tokens']}",
            ]
        )
    else:
        lines.append(f"max_image_data_url_bytes: {args.get('max_image_data_url_bytes', DEFAULT_MAX_IMAGE_DATA_URL_BYTES)}")

    lines.extend(["=" * 72, ""])
    print("\n".join(lines))


def print_effective_runtime_engine_config(engine_name: str, info: Dict[str, Any]) -> None:
    """
    打印模型实例初始化后的运行时实际生效参数明细（包含实际 device / dtype）。
    """
    lines = [
        "",
        "=" * 72,
        f"运行时实际生效明细 ({engine_name})",
        "=" * 72,
        f"model_name: {info['model_name']}",
        f"model_path: {info['model_path']}",
        f"model_size_tier: {info['model_size_tier']}",
        f"mode_source: {info.get('mode_source', '-')}",
        f"device_type(detected): {info['device_type']}",
        f"device(actual): {info['device_actual']}",
        f"model_param_dtype(actual): {info['model_param_dtype_actual']}",
        f"model_dtype_arg(requested): {info['model_dtype_arg']}",
        f"model_loader_class: {info.get('model_loader_class', '-')}",
        f"max_concurrent_generations: {info['max_concurrent_generations']}",
        f"concurrency_timeout: {info['concurrency_timeout']}",
        f"use_kv_cache: {info['use_kv_cache']}",
        f"kv_cache_implementation: {info['kv_cache_implementation']}",
        f"cache_implementation_supported: {info['cache_implementation_supported']}",
    ]

    if info.get("session_kv_cache") is not None:
        session_kv_cache = info["session_kv_cache"]
        lines.extend(
            [
                f"session_kv_cache_max_entries: {session_kv_cache['max_entries']}",
                f"session_kv_cache_ttl_seconds: {session_kv_cache['ttl_seconds']}",
                f"session_kv_cache_min_prefix_tokens: {session_kv_cache['min_prefix_tokens']}",
            ]
        )
    if info.get("max_image_data_url_bytes") is not None:
        lines.append(f"max_image_data_url_bytes: {info['max_image_data_url_bytes']}")

    lines.extend(["=" * 72, ""])
    print("\n".join(lines))


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"


def safe_exception_message(exc: Exception, max_chars: int = MAX_STREAM_ERROR_MESSAGE_CHARS) -> str:
    message = str(exc).replace("\n", " ").strip()
    # 避免把超长 data URL/base64 原样打到日志里。
    message = re.sub(r"data:image/[^,\s]+,[A-Za-z0-9+/=_-]+", "data:image/...,(omitted)", message)
    if len(message) > max_chars:
        return message[:max_chars] + "...(truncated)"
    return message


def estimate_data_url_payload_bytes(data_url: str) -> tuple[int, str]:
    """
    返回 data URL 负载估算字节数和编码类型（base64/plain）。
    """
    header, sep, payload = data_url.partition(",")
    if sep != ",":
        raise ValueError("Invalid data URL: missing comma separator.")

    is_base64 = ";base64" in header.lower()
    if is_base64:
        payload_compact = "".join(payload.split())
        pad = len(payload_compact) - len(payload_compact.rstrip("="))
        estimated = (len(payload_compact) * 3) // 4 - min(max(pad, 0), 2)
        return max(estimated, 0), "base64"

    # 非 base64 data URL（通常是 percent-encoded 文本）；字符数作为保守上界。
    return len(payload), "plain"


def validate_vl_image_url(url: str, max_image_data_url_bytes: int) -> None:
    if not isinstance(url, str) or not url:
        raise ValueError("Invalid image_url: expected non-empty string.")
    if not url.startswith("data:"):
        return

    if not url.lower().startswith("data:image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image data URLs are supported for image_url inputs (expected 'data:image/...').",
        )

    estimated_bytes, encoding = estimate_data_url_payload_bytes(url)
    if estimated_bytes > max_image_data_url_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "Image data URL payload is too large. "
                f"Estimated decoded size={human_bytes(estimated_bytes)} "
                f"(encoding={encoding}), limit={human_bytes(max_image_data_url_bytes)}. "
                "Please resize/compress the image or use a remote URL/path instead of embedding a large base64 data URL."
            ),
        )


def build_openai_stream_final_chunk(chat_id: str, created: int, model_name: str) -> Dict[str, Any]:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }


def resolve_dtype_for_device(requested_dtype: torch.dtype, device_type: str) -> torch.dtype:
    if device_type == "cuda":
        if requested_dtype == torch.bfloat16:
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
            if not is_bf16_supported():
                print("CUDA device does not support bf16, fallback to fp16.")
                return torch.float16
        return requested_dtype

    if device_type == "mps":
        if requested_dtype == torch.bfloat16:
            print("MPS backend does not fully support bf16, fallback to fp16.")
            return torch.float16
        return requested_dtype

    # CPU 场景优先保证兼容性
    if requested_dtype != torch.float32:
        print("CPU backend fallback to fp32 for compatibility.")
    return torch.float32


def load_pretrained_with_dtype_compat(model_cls: Any, model_path: str, load_kwargs: Dict[str, Any]) -> Any:
    """
    兼容 transformers 参数演进：
    - 新版本优先使用 `dtype`
    - 旧版本若不支持 `dtype`，自动回退到 `torch_dtype`
    """
    preferred_kwargs = dict(load_kwargs)
    if "torch_dtype" in preferred_kwargs and "dtype" not in preferred_kwargs:
        preferred_kwargs["dtype"] = preferred_kwargs.pop("torch_dtype")

    try:
        return model_cls.from_pretrained(model_path, **preferred_kwargs)
    except TypeError as e:
        if "dtype" not in str(e):
            raise

        fallback_kwargs = dict(preferred_kwargs)
        if "dtype" in fallback_kwargs:
            fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
            print("Current transformers version does not accept `dtype`; fallback to `torch_dtype`.")
            return model_cls.from_pretrained(model_path, **fallback_kwargs)
        raise


def get_default_max_concurrent_generations(device_type: str) -> int:
    # 保留此函数做兼容；默认取 text 模式推荐值。
    return int(get_recommended_runtime_profile(device_type, mode="text", model_size_tier="7b")["max_concurrent_generations"])


def normalize_kv_cache_implementation(cache_impl: str) -> Optional[str]:
    normalized = cache_impl.strip().lower()
    if normalized not in SUPPORTED_KV_CACHE_IMPLEMENTATIONS:
        raise ValueError(
            f"unsupported kv cache implementation: {cache_impl}, "
            f"expected one of {sorted(SUPPORTED_KV_CACHE_IMPLEMENTATIONS)}"
        )
    return None if normalized == "auto" else normalized


def model_supports_cache_implementation(model: Any) -> bool:
    generation_config = getattr(model, "generation_config", None)
    return generation_config is not None and hasattr(generation_config, "cache_implementation")


def build_generation_kwargs(
    req: ChatCompletionRequest,
    use_kv_cache: bool,
    kv_cache_implementation: Optional[str],
    cache_implementation_supported: bool,
    device_type: Optional[str] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    if req.max_tokens <= 0:
        raise ValueError("max_tokens must be >= 1.")
    if not math.isfinite(req.temperature) or req.temperature < 0:
        raise ValueError("temperature must be a finite number and >= 0.")
    if not math.isfinite(req.top_p) or req.top_p <= 0 or req.top_p > 1:
        raise ValueError("top_p must be a finite number in (0, 1].")

    do_sample = req.temperature > 0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": req.max_tokens,
        "do_sample": do_sample,
        # 避免 fp16/mps 等场景下 logits 出现 inf/nan 导致 multinomial 报错。
        "remove_invalid_values": True,
        "renormalize_logits": True,
        "use_cache": use_kv_cache,
    }

    if do_sample:
        gen_kwargs["temperature"] = req.temperature
        gen_kwargs["top_p"] = req.top_p
        # MPS + VL（尤其较大模型）在 fp16 采样时更容易出现概率张量数值异常。
        # 先开启无效值清理和重归一化；若仍出错，用户可将 temperature 设为 0 显式关闭采样。
        if device_type == "mps" and mode == "vl":
            print(
                "MPS+VL sampling enabled (temperature > 0). "
                "Stability guards active: remove_invalid_values=True, renormalize_logits=True. "
                "If probability tensor NaN/Inf persists, set temperature=0 to disable sampling."
            )

    if use_kv_cache and kv_cache_implementation is not None:
        if cache_implementation_supported:
            gen_kwargs["cache_implementation"] = kv_cache_implementation
        else:
            print(
                f"cache_implementation='{kv_cache_implementation}' is ignored because "
                "the current transformers version/model does not support it."
            )
    return gen_kwargs


def is_probability_tensor_invalid_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "probability tensor contains either" in message
        and ("nan" in message or "inf" in message or "element < 0" in message)
    )


def should_retry_with_greedy_fallback(exc: Exception, gen_kwargs: Dict[str, Any]) -> bool:
    return bool(gen_kwargs.get("do_sample")) and is_probability_tensor_invalid_error(exc)


def build_greedy_fallback_gen_kwargs(gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    fallback_kwargs = dict(gen_kwargs)
    fallback_kwargs["do_sample"] = False
    fallback_kwargs.pop("temperature", None)
    fallback_kwargs.pop("top_p", None)
    # 有些调用侧未来可能补 top_k；贪心模式下不需要。
    fallback_kwargs.pop("top_k", None)
    return fallback_kwargs


class AbortEventStoppingCriteria(StoppingCriteria):
    """
    通过线程事件中止 generate（用于客户端断开连接后的流式请求）。
    """

    def __init__(self, abort_event: Event) -> None:
        super().__init__()
        self.abort_event = abort_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        return self.abort_event.is_set()


def attach_abort_stopping_criteria(gen_kwargs: Dict[str, Any], abort_event: Event) -> Dict[str, Any]:
    patched_kwargs = dict(gen_kwargs)
    existing = patched_kwargs.get("stopping_criteria")

    if existing is None:
        criteria_items: List[StoppingCriteria] = []
    elif isinstance(existing, StoppingCriteriaList):
        criteria_items = list(existing)
    else:
        try:
            criteria_items = list(existing)
        except TypeError:
            criteria_items = [existing]

    criteria_items.append(AbortEventStoppingCriteria(abort_event))
    patched_kwargs["stopping_criteria"] = StoppingCriteriaList(criteria_items)
    return patched_kwargs


def str2bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


class InferenceConcurrencyLimiter:
    def __init__(self, max_concurrent_generations: int, timeout_seconds: float) -> None:
        if max_concurrent_generations <= 0:
            raise ValueError("max_concurrent_generations must be >= 1")
        if timeout_seconds <= 0:
            raise ValueError("concurrency_timeout must be > 0")

        self.max_concurrent_generations = max_concurrent_generations
        self.timeout_seconds = timeout_seconds
        self._semaphore = BoundedSemaphore(max_concurrent_generations)
        self._lock = Lock()
        self._active_generations = 0
        self._waiting_requests = 0

    def acquire(self, request_tag: Optional[str] = None) -> None:
        tag = request_tag or "-"
        with self._lock:
            self._waiting_requests += 1
            waiting_snapshot = self._waiting_requests
            active_snapshot = self._active_generations
        print(
            f"[queue-enter] tag={tag} active={active_snapshot} "
            f"waiting={waiting_snapshot} limit={self.max_concurrent_generations}"
        )

        t0 = time.time()
        acquired = self._semaphore.acquire(timeout=self.timeout_seconds)
        wait_ms = int((time.time() - t0) * 1000)

        with self._lock:
            self._waiting_requests = max(0, self._waiting_requests - 1)
            if acquired:
                self._active_generations += 1
            waiting_snapshot = self._waiting_requests
            active_snapshot = self._active_generations

        if not acquired:
            print(
                f"[queue-timeout] tag={tag} wait_ms={wait_ms} active={active_snapshot} "
                f"waiting={waiting_snapshot} limit={self.max_concurrent_generations}"
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    "Model is busy, request queue timeout reached. "
                    f"max_concurrent_generations={self.max_concurrent_generations}"
                ),
            )

        print(
            f"[slot-acquired] tag={tag} wait_ms={wait_ms} active={active_snapshot} "
            f"waiting={waiting_snapshot} limit={self.max_concurrent_generations}"
        )

    def release(self, request_tag: Optional[str] = None) -> None:
        tag = request_tag or "-"
        self._semaphore.release()
        with self._lock:
            self._active_generations = max(0, self._active_generations - 1)
            active_snapshot = self._active_generations
            waiting_snapshot = self._waiting_requests
        print(
            f"[slot-released] tag={tag} active={active_snapshot} "
            f"waiting={waiting_snapshot} limit={self.max_concurrent_generations}"
        )

    @contextmanager
    def hold(self, request_tag: Optional[str] = None) -> Iterator[None]:
        self.acquire(request_tag=request_tag)
        try:
            yield
        finally:
            self.release(request_tag=request_tag)


class SessionKVCacheStore:
    """
    仅用于 text 模型的会话 KV cache（内存 LRU + TTL）。
    - key: session_id
    - value: {input_ids, past_key_values}
    """

    def __init__(
        self,
        max_entries: int,
        ttl_seconds: int,
        min_prefix_tokens: int,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("session_kv_cache_max_entries must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("session_kv_cache_ttl_seconds must be >= 1")
        if min_prefix_tokens <= 0:
            raise ValueError("session_kv_cache_min_prefix_tokens must be >= 1")

        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.min_prefix_tokens = min_prefix_tokens
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()

    def _evict_locked(self) -> None:
        now = time.time()
        expired_keys = [
            session_id
            for session_id, item in self._cache.items()
            if now - item["updated_at"] > self.ttl_seconds
        ]
        for session_id in expired_keys:
            self._cache.pop(session_id, None)

        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def get(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None

        with self._lock:
            self._evict_locked()
            item = self._cache.get(session_id)
            if item is None:
                return None
            self._cache.move_to_end(session_id)
            item["updated_at"] = time.time()
            return {
                "input_ids": item["input_ids"],
                "past_key_values": item["past_key_values"],
            }

    def put(self, session_id: Optional[str], input_ids: torch.Tensor, past_key_values: Any) -> None:
        if not session_id or past_key_values is None:
            return

        with self._lock:
            self._cache[session_id] = {
                "input_ids": input_ids.detach(),
                "past_key_values": past_key_values,
                "updated_at": time.time(),
            }
            self._cache.move_to_end(session_id)
            self._evict_locked()


def build_cached_model_inputs(
    full_inputs: Dict[str, torch.Tensor],
    cache_item: Optional[Dict[str, Any]],
    min_prefix_tokens: int,
) -> tuple[Dict[str, Any], int]:
    if cache_item is None:
        return full_inputs, 0

    input_ids: torch.Tensor = full_inputs["input_ids"]
    cached_ids: torch.Tensor = cache_item["input_ids"]

    if input_ids.ndim != 2 or cached_ids.ndim != 2:
        return full_inputs, 0
    if input_ids.shape[0] != cached_ids.shape[0]:
        return full_inputs, 0
    if input_ids.device != cached_ids.device:
        return full_inputs, 0

    cached_prefix_len = int(cached_ids.shape[1])
    prompt_len = int(input_ids.shape[1])

    if cached_prefix_len < min_prefix_tokens or cached_prefix_len >= prompt_len:
        return full_inputs, 0
    if not torch.equal(input_ids[:, :cached_prefix_len], cached_ids):
        return full_inputs, 0

    model_inputs: Dict[str, Any] = dict(full_inputs)
    model_inputs["input_ids"] = input_ids[:, cached_prefix_len:]
    model_inputs["past_key_values"] = cache_item["past_key_values"]
    return model_inputs, cached_prefix_len


def restore_full_sequences(
    generated_sequences: torch.Tensor,
    cache_item: Optional[Dict[str, Any]],
    cached_prefix_len: int,
) -> torch.Tensor:
    if cache_item is None or cached_prefix_len <= 0:
        return generated_sequences

    cached_ids: torch.Tensor = cache_item["input_ids"]
    prefix_ids = cached_ids[:, :cached_prefix_len]
    if generated_sequences.shape[1] >= cached_prefix_len:
        if torch.equal(generated_sequences[:, :cached_prefix_len], prefix_ids):
            return generated_sequences
    return torch.cat([prefix_ids, generated_sequences], dim=1)


def openai_to_qwenvl_messages(
    messages: List[ChatMessage],
    max_image_data_url_bytes: int = DEFAULT_MAX_IMAGE_DATA_URL_BYTES,
) -> List[Dict[str, Any]]:
    """
    将 OpenAI 风格 messages 转成 Qwen/GUI-Owl 期望的多模态格式：
    [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "<url-or-path>"},
          {"type": "text",  "text": "xxx"}
        ]
      },
      ...
    ]
    """
    qwen_messages: List[Dict[str, Any]] = []

    for m in messages:
        role = m.role
        content = m.content

        # 纯字符串内容，转成单一 text 块
        if isinstance(content, str):
            qwen_messages.append(
                {
                    "role": role,
                    "content": [
                        {"type": "text", "text": content}
                    ],
                }
            )
            continue

        # 列表形式，可能是 text / image_url 混合
        qwen_content: List[Dict[str, Any]] = []
        for part in content:
            if isinstance(part, ChatMessagePartText):
                qwen_content.append({"type": "text", "text": part.text})
            elif isinstance(part, ChatMessagePartImageURL):
                url = part.image_url.get("url")
                if url:
                    validate_vl_image_url(url, max_image_data_url_bytes=max_image_data_url_bytes)
                    qwen_content.append({"type": "image", "image": url})
        # 如果一个 message 里只给了图片，没有 text，可以根据需要追加一个空 text
        if not qwen_content:
            qwen_content = [{"type": "text", "text": ""}]

        qwen_messages.append({"role": role, "content": qwen_content})

    return qwen_messages


def openai_to_qwen_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    qwen_messages: List[Dict[str, Any]] = []

    for m in messages:
        role = m.role
        content = m.content

        if isinstance(content, str):
            qwen_messages.append({"role": role, "content": content})
            continue

        texts: List[str] = []
        for part in content:
            if isinstance(part, ChatMessagePartText):
                texts.append(part.text)
            elif isinstance(part, ChatMessagePartImageURL):
                raise ValueError("This text-only Qwen model does not support image inputs.")

        qwen_messages.append({"role": role, "content": "".join(texts)})

    return qwen_messages


class DeployQwenVL(object):

    def __init__(self,
                 model_path: str,
                 model_name: str,
                 api_key: str,
                 model_dtype: str = "bf16",
                 max_concurrent_generations: Optional[int] = None,
                 concurrency_timeout: float = 30.0,
                 use_kv_cache: bool = True,
                 kv_cache_implementation: str = "auto",
                 model_size_tier: Optional[str] = None,
                 mode_source: Optional[str] = None,
                 max_image_data_url_bytes: int = DEFAULT_MAX_IMAGE_DATA_URL_BYTES) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.mode_source = mode_source or "unknown"
        self.use_kv_cache = use_kv_cache
        self.kv_cache_implementation = normalize_kv_cache_implementation(kv_cache_implementation)
        if max_image_data_url_bytes <= 0:
            raise ValueError("max_image_data_url_bytes must be >= 1")
        self.max_image_data_url_bytes = int(max_image_data_url_bytes)
        self.model_size_tier = (
            normalize_model_size_tier(model_size_tier)
            if model_size_tier
            else infer_model_size_tier(model_name, model_path)
        )

        self.model, self.processor, self.device, self.device_type = self.load_model(model_path, model_dtype)
        self.cache_implementation_supported = model_supports_cache_implementation(self.model)

        if max_concurrent_generations is None:
            max_concurrent_generations = int(
                get_recommended_runtime_profile(
                    self.device_type,
                    mode="vl",
                    model_size_tier=self.model_size_tier,
                )["max_concurrent_generations"]
            )
        self.concurrency_limiter = InferenceConcurrencyLimiter(
            max_concurrent_generations=max_concurrent_generations,
            timeout_seconds=concurrency_timeout,
        )

        primary_param = next(self.model.parameters())
        print_effective_runtime_engine_config(
            engine_name="Qwen-VL",
            info={
                "model_name": self.model_name,
                "model_path": model_path,
                "model_size_tier": self.model_size_tier,
                "mode_source": self.mode_source,
                "device_type": self.device_type,
                "device_actual": str(primary_param.device),
                "model_param_dtype_actual": str(primary_param.dtype),
                "model_dtype_arg": model_dtype,
                "model_loader_class": _VL_MODEL_CLASS_NAME,
                "max_concurrent_generations": self.concurrency_limiter.max_concurrent_generations,
                "concurrency_timeout": self.concurrency_limiter.timeout_seconds,
                "use_kv_cache": self.use_kv_cache,
                "kv_cache_implementation": self.kv_cache_implementation or "auto",
                "cache_implementation_supported": self.cache_implementation_supported,
                "max_image_data_url_bytes": self.max_image_data_url_bytes,
                "session_kv_cache": None,
            },
        )
    
    def load_model(self,
                   model_path: str,
                   model_dtype: str) -> tuple[Any, AutoProcessor, torch.device, str]:
        print(f"Loading model: {self.model_name}")

        requested_dtype = parse_model_dtype(model_dtype)
        device_type = detect_device_type()
        resolved_dtype = resolve_dtype_for_device(requested_dtype, device_type)

        print(
            f"Detected device_type={device_type}, "
            f"requested_dtype={requested_dtype}, resolved_dtype={resolved_dtype}"
        )

        load_kwargs: Dict[str, Any] = {
            "dtype": resolved_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if device_type == "cuda":
            # CUDA 场景支持单卡/多卡自动映射。
            load_kwargs["device_map"] = "auto"

        model = load_pretrained_with_dtype_compat(AutoModelForVL, model_path, load_kwargs)

        if device_type in {"mps", "cpu"}:
            model = model.to(torch.device(device_type))

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        model.eval()
        device = next(model.parameters()).device

        print(f"Model loaded on {device}")
        return model, processor, device, device_type
    
    def chat_completion(self, 
                        req: ChatCompletionRequest) -> Any:
        """
        OpenAI 风格的 ChatCompletions：
        - stream = False: 一次性返回完整结果
        - stream = True: 以 text/event-stream 方式流式返回 OpenAI 样式的 chunk
        """
        if req.model and req.model != self.model_name:
            raise ValueError(
                f"Unsupported model '{req.model}'. This engine is '{self.model_name}'."
            )
        
        t_start = time.time()
        request_tag = f"vl-{'stream' if req.stream else 'sync'}-{uuid.uuid4().hex[:8]}"

        # 1. 转成 Qwen 多模态 messages
        qwen_messages = openai_to_qwenvl_messages(
            req.messages,
            max_image_data_url_bytes=self.max_image_data_url_bytes,
        )

        # 2. 用 AutoProcessor 构造文本 prompt + 视觉输入
        prompt_text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)

        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 公共的生成参数
        gen_kwargs = build_generation_kwargs(
            req=req,
            use_kv_cache=self.use_kv_cache,
            kv_cache_implementation=self.kv_cache_implementation,
            cache_implementation_supported=self.cache_implementation_supported,
            device_type=self.device_type,
            mode="vl",
        )

        # ======================
        #  A. 非流式：保持你原来的逻辑
        # ======================
        if not req.stream:
            with self.concurrency_limiter.hold(request_tag=request_tag):
                with torch.inference_mode():
                    try:
                        generated_ids = self.model.generate(**inputs, **gen_kwargs)
                    except Exception as e:
                        if not should_retry_with_greedy_fallback(e, gen_kwargs):
                            raise
                        print(
                            f"[generate-fallback][{self.model_name}] "
                            f"{type(e).__name__}: {safe_exception_message(e)} | "
                            "retrying once with greedy decoding (non-stream, current request only)."
                        )
                        fallback_gen_kwargs = build_greedy_fallback_gen_kwargs(gen_kwargs)
                        generated_ids = self.model.generate(**inputs, **fallback_gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answer = output_texts[0]

            prompt_tokens = int(inputs.input_ids.shape[1])
            completion_tokens = int(generated_ids_trimmed[0].shape[0])
            total_tokens = int(generated_ids[0].shape[0])

            resp = ChatCompletionResponse(
                id="chatcmpl-" + uuid.uuid4().hex,
                object="chat.completion",
                created=int(t_start),
                model=self.model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(
                            role="assistant",
                            content=answer,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
            )
            return resp

        # ======================
        #  B. 流式：TextIteratorStreamer + SSE
        # ======================

        # 使用 transformers 自带的流式工具
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # 将 streamer 塞到 generate 参数里
        abort_event = Event()
        stream_gen_kwargs = dict(gen_kwargs)
        stream_gen_kwargs["streamer"] = streamer
        stream_gen_kwargs = attach_abort_stopping_criteria(stream_gen_kwargs, abort_event)

        self.concurrency_limiter.acquire(request_tag=request_tag)
        generation_error: List[Exception] = []
        stream_has_emitted_token = [False]

        def generate_in_background():
            try:
                with torch.inference_mode():
                    self.model.generate(**inputs, **stream_gen_kwargs)
            except Exception as e:
                if (
                    not stream_has_emitted_token[0]
                    and should_retry_with_greedy_fallback(e, stream_gen_kwargs)
                ):
                    try:
                        print(
                            f"[generate-fallback][{self.model_name}] "
                            f"{type(e).__name__}: {safe_exception_message(e)} | "
                            "retrying once with greedy decoding (stream, no token emitted yet)."
                        )
                        fallback_stream_gen_kwargs = build_greedy_fallback_gen_kwargs(stream_gen_kwargs)
                        with torch.inference_mode():
                            self.model.generate(**inputs, **fallback_stream_gen_kwargs)
                        return
                    except Exception as retry_e:
                        e = retry_e
                generation_error.append(e)
                streamer.on_finalized_text("", stream_end=True)

        # 后台线程跑 generate，当前线程消费 streamer
        thread = Thread(target=generate_in_background, daemon=True)
        try:
            thread.start()
        except Exception:
            self.concurrency_limiter.release(request_tag=request_tag)
            raise

        def event_stream():
            chat_id = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())

            # OpenAI 风格：每个 chunk 是一个 JSON，包在 "data: ...\n\n" 里
            # 第一块可以带上 role，后续只追加 content
            first_chunk = True

            try:
                for piece in streamer:
                    if not piece:
                        continue
                    stream_has_emitted_token[0] = True

                    delta: Dict[str, Any] = {}
                    if first_chunk:
                        # 第一块返回 role + 第一截内容
                        delta = {"role": "assistant", "content": piece}
                        first_chunk = False
                    else:
                        delta = {"content": piece}

                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                            }
                        ],
                    }

                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                if generation_error:
                    raise generation_error[0]

                # 最后补一个 finish_reason=stop 的 chunk
                final_chunk = build_openai_stream_final_chunk(chat_id, created, self.model_name)
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                # OpenAI 习惯再补一个 [DONE]
                yield "data: [DONE]\n\n"
            except GeneratorExit:
                abort_event.set()
                print(f"[client-disconnect][{self.model_name}] Stream client disconnected, aborting generation.")
                raise
            except asyncio.CancelledError:
                abort_event.set()
                print(f"[client-disconnect][{self.model_name}] Stream task cancelled, aborting generation.")
                raise
            except Exception as e:
                print(
                    f"[stream-error][{self.model_name}] "
                    f"{type(e).__name__}: {safe_exception_message(e)}"
                )
                # 为提升 OpenAI 兼容性，不发送自定义 error SSE 结构，只发送标准终止 chunk + [DONE]。
                final_chunk = build_openai_stream_final_chunk(chat_id, created, self.model_name)
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if abort_event.is_set():
                    streamer.on_finalized_text("", stream_end=True)
                thread.join()
                self.concurrency_limiter.release(request_tag=request_tag)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

class DeployQwen(object):

    def __init__(self,
                 model_path: str,
                 model_name: str,
                 api_key: str,
                 model_dtype: str = "bf16",
                 max_concurrent_generations: Optional[int] = None,
                 concurrency_timeout: float = 30.0,
                 use_kv_cache: bool = True,
                 kv_cache_implementation: str = "auto",
                 session_kv_cache_max_entries: int = 128,
                 session_kv_cache_ttl_seconds: int = 1800,
                 session_kv_cache_min_prefix_tokens: int = 32,
                 model_size_tier: Optional[str] = None,
                 mode_source: Optional[str] = None) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.mode_source = mode_source or "unknown"
        self.use_kv_cache = use_kv_cache
        self.kv_cache_implementation = normalize_kv_cache_implementation(kv_cache_implementation)
        self.model_size_tier = (
            normalize_model_size_tier(model_size_tier)
            if model_size_tier
            else infer_model_size_tier(model_name, model_path)
        )
        self.session_kv_cache = SessionKVCacheStore(
            max_entries=session_kv_cache_max_entries,
            ttl_seconds=session_kv_cache_ttl_seconds,
            min_prefix_tokens=session_kv_cache_min_prefix_tokens,
        )

        self.model, self.tokenizer, self.device, self.device_type = self.load_model(model_path, model_dtype)
        self.cache_implementation_supported = model_supports_cache_implementation(self.model)

        if max_concurrent_generations is None:
            max_concurrent_generations = int(
                get_recommended_runtime_profile(
                    self.device_type,
                    mode="text",
                    model_size_tier=self.model_size_tier,
                )["max_concurrent_generations"]
            )
        self.concurrency_limiter = InferenceConcurrencyLimiter(
            max_concurrent_generations=max_concurrent_generations,
            timeout_seconds=concurrency_timeout,
        )

        primary_param = next(self.model.parameters())
        print_effective_runtime_engine_config(
            engine_name="Qwen-text",
            info={
                "model_name": self.model_name,
                "model_path": model_path,
                "model_size_tier": self.model_size_tier,
                "mode_source": self.mode_source,
                "device_type": self.device_type,
                "device_actual": str(primary_param.device),
                "model_param_dtype_actual": str(primary_param.dtype),
                "model_dtype_arg": model_dtype,
                "model_loader_class": AutoModelForCausalLM.__name__,
                "max_concurrent_generations": self.concurrency_limiter.max_concurrent_generations,
                "concurrency_timeout": self.concurrency_limiter.timeout_seconds,
                "use_kv_cache": self.use_kv_cache,
                "kv_cache_implementation": self.kv_cache_implementation or "auto",
                "cache_implementation_supported": self.cache_implementation_supported,
                "session_kv_cache": {
                    "max_entries": self.session_kv_cache.max_entries,
                    "ttl_seconds": self.session_kv_cache.ttl_seconds,
                    "min_prefix_tokens": self.session_kv_cache.min_prefix_tokens,
                },
            },
        )

    def load_model(self,
                   model_path: str,
                   model_dtype: str) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device, str]:
        print(f"Loading text-only model: {self.model_name}")

        requested_dtype = parse_model_dtype(model_dtype)
        device_type = detect_device_type()
        resolved_dtype = resolve_dtype_for_device(requested_dtype, device_type)

        print(
            f"Detected device_type={device_type}, "
            f"requested_dtype={requested_dtype}, resolved_dtype={resolved_dtype}"
        )

        load_kwargs: Dict[str, Any] = {
            "dtype": resolved_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if device_type == "cuda":
            load_kwargs["device_map"] = "auto"

        model = load_pretrained_with_dtype_compat(AutoModelForCausalLM, model_path, load_kwargs)

        if device_type in {"mps", "cpu"}:
            model = model.to(torch.device(device_type))

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        model.eval()
        device = next(model.parameters()).device
        print(f"Model loaded on {device}")
        return model, tokenizer, device, device_type

    def chat_completion(self, 
                        req: ChatCompletionRequest) -> Any:
        if req.model and req.model != self.model_name:
            raise ValueError(f"Unsupported model '{req.model}'. This engine is '{self.model_name}'.")

        t_start = time.time()
        request_tag = f"text-{'stream' if req.stream else 'sync'}-{uuid.uuid4().hex[:8]}"

        # 1) OpenAI -> Qwen(text) messages
        qwen_messages = openai_to_qwen_messages(req.messages)

        # 2) prompt（纯文本用 tokenizer.apply_chat_template）
        prompt_text = self.tokenizer.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 3) tokenize
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        ).to(self.device)
        full_model_inputs: Dict[str, torch.Tensor] = {
            k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }

        cache_item: Optional[Dict[str, Any]] = None
        cached_prefix_len = 0
        model_inputs: Dict[str, Any] = full_model_inputs
        if self.use_kv_cache and req.session_id:
            cache_item = self.session_kv_cache.get(req.session_id)
            model_inputs, cached_prefix_len = build_cached_model_inputs(
                full_inputs=full_model_inputs,
                cache_item=cache_item,
                min_prefix_tokens=self.session_kv_cache.min_prefix_tokens,
            )

        gen_kwargs = build_generation_kwargs(
            req=req,
            use_kv_cache=self.use_kv_cache,
            kv_cache_implementation=self.kv_cache_implementation,
            cache_implementation_supported=self.cache_implementation_supported,
            device_type=self.device_type,
            mode="text",
        )

        # ========== 非流式 ==========
        if not req.stream:
            with self.concurrency_limiter.hold(request_tag=request_tag):
                with torch.inference_mode():
                    try:
                        generation_output = self.model.generate(
                            **model_inputs,
                            **gen_kwargs,
                            return_dict_in_generate=True,
                        )
                    except Exception as e:
                        if not should_retry_with_greedy_fallback(e, gen_kwargs):
                            raise
                        print(
                            f"[generate-fallback][{self.model_name}] "
                            f"{type(e).__name__}: {safe_exception_message(e)} | "
                            "retrying once with greedy decoding (non-stream, current request only)."
                        )
                        fallback_gen_kwargs = build_greedy_fallback_gen_kwargs(gen_kwargs)
                        generation_output = self.model.generate(
                            **model_inputs,
                            **fallback_gen_kwargs,
                            return_dict_in_generate=True,
                        )

            if isinstance(generation_output, torch.Tensor):
                generated_sequences = generation_output
                generated_past_key_values = None
            else:
                generated_sequences = generation_output.sequences
                generated_past_key_values = getattr(generation_output, "past_key_values", None)

            full_sequences = restore_full_sequences(generated_sequences, cache_item, cached_prefix_len)
            prompt_len = int(inputs.input_ids.shape[1])
            gen_trimmed = full_sequences[:, prompt_len:]

            answer = self.tokenizer.batch_decode(
                gen_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            if self.use_kv_cache and req.session_id and generated_past_key_values is not None:
                self.session_kv_cache.put(req.session_id, full_sequences, generated_past_key_values)

            prompt_tokens = int(inputs.input_ids.shape[1])
            completion_tokens = int(gen_trimmed.shape[1])
            total_tokens = int(full_sequences.shape[1])

            resp = ChatCompletionResponse(
                id="chatcmpl-" + uuid.uuid4().hex,
                object="chat.completion",
                created=int(t_start),
                model=self.model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(role="assistant", content=answer),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
            )
            return resp

        # ========== 流式 ==========
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        stream_gen_kwargs = dict(gen_kwargs)
        abort_event = Event()
        stream_gen_kwargs["streamer"] = streamer
        stream_gen_kwargs["return_dict_in_generate"] = True
        stream_gen_kwargs = attach_abort_stopping_criteria(stream_gen_kwargs, abort_event)

        self.concurrency_limiter.acquire(request_tag=request_tag)
        generation_error: List[Exception] = []
        generation_output_list: List[Any] = []
        stream_has_emitted_token = [False]

        def generate_in_background():
            try:
                with torch.inference_mode():
                    generation_output = self.model.generate(**model_inputs, **stream_gen_kwargs)
                    generation_output_list.append(generation_output)
            except Exception as e:
                if (
                    not stream_has_emitted_token[0]
                    and should_retry_with_greedy_fallback(e, stream_gen_kwargs)
                ):
                    try:
                        print(
                            f"[generate-fallback][{self.model_name}] "
                            f"{type(e).__name__}: {safe_exception_message(e)} | "
                            "retrying once with greedy decoding (stream, no token emitted yet)."
                        )
                        fallback_stream_gen_kwargs = build_greedy_fallback_gen_kwargs(stream_gen_kwargs)
                        with torch.inference_mode():
                            generation_output = self.model.generate(**model_inputs, **fallback_stream_gen_kwargs)
                        generation_output_list.append(generation_output)
                        return
                    except Exception as retry_e:
                        e = retry_e
                generation_error.append(e)
                streamer.on_finalized_text("", stream_end=True)

        thread = Thread(target=generate_in_background, daemon=True)
        try:
            thread.start()
        except Exception:
            self.concurrency_limiter.release(request_tag=request_tag)
            raise

        def event_stream():
            chat_id = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            first_chunk = True

            try:
                for piece in streamer:
                    if not piece:
                        continue
                    stream_has_emitted_token[0] = True

                    if first_chunk:
                        delta = {"role": "assistant", "content": piece}
                        first_chunk = False
                    else:
                        delta = {"content": piece}

                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                if generation_error:
                    raise generation_error[0]

                if self.use_kv_cache and req.session_id and generation_output_list:
                    stream_output = generation_output_list[0]
                    if isinstance(stream_output, torch.Tensor):
                        stream_sequences = stream_output
                        stream_past_key_values = None
                    else:
                        stream_sequences = stream_output.sequences
                        stream_past_key_values = getattr(stream_output, "past_key_values", None)

                    if stream_past_key_values is not None:
                        full_sequences = restore_full_sequences(stream_sequences, cache_item, cached_prefix_len)
                        self.session_kv_cache.put(req.session_id, full_sequences, stream_past_key_values)

                final_chunk = build_openai_stream_final_chunk(chat_id, created, self.model_name)
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except GeneratorExit:
                abort_event.set()
                print(f"[client-disconnect][{self.model_name}] Stream client disconnected, aborting generation.")
                raise
            except asyncio.CancelledError:
                abort_event.set()
                print(f"[client-disconnect][{self.model_name}] Stream task cancelled, aborting generation.")
                raise
            except Exception as e:
                print(
                    f"[stream-error][{self.model_name}] "
                    f"{type(e).__name__}: {safe_exception_message(e)}"
                )
                # 为提升 OpenAI 兼容性，不发送自定义 error SSE 结构，只发送标准终止 chunk + [DONE]。
                final_chunk = build_openai_stream_final_chunk(chat_id, created, self.model_name)
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if abort_event.is_set():
                    streamer.on_finalized_text("", stream_end=True)
                thread.join()
                self.concurrency_limiter.release(request_tag=request_tag)

        return StreamingResponse(event_stream(), media_type="text/event-stream")


app = FastAPI(title="OpenAI-compatible API")
qwen_engine: Optional[Union[DeployQwenVL, DeployQwen]] = None


@app.post("/v1/chat/completions")
def create_chat_completion(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None)
) -> Any:
    assert qwen_engine is not None, "Model not initialized"

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    api_key = authorization.removeprefix("Bearer ").strip()
    if api_key != qwen_engine.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        return qwen_engine.chat_completion(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def main(args: dict) -> None:
    global qwen_engine
    args = apply_recommended_runtime_args(args)
    print_effective_startup_config(args)

    common_kwargs = {
        "model_path": args["model_path"],
        "model_name": args["model_name"],
        "api_key": args["api_key"],
        "model_size_tier": args["model_size_tier"],
        "mode_source": args.get("mode_source"),
        "model_dtype": args["model_dtype"],
        "max_concurrent_generations": args["max_concurrent_generations"],
        "concurrency_timeout": args["concurrency_timeout"],
        "use_kv_cache": args["use_kv_cache"],
        "kv_cache_implementation": args["kv_cache_implementation"],
    }

    if args["mode"] == "vl":
        qwen_engine = DeployQwenVL(
            **common_kwargs,
            max_image_data_url_bytes=args["max_image_data_url_bytes"],
        )
    else:
        qwen_engine = DeployQwen(
            **common_kwargs,
            session_kv_cache_max_entries=args["session_kv_cache_max_entries"],
            session_kv_cache_ttl_seconds=args["session_kv_cache_ttl_seconds"],
            session_kv_cache_min_prefix_tokens=args["session_kv_cache_min_prefix_tokens"],
        )
    
    uvicorn.run(app, host="0.0.0.0", port=args["port"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run qwen deployment")
    parser.add_argument(
        "--model_path", 
        default="Qwen/Qwen3-VL-4B-Instruct",
        type=str
    )
    parser.add_argument(
        "--model_name", 
        default="Qwen3-VL-4B-Instruct", 
        type=str
    )
    parser.add_argument(
        "--model_size_tier",
        default="auto",
        choices=MODEL_SIZE_TIER_CLI_CHOICES,
        type=str,
        help="Recommended profile size tier. 'auto' infers from model_name/model_path (<=14B exact buckets); valid tiers depend on mode.",
    )
    parser.add_argument(
        "--api_key", 
        default="sk-1234", 
        type=str
    )
    parser.add_argument(
        "--model_dtype", 
        default="auto", 
        choices=["auto", "bf16", "fp16", "fp32"],
        type=str
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "vl", "text"],
        type=str,
        help="Model mode. 'auto' infers from model config.json/AutoConfig first, then falls back to name/path heuristic.",
    )
    parser.add_argument(
        "--port", 
        default=1234, 
        type=int
    )
    parser.add_argument(
        "--max_concurrent_generations",
        default=None,
        type=int,
        help="Maximum parallel generation jobs. Default uses recommended device+mode+size-tier profile.",
    )
    parser.add_argument(
        "--concurrency_timeout",
        default=None,
        type=float,
        help="How long a request waits (seconds) when concurrency slots are full. Default uses recommended profile.",
    )
    parser.add_argument(
        "--max_image_data_url_bytes",
        default=DEFAULT_MAX_IMAGE_DATA_URL_BYTES,
        type=int,
        help="VL mode only: maximum decoded payload size allowed for data:image/... URLs (bytes).",
    )
    parser.add_argument(
        "--use_kv_cache",
        default=None,
        type=str2bool,
        help="Whether to enable KV cache during generation. Default uses recommended profile.",
    )
    parser.add_argument(
        "--kv_cache_implementation",
        default="recommended",
        choices=KV_CACHE_CLI_CHOICES,
        type=str,
        help="KV cache backend strategy. 'recommended' applies device+mode+size-tier profile. 'auto' uses transformers default behavior.",
    )
    parser.add_argument(
        "--session_kv_cache_max_entries",
        default=None,
        type=int,
        help="Text mode only: max session KV cache entries. Default uses recommended profile.",
    )
    parser.add_argument(
        "--session_kv_cache_ttl_seconds",
        default=None,
        type=int,
        help="Text mode only: KV cache entry TTL in seconds. Default uses recommended profile.",
    )
    parser.add_argument(
        "--session_kv_cache_min_prefix_tokens",
        default=None,
        type=int,
        help="Text mode only: minimum shared prefix length to reuse session KV cache. Default uses recommended profile.",
    )
    args = vars(parser.parse_args())
    main(args)
