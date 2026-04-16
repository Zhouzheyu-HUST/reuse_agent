# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import random
import time
from typing import Literal, Optional

import requests


class NonRetriableError(RuntimeError):
    pass


class PaddleOcrClient(object):
    def __init__(
        self,
        model: str,
        endpoint: str,
        api_key: str,
        timeout: float = 60.0,
        max_retry: int = 3,
        sleep_sec: float = 2.0,
        backoff_base_sec: float = 0.5,
        backoff_cap_sec: float = 8.0,
        retry_on_5xx: bool = True,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.max_retry = max_retry
        self.sleep_sec = sleep_sec
        self.backoff_base_sec = backoff_base_sec
        self.backoff_cap_sec = backoff_cap_sec
        self.retry_on_5xx = retry_on_5xx

    def _compute_backoff_seconds(self, attempt_index: int) -> float:
        exp_delay = self.backoff_base_sec * (2 ** attempt_index)
        bounded = min(self.backoff_cap_sec, exp_delay)
        jitter = random.uniform(0.0, max(0.1, bounded * 0.2))
        return max(0.0, bounded + jitter)

    @staticmethod
    def _parse_retry_after_seconds(headers: requests.structures.CaseInsensitiveDict) -> Optional[float]:
        retry_after = headers.get("Retry-After")
        if retry_after is None:
            return None
        retry_after = retry_after.strip()
        if not retry_after:
            return None
        try:
            parsed = float(retry_after)
            if parsed >= 0:
                return parsed
        except Exception:
            return None
        return None

    def infer(
        self,
        image_file: str,
        file_type: Literal["path", "base64"],
    ) -> list[str]:
        """
        调用本地 PaddleOCR 服务，返回识别出来的文本列表。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = {
            "model": self.model,
            "image_file": image_file,
            "file_type": file_type
        }

        last_err = None
        for attempt in range(self.max_retry):
            try:
                resp = requests.post(
                    self.endpoint,
                    json=data,
                    headers=headers,
                    timeout=self.timeout,
                )
                status_code = resp.status_code

                if status_code == 200:
                    resp_json = resp.json()
                    return resp_json.get("response", [])

                if status_code == 429:
                    if attempt >= self.max_retry - 1:
                        raise RuntimeError(f"HTTP 429: {resp.text}")
                    retry_after_sec = self._parse_retry_after_seconds(resp.headers)
                    delay = retry_after_sec if retry_after_sec is not None else self._compute_backoff_seconds(attempt)
                    time.sleep(delay)
                    continue

                if 500 <= status_code < 600:
                    if not self.retry_on_5xx or attempt >= self.max_retry - 1:
                        raise RuntimeError(f"HTTP {status_code}: {resp.text}")
                    delay = self._compute_backoff_seconds(attempt)
                    time.sleep(delay)
                    continue

                # 非 429 的 4xx 请求通常是不可重试错误，直接返回
                raise NonRetriableError(f"HTTP {status_code}: {resp.text}")

            except NonRetriableError:
                raise
            except Exception as e:
                last_err = e
                if attempt >= self.max_retry - 1:
                    break
                time.sleep(max(self.sleep_sec, self._compute_backoff_seconds(attempt)))

        raise last_err
