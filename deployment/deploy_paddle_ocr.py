# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import argparse
import base64
import gc
import math
import os
import platform
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, Iterator, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
import numpy as np
from pydantic import BaseModel, Field
import uvicorn

try:
    import torch
except Exception:
    torch = None

_cv2_import_error: Optional[Exception] = None
try:
    import cv2
except Exception as e:
    cv2 = None
    _cv2_import_error = e

_paddle_import_error: Optional[Exception] = None
try:
    import paddle
except Exception as e:
    paddle = None
    _paddle_import_error = e

_paddleocr_import_error: Optional[Exception] = None
try:
    from paddleocr import PaddleOCR
except Exception as e:
    PaddleOCR = None
    _paddleocr_import_error = e


class OcrCompletionRequest(BaseModel):
    model: str = Field(...)
    image_file: str = Field(..., description="图片路径或 base64 字符串")
    file_type: Literal["path", "base64"] = Field(
        default="path",
        description="image_file 的类型: 'path' 或 'base64'",
    )


class OcrCompletionResponse(BaseModel):
    id: str = Field(...)
    created: int = Field(...)
    model: str = Field(...)
    response: list = Field(...)


@dataclass
class DeviceInfo:
    requested_device: str
    selected_device: str
    os_name: str
    cuda_compiled: bool
    cuda_available: bool
    cuda_device_count: int
    mps_detected: bool
    fallback_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "selected_device": self.selected_device,
            "os": self.os_name,
            "cuda_compiled": self.cuda_compiled,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "mps_detected": self.mps_detected,
            "fallback_reason": self.fallback_reason,
        }


class RuntimeStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0
        self.queue_full_rejections = 0
        self.queue_timeout_rejections = 0
        self.total_success_latency_ms = 0.0
        self.last_error_ts: Optional[int] = None

    def on_request_start(self) -> None:
        with self._lock:
            self.total_requests += 1

    def on_request_success(self, latency_ms: float) -> None:
        with self._lock:
            self.success_requests += 1
            self.total_success_latency_ms += max(0.0, latency_ms)

    def on_request_failure(self) -> None:
        with self._lock:
            self.failed_requests += 1

    def on_queue_full_rejection(self) -> None:
        with self._lock:
            self.queue_full_rejections += 1

    def on_queue_timeout_rejection(self) -> None:
        with self._lock:
            self.queue_timeout_rejections += 1

    def on_error_timestamp(self) -> None:
        with self._lock:
            self.last_error_ts = int(time.time())

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            avg_latency_ms = (
                self.total_success_latency_ms / self.success_requests
                if self.success_requests > 0
                else 0.0
            )
            return {
                "total_requests": self.total_requests,
                "success_requests": self.success_requests,
                "failed_requests": self.failed_requests,
                "queue_full_rejections": self.queue_full_rejections,
                "queue_timeout_rejections": self.queue_timeout_rejections,
                "avg_latency_ms": round(avg_latency_ms, 2),
                "last_error_ts": self.last_error_ts,
            }


class InferenceConcurrencyLimiter:
    def __init__(self, max_concurrent_ocr: int, timeout_seconds: float, max_waiting_requests: int) -> None:
        if max_concurrent_ocr <= 0:
            raise ValueError("max_concurrent_ocr must be >= 1")
        if timeout_seconds <= 0:
            raise ValueError("concurrency_timeout must be > 0")
        if max_waiting_requests <= 0:
            raise ValueError("max_waiting_requests must be >= 1")

        self.max_concurrent_ocr = max_concurrent_ocr
        self.timeout_seconds = timeout_seconds
        self.max_waiting_requests = max_waiting_requests

        self._semaphore = threading.BoundedSemaphore(max_concurrent_ocr)
        self._lock = threading.Lock()
        self._active_requests = 0
        self._waiting_requests = 0

    @staticmethod
    def _raise_queue_full_429(max_waiting_requests: int) -> None:
        raise HTTPException(
            status_code=429,
            detail=(
                "reason=queue_full; "
                f"Model is busy, waiting queue is full. max_waiting_requests={max_waiting_requests}"
            ),
            headers={"Retry-After": "1"},
        )

    def _raise_timeout_429(self) -> None:
        retry_after = max(1, int(math.ceil(self.timeout_seconds)))
        raise HTTPException(
            status_code=429,
            detail=(
                "reason=queue_timeout; "
                "Model is busy, request queue timeout reached. "
                f"max_concurrent_ocr={self.max_concurrent_ocr}"
            ),
            headers={"Retry-After": str(retry_after)},
        )

    def acquire(self, request_tag: Optional[str] = None) -> None:
        tag = request_tag or "-"
        with self._lock:
            if self._waiting_requests >= self.max_waiting_requests:
                waiting_snapshot = self._waiting_requests
                active_snapshot = self._active_requests
                print(
                    f"[queue-reject] tag={tag} active={active_snapshot} "
                    f"waiting={waiting_snapshot} limit={self.max_concurrent_ocr} "
                    f"max_waiting_requests={self.max_waiting_requests}"
                )
                self._raise_queue_full_429(self.max_waiting_requests)

            self._waiting_requests += 1
            waiting_snapshot = self._waiting_requests
            active_snapshot = self._active_requests

        print(
            f"[queue-enter] tag={tag} active={active_snapshot} "
            f"waiting={waiting_snapshot} limit={self.max_concurrent_ocr}"
        )

        t0 = time.time()
        acquired = self._semaphore.acquire(timeout=self.timeout_seconds)
        wait_ms = int((time.time() - t0) * 1000)

        with self._lock:
            self._waiting_requests = max(0, self._waiting_requests - 1)
            if acquired:
                self._active_requests += 1
            waiting_snapshot = self._waiting_requests
            active_snapshot = self._active_requests

        if not acquired:
            print(
                f"[queue-timeout] tag={tag} wait_ms={wait_ms} active={active_snapshot} "
                f"waiting={waiting_snapshot} limit={self.max_concurrent_ocr}"
            )
            self._raise_timeout_429()

        print(
            f"[slot-acquired] tag={tag} wait_ms={wait_ms} active={active_snapshot} "
            f"waiting={waiting_snapshot} limit={self.max_concurrent_ocr}"
        )

    def release(self, request_tag: Optional[str] = None) -> None:
        tag = request_tag or "-"
        self._semaphore.release()
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            active_snapshot = self._active_requests
            waiting_snapshot = self._waiting_requests
        print(
            f"[slot-released] tag={tag} active={active_snapshot} "
            f"waiting={waiting_snapshot} limit={self.max_concurrent_ocr}"
        )

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "active_requests": self._active_requests,
                "waiting_requests": self._waiting_requests,
            }

    @contextmanager
    def hold(self, request_tag: Optional[str] = None) -> Iterator[None]:
        self.acquire(request_tag=request_tag)
        try:
            yield
        finally:
            self.release(request_tag=request_tag)


class OcrEnginePool:
    def __init__(self, pool_size: int, engine_loader: Any) -> None:
        if pool_size <= 0:
            raise ValueError("pool_size must be >= 1")
        self.pool_size = pool_size
        self._queue: Queue = Queue(maxsize=pool_size)
        for idx in range(pool_size):
            self._queue.put(engine_loader(idx))

    @contextmanager
    def hold_engine(self) -> Iterator[Any]:
        try:
            engine = self._queue.get_nowait()
        except Empty as e:
            raise RuntimeError(
                "OCR engine pool is exhausted unexpectedly. "
                "Please check limiter and pool size configuration."
            ) from e
        try:
            yield engine
        finally:
            self._queue.put(engine)


def ensure_runtime_dependencies() -> None:
    missing = []
    if paddle is None:
        missing.append(("paddlepaddle", _paddle_import_error))
    if PaddleOCR is None:
        missing.append(("paddleocr", _paddleocr_import_error))
    if cv2 is None:
        missing.append(("opencv-python", _cv2_import_error))

    if not missing:
        return

    detail_lines = []
    for pkg_name, err in missing:
        reason = str(err) if err is not None else "unknown import error"
        detail_lines.append(f"- {pkg_name}: {reason}")
    detail = "\n".join(detail_lines)
    raise RuntimeError(
        "Missing required runtime dependencies for PaddleOCR service.\n"
        f"{detail}\n"
        "Please run: pip install -r requirements.txt"
    )


def _detect_mps_available() -> bool:
    if torch is None:
        return False
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    try:
        return bool(mps_backend.is_built() and mps_backend.is_available())
    except Exception:
        return False


def _probe_cuda_with_paddle() -> tuple[bool, bool, int]:
    if paddle is None:
        return False, False, 0
    try:
        cuda_compiled = bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        return False, False, 0
    if not cuda_compiled:
        return False, False, 0

    cuda_device_count = 0
    try:
        cuda_device_count = int(paddle.device.cuda.device_count())
    except Exception:
        cuda_device_count = 0
    return True, cuda_device_count > 0, cuda_device_count


def _normalize_preferred_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized in {"auto", "cpu", "gpu", "gpu:0"}:
        return "gpu:0" if normalized == "gpu" else normalized
    if normalized.startswith("gpu:"):
        _, _, index_str = normalized.partition(":")
        if index_str.isdigit() and int(index_str) >= 0:
            return normalized
    raise argparse.ArgumentTypeError("device must be one of: auto, cpu, gpu:0 (or gpu:N)")


def detect_runtime_device(preferred_device: str) -> DeviceInfo:
    requested_device = _normalize_preferred_device(preferred_device)
    os_name = platform.system().lower()
    cuda_compiled, cuda_available, cuda_device_count = _probe_cuda_with_paddle()
    mps_detected = _detect_mps_available() if os_name == "darwin" else False

    fallback_reason: Optional[str] = None

    if requested_device == "auto":
        if cuda_available:
            selected_device = "gpu:0"
        else:
            selected_device = "cpu"
            if cuda_compiled and cuda_device_count == 0:
                fallback_reason = "CUDA is compiled, but no available CUDA devices were detected."
            elif not cuda_compiled:
                fallback_reason = "CUDA runtime is not available in current Paddle build."
    elif requested_device == "cpu":
        selected_device = "cpu"
    else:
        selected_device = requested_device
        if not cuda_available:
            selected_device = "cpu"
            fallback_reason = (
                f"Requested '{requested_device}', but CUDA is unavailable. "
                "Fallback to CPU."
            )
        else:
            _, _, index_str = requested_device.partition(":")
            gpu_index = int(index_str) if index_str else 0
            if gpu_index >= cuda_device_count:
                selected_device = "cpu"
                fallback_reason = (
                    f"Requested GPU index {gpu_index}, but only "
                    f"{cuda_device_count} CUDA devices are available. Fallback to CPU."
                )

    if os_name == "darwin" and selected_device == "cpu":
        mac_cpu_only_note = (
            "PaddleOCR on macOS currently runs on CPU only. "
            "MPS may be detected by PyTorch, but Paddle backend does not use MPS here."
        )
        if fallback_reason:
            fallback_reason = f"{fallback_reason} {mac_cpu_only_note}"
        else:
            fallback_reason = mac_cpu_only_note

    return DeviceInfo(
        requested_device=requested_device,
        selected_device=selected_device,
        os_name=os_name,
        cuda_compiled=cuda_compiled,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        mps_detected=mps_detected,
        fallback_reason=fallback_reason,
    )


def parse_auto_or_positive_int(value: Any, arg_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{arg_name} must be >= 1")
        return value

    normalized = str(value).strip().lower()
    if normalized == "auto":
        return None
    try:
        parsed = int(normalized)
    except Exception as e:
        raise ValueError(f"{arg_name} must be 'auto' or positive int") from e
    if parsed <= 0:
        raise ValueError(f"{arg_name} must be >= 1")
    return parsed


class DeployPaddleOcr:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        health_api_key: str = "",
        device: str = "auto",
        max_concurrent_ocr: Optional[int] = None,
        concurrency_timeout: float = 15.0,
        max_waiting_requests: Optional[int] = None,
        reload_fail_threshold: int = 3,
        reload_cooldown_sec: int = 60,
        reload_every: int = 0,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.health_api_key = health_api_key.strip()
        self.device_info = detect_runtime_device(device)
        self.started_at = time.time()

        if reload_every > 0:
            print(
                "[deprecated] --reload_every is deprecated and ignored by default. "
                "Current policy uses failure-triggered reload only."
            )

        self.max_concurrent_ocr = max_concurrent_ocr or self._get_default_pool_size(
            self.device_info.selected_device
        )
        self.max_waiting_requests = max_waiting_requests or self.max_concurrent_ocr * 8
        self.concurrency_timeout = concurrency_timeout

        self.reload_fail_threshold = reload_fail_threshold
        self.reload_cooldown_sec = reload_cooldown_sec

        self.stats = RuntimeStats()
        self.concurrency_limiter = InferenceConcurrencyLimiter(
            max_concurrent_ocr=self.max_concurrent_ocr,
            timeout_seconds=self.concurrency_timeout,
            max_waiting_requests=self.max_waiting_requests,
        )

        self._pool_lock = threading.Lock()
        self._failure_lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self._consecutive_failures = 0
        self._last_error: Optional[str] = None
        self._last_error_ts: Optional[int] = None
        self._last_reload_ts = 0.0
        self._reload_count = 0

        self._engine_pool = self._build_engine_pool()
        print(
            "[startup] "
            f"os={self.device_info.os_name}, "
            f"selected_device={self.device_info.selected_device}, "
            f"cuda_available={self.device_info.cuda_available}, "
            f"mps_detected={self.device_info.mps_detected}, "
            f"pool_size={self.max_concurrent_ocr}, "
            f"max_waiting_requests={self.max_waiting_requests}, "
            f"concurrency_timeout={self.concurrency_timeout}s, "
            f"fallback_reason={self.device_info.fallback_reason}, "
            f"health_auth={'enabled' if self.health_api_key else 'disabled'}"
        )

    @staticmethod
    def _truncate_message(message: str, max_len: int = 256) -> str:
        safe = (message or "").strip().replace("\n", " ")
        if len(safe) <= max_len:
            return safe
        return safe[: max_len - 3] + "..."

    @staticmethod
    def _decode_base64_to_ndarray(b64_str: str) -> Any:
        if cv2 is None:
            raise RuntimeError("opencv-python is unavailable.")

        if b64_str.startswith("data:"):
            _, b64_str = b64_str.split(",", 1)

        try:
            img_bytes = base64.b64decode(b64_str)
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}")

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode base64 image to ndarray")
        return img

    @staticmethod
    def _get_default_pool_size(selected_device: str) -> int:
        if selected_device.startswith("gpu"):
            return 2
        cpu_count = os.cpu_count() or 1
        return min(4, max(1, cpu_count // 2))

    def _load_model(self, instance_idx: int) -> Any:
        if PaddleOCR is None:
            raise RuntimeError("paddleocr is not installed. Please install dependencies first.")

        print(
            f"[model-load] instance={instance_idx + 1}/{self.max_concurrent_ocr} "
            f"device={self.device_info.selected_device}"
        )
        try:
            ocr = PaddleOCR(device=self.device_info.selected_device)
        except TypeError:
            use_gpu = self.device_info.selected_device.startswith("gpu")
            ocr = PaddleOCR(use_gpu=use_gpu)
        print(f"[model-ready] instance={instance_idx + 1}")
        return ocr

    def _build_engine_pool(self) -> OcrEnginePool:
        print(
            f"[pool-build] start loading {self.max_concurrent_ocr} model instance(s) "
            f"on {self.device_info.selected_device}"
        )
        pool = OcrEnginePool(pool_size=self.max_concurrent_ocr, engine_loader=self._load_model)
        print("[pool-build] done")
        return pool

    def _extract_texts(self, result: Any) -> list[str]:
        if not result:
            return []

        first = result[0]
        if isinstance(first, dict):
            texts = first.get("rec_texts", [])
            if isinstance(texts, list):
                return [str(x) for x in texts]
            return []

        parsed: list[str] = []
        if isinstance(first, (list, tuple)):
            for item in first:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[1], (list, tuple))
                    and len(item[1]) >= 1
                ):
                    parsed.append(str(item[1][0]))
        return parsed

    def _pool_snapshot(self) -> OcrEnginePool:
        with self._pool_lock:
            return self._engine_pool

    def _record_inference_success(self) -> None:
        with self._failure_lock:
            self._consecutive_failures = 0

    def _record_inference_failure(self, err: Exception) -> None:
        err_msg = self._truncate_message(str(err))
        with self._failure_lock:
            self._consecutive_failures += 1
            self._last_error = err_msg
            self._last_error_ts = int(time.time())
            failure_count = self._consecutive_failures

        self.stats.on_error_timestamp()
        print(f"[ocr-failure] consecutive_failures={failure_count}, error={err_msg}")
        self._maybe_reload_on_failure(failure_count=failure_count, reason=err_msg)

    def _maybe_reload_on_failure(self, failure_count: int, reason: str) -> None:
        if self.reload_fail_threshold <= 0:
            return
        if failure_count < self.reload_fail_threshold:
            return

        now = time.time()
        with self._reload_lock:
            with self._failure_lock:
                latest_failure_count = self._consecutive_failures
            if latest_failure_count < self.reload_fail_threshold:
                return

            if self._last_reload_ts > 0 and (now - self._last_reload_ts) < self.reload_cooldown_sec:
                return

            print(
                f"[reload-start] reason={reason}, "
                f"consecutive_failures={latest_failure_count}, "
                f"cooldown={self.reload_cooldown_sec}s"
            )
            try:
                new_pool = self._build_engine_pool()
            except Exception as e:
                print(f"[reload-failed] {e}")
                return

            with self._pool_lock:
                old_pool = self._engine_pool
                self._engine_pool = new_pool

            del old_pool
            gc.collect()
            with self._failure_lock:
                self._consecutive_failures = 0
            self._last_reload_ts = time.time()
            self._reload_count += 1
            print(f"[reload-done] reload_count={self._reload_count}")

    @staticmethod
    def _parse_429_reason(exc: HTTPException) -> Optional[str]:
        detail = str(exc.detail or "")
        if "reason=queue_full" in detail:
            return "queue_full"
        if "reason=queue_timeout" in detail:
            return "queue_timeout"
        return None

    def _run_ocr(self, image_file: str, file_type: str, request_tag: str) -> list[str]:
        if file_type == "path":
            input_data = image_file
        elif file_type == "base64":
            input_data = self._decode_base64_to_ndarray(image_file)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")

        with self.concurrency_limiter.hold(request_tag=request_tag):
            pool = self._pool_snapshot()
            with pool.hold_engine() as ocr_engine:
                result = ocr_engine.predict(input_data)
        return self._extract_texts(result)

    def ocr_completion(self, req: OcrCompletionRequest) -> OcrCompletionResponse:
        if req.model and req.model != self.model_name:
            raise ValueError(
                f"Unsupported model '{req.model}'. This engine is '{self.model_name}'."
            )

        request_tag = uuid.uuid4().hex[:8]
        created = int(time.time())
        started_at = time.time()
        self.stats.on_request_start()

        try:
            ocr_results = self._run_ocr(req.image_file, req.file_type, request_tag=request_tag)
        except HTTPException as e:
            self.stats.on_request_failure()
            if e.status_code == 429:
                reason = self._parse_429_reason(e)
                if reason == "queue_full":
                    self.stats.on_queue_full_rejection()
                elif reason == "queue_timeout":
                    self.stats.on_queue_timeout_rejection()
            elif e.status_code >= 500:
                self._record_inference_failure(e)
            raise
        except ValueError:
            self.stats.on_request_failure()
            raise
        except Exception as e:
            self.stats.on_request_failure()
            self._record_inference_failure(e)
            raise

        latency_ms = (time.time() - started_at) * 1000.0
        self.stats.on_request_success(latency_ms)
        self._record_inference_success()

        return OcrCompletionResponse(
            id="ocrcmpl-" + uuid.uuid4().hex,
            created=created,
            model=self.model_name,
            response=ocr_results,
        )

    def healthz(self) -> Dict[str, Any]:
        with self._failure_lock:
            consecutive_failures = self._consecutive_failures
            last_error = self._last_error
            last_error_ts = self._last_error_ts
        with self._pool_lock:
            engine_ready = self._engine_pool is not None

        ready_threshold = self.reload_fail_threshold * 2
        ready = engine_ready and consecutive_failures < ready_threshold
        degraded = consecutive_failures >= self.reload_fail_threshold
        last_reload_ts = int(self._last_reload_ts) if self._last_reload_ts > 0 else None

        return {
            "ok": True,
            "ready": ready,
            "degraded": degraded,
            "last_error": last_error,
            "last_error_ts": last_error_ts,
            "model_name": self.model_name,
            "selected_device": self.device_info.selected_device,
            "device_probe": self.device_info.to_dict(),
            "pool_size": self.max_concurrent_ocr,
            "uptime_sec": int(time.time() - self.started_at),
            "consecutive_failures": consecutive_failures,
            "last_reload_ts": last_reload_ts,
        }

    def metrics_lite(self) -> Dict[str, Any]:
        stats = self.stats.snapshot()
        queue_snapshot = self.concurrency_limiter.snapshot()
        return {
            "total_requests": stats["total_requests"],
            "success_requests": stats["success_requests"],
            "failed_requests": stats["failed_requests"],
            "active_requests": queue_snapshot["active_requests"],
            "waiting_requests": queue_snapshot["waiting_requests"],
            "queue_full_rejections": stats["queue_full_rejections"],
            "queue_timeout_rejections": stats["queue_timeout_rejections"],
            "reload_count": self._reload_count,
            "avg_latency_ms": stats["avg_latency_ms"],
            "last_error_ts": stats["last_error_ts"],
        }


app = FastAPI(title="PaddleOCR API")
paddle_ocr_engine: Optional[DeployPaddleOcr] = None


def get_engine_or_503() -> DeployPaddleOcr:
    if paddle_ocr_engine is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return paddle_ocr_engine


def _require_health_auth(engine: DeployPaddleOcr, authorization: Optional[str]) -> None:
    if not engine.health_api_key:
        return
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    api_key = authorization.removeprefix("Bearer ").strip()
    if api_key != engine.health_api_key:
        raise HTTPException(status_code=401, detail="Invalid health API key")


@app.get("/healthz")
def healthz(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    engine = get_engine_or_503()
    _require_health_auth(engine, authorization)
    return engine.healthz()


@app.get("/metrics-lite")
def metrics_lite(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    engine = get_engine_or_503()
    _require_health_auth(engine, authorization)
    return engine.metrics_lite()


@app.post("/v1/ocr")
def create_ocr_completion(
    req: OcrCompletionRequest,
    authorization: Optional[str] = Header(default=None),
) -> OcrCompletionResponse:
    engine = get_engine_or_503()

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    api_key = authorization.removeprefix("Bearer ").strip()
    if api_key != engine.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        return engine.ocr_completion(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def normalize_startup_args(args: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(args)
    normalized["device"] = _normalize_preferred_device(normalized["device"])
    normalized["max_concurrent_ocr"] = parse_auto_or_positive_int(
        normalized["max_concurrent_ocr"], "--max_concurrent_ocr"
    )
    normalized["max_waiting_requests"] = parse_auto_or_positive_int(
        normalized["max_waiting_requests"], "--max_waiting_requests"
    )
    normalized["host"] = str(normalized["host"]).strip()
    if not normalized["host"]:
        raise ValueError("--host must not be empty")

    normalized["health_api_key"] = str(normalized.get("health_api_key", "")).strip()

    if normalized["concurrency_timeout"] <= 0:
        raise ValueError("--concurrency_timeout must be > 0")
    if normalized["reload_fail_threshold"] <= 0:
        raise ValueError("--reload_fail_threshold must be >= 1")
    if normalized["reload_cooldown_sec"] < 0:
        raise ValueError("--reload_cooldown_sec must be >= 0")
    if normalized["reload_every"] < 0:
        raise ValueError("--reload_every must be >= 0")
    return normalized


def main(args: Dict[str, Any]) -> None:
    global paddle_ocr_engine
    ensure_runtime_dependencies()
    args = normalize_startup_args(args)

    paddle_ocr_engine = DeployPaddleOcr(
        model_name=args["model_name"],
        api_key=args["api_key"],
        health_api_key=args["health_api_key"],
        device=args["device"],
        max_concurrent_ocr=args["max_concurrent_ocr"],
        concurrency_timeout=args["concurrency_timeout"],
        max_waiting_requests=args["max_waiting_requests"],
        reload_fail_threshold=args["reload_fail_threshold"],
        reload_cooldown_sec=args["reload_cooldown_sec"],
        reload_every=args["reload_every"],
    )

    uvicorn.run(app, host=args["host"], port=args["port"], workers=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run PaddleOCR deployment")
    parser.add_argument("--model_name", default="PaddleOCR", type=str)
    parser.add_argument("--api_key", default="sk-1234", type=str)
    parser.add_argument("--health_api_key", default="", type=str)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=1234, type=int)
    parser.add_argument(
        "--max_concurrent_ocr",
        default="1",
        type=str,
        help="Maximum concurrent OCR jobs. Use 'auto' for device-based defaults.",
    )
    parser.add_argument(
        "--concurrency_timeout",
        default=120.0,
        type=float,
        help="How long a request waits for an OCR slot (seconds).",
    )
    parser.add_argument(
        "--max_waiting_requests",
        default="auto",
        type=str,
        help="Maximum waiting requests. Use 'auto' for max_concurrent_ocr * 8.",
    )
    parser.add_argument(
        "--reload_fail_threshold",
        default=3,
        type=int,
        help="Consecutive failure threshold for triggering model pool reload.",
    )
    parser.add_argument(
        "--reload_cooldown_sec",
        default=60,
        type=int,
        help="Cooldown between two failure-triggered reloads.",
    )
    parser.add_argument(
        "--reload_every",
        default=0,
        type=int,
        help="Deprecated. Kept for compatibility and ignored by default.",
    )
    main(vars(parser.parse_args()))
