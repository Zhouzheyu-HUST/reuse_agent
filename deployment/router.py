# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.


from __future__ import annotations

__author__ = "Zhipeng Hou"

import argparse
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse


# -----------------------------
# Config
# -----------------------------
CONFIG_PATH = os.environ.get("ROUTER_CONFIG", "./router_services.json")
ADMIN_TOKEN = os.environ.get("ROUTER_ADMIN_TOKEN", "sk-1234")
ADMIN_PREFIX = "/_admin"

DEFAULT_TIMEOUT = float(os.environ.get("ROUTER_TIMEOUT", "300"))  # seconds
MAX_CONNECTIONS = int(os.environ.get("ROUTER_MAX_CONNECTIONS", "200"))

# WebSocket upstream max message size (None = unlimited). You can set e.g. 16MB.
WS_MAX_SIZE = os.environ.get("ROUTER_WS_MAX_SIZE")
WS_MAX_SIZE = None if WS_MAX_SIZE in (None, "", "none", "None") else int(WS_MAX_SIZE)


# -----------------------------
# Data Model
# -----------------------------
@dataclass
class Service:
    name: str
    prefix: str              # e.g. "/ocr" (must start with "/")
    target: str              # e.g. "http://127.0.0.1:9000" (for ws it will be converted)
    strip_prefix: bool = True
    enabled: bool = True
    created_at: int = 0
    updated_at: int = 0

    def normalize(self) -> "Service":
        if not self.prefix.startswith("/"):
            self.prefix = "/" + self.prefix
        if self.prefix != "/" and self.prefix.endswith("/"):
            self.prefix = self.prefix[:-1]
        self.target = self.target.rstrip("/")
        now = int(time.time())
        if self.created_at == 0:
            self.created_at = now
        self.updated_at = now
        return self


class Registry:
    def __init__(self, path: str):
        self.path = path
        self._lock = asyncio.Lock()
        self.services: Dict[str, Service] = {}

    async def load(self) -> None:
        async with self._lock:
            if not os.path.exists(self.path):
                self.services = {}
                return
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            services = {}
            for item in raw.get("services", []):
                svc = Service(**item).normalize()
                services[svc.name] = svc
            self.services = services

    async def save(self) -> None:
        async with self._lock:
            data = {"services": [asdict(s) for s in self.services.values()]}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)

    async def list(self) -> List[Service]:
        async with self._lock:
            return sorted(self.services.values(), key=lambda s: s.name)

    async def upsert(self, svc: Service) -> Service:
        svc.normalize()
        async with self._lock:
            if svc.name in self.services:
                svc.created_at = self.services[svc.name].created_at
            self.services[svc.name] = svc
        await self.save()
        return svc

    async def delete(self, name: str) -> None:
        async with self._lock:
            if name not in self.services:
                raise KeyError(name)
            del self.services[name]
        await self.save()

    async def match(self, path: str) -> Optional[Tuple[Service, str]]:
        """
        Longest-prefix match.
        Returns (service, remainder_path_with_leading_slash)
        """
        async with self._lock:
            candidates = [
                s for s in self.services.values()
                if s.enabled and (path == s.prefix or path.startswith(s.prefix + "/"))
            ]
            if not candidates:
                return None
            svc = max(candidates, key=lambda s: len(s.prefix))
            if svc.strip_prefix:
                remainder = path[len(svc.prefix):]
                if remainder == "":
                    remainder = "/"
            else:
                remainder = path
            return svc, remainder


# -----------------------------
# Helpers
# -----------------------------
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _require_admin(x_admin_token: str) -> None:
    if not ADMIN_TOKEN:
        # allow dev mode; production strongly建议设置
        return
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid admin token")


def _filter_request_headers(req_headers: httpx.Headers) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in req_headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS:
            continue
        if lk in ("host", "content-length"):
            continue
        out[k] = v
    return out


def _filter_response_headers(resp_headers: httpx.Headers) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in resp_headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS:
            continue
        if lk == "content-length":
            continue
        out[k] = v
    return out


def _to_ws_url(http_url: str) -> str:
    # http:// -> ws:// ; https:// -> wss://
    if http_url.startswith("https://"):
        return "wss://" + http_url[len("https://"):]
    if http_url.startswith("http://"):
        return "ws://" + http_url[len("http://"):]
    # allow already ws/wss
    return http_url


# -----------------------------
# Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = Registry(CONFIG_PATH)
    await registry.load()

    limits = httpx.Limits(
        max_connections=MAX_CONNECTIONS,
        max_keepalive_connections=MAX_CONNECTIONS,
    )
    client = httpx.AsyncClient(
        limits=limits,
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
        follow_redirects=False,
    )

    app.state.registry = registry
    app.state.client = client

    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(
    title="Enhanced Single-Port Multi-Service Router",
    lifespan=lifespan,
    docs_url=f"{ADMIN_PREFIX}/docs",
    redoc_url=None,
)


# -----------------------------
# Basic
# -----------------------------
@app.get("/health")
async def health():
    return {"ok": True, "ts": int(time.time()), "config_path": CONFIG_PATH}


# -----------------------------
# Admin APIs (hot effective)
# -----------------------------
@app.get(f"{ADMIN_PREFIX}/services")
async def list_services(x_admin_token: str = Header(default="")):
    _require_admin(x_admin_token)
    registry: Registry = app.state.registry
    svcs = await registry.list()
    return {"services": [asdict(s) for s in svcs], "config_path": CONFIG_PATH}


@app.post(f"{ADMIN_PREFIX}/services")
async def register_service(payload: dict, x_admin_token: str = Header(default="")):
    """
    payload example:
    {
      "name": "ocr",
      "prefix": "/ocr",
      "target": "http://127.0.0.1:9000",
      "strip_prefix": true,
      "enabled": true
    }
    """
    _require_admin(x_admin_token)
    registry: Registry = app.state.registry

    try:
        svc = Service(
            name=payload["name"],
            prefix=payload["prefix"],
            target=payload["target"],
            strip_prefix=bool(payload.get("strip_prefix", True)),
            enabled=bool(payload.get("enabled", True)),
            created_at=int(payload.get("created_at", 0)),
            updated_at=0,
        ).normalize()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    svc = await registry.upsert(svc)
    return {"ok": True, "service": asdict(svc)}


@app.delete(f"{ADMIN_PREFIX}/services/{{name}}")
async def delete_service(name: str, x_admin_token: str = Header(default="")):
    _require_admin(x_admin_token)
    registry: Registry = app.state.registry
    try:
        await registry.delete(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Service not found: {name}")
    return {"ok": True, "deleted": name}


# -----------------------------
# WebSocket proxy
# -----------------------------
@app.websocket("/{full_path:path}")
async def websocket_proxy(websocket: WebSocket, full_path: str):
    path = "/" + full_path

    # admin path should not be proxied
    if path.startswith(ADMIN_PREFIX + "/") or path == ADMIN_PREFIX:
        await websocket.close(code=1008)  # Policy Violation
        return

    registry: Registry = app.state.registry
    matched = await registry.match(path)
    if not matched:
        await websocket.accept()
        await websocket.send_json({"error": "No route matched", "path": path})
        await websocket.close(code=1000)
        return

    svc, remainder = matched
    upstream_ws_url = _to_ws_url(svc.target) + remainder
    if websocket.url.query:
        upstream_ws_url = upstream_ws_url + "?" + websocket.url.query

    # Accept client websocket first
    await websocket.accept()

    # Build headers to upstream (optional: forward some headers)
    # NOTE: websockets expects a list of (k, v)
    upstream_headers = []
    for k, v in websocket.headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS or lk == "host":
            continue
        # You may want to forward auth/cookies; keep as-is.
        upstream_headers.append((k, v))

    try:
        async with websockets.connect(
            upstream_ws_url,
            extra_headers=upstream_headers,
            max_size=WS_MAX_SIZE,
        ) as upstream:
            async def client_to_upstream():
                try:
                    while True:
                        msg = await websocket.receive()
                        if "text" in msg and msg["text"] is not None:
                            await upstream.send(msg["text"])
                        elif "bytes" in msg and msg["bytes"] is not None:
                            await upstream.send(msg["bytes"])
                        elif msg.get("type") == "websocket.disconnect":
                            break
                except WebSocketDisconnect:
                    pass
                except Exception:
                    # if upstream already closed, ignore
                    pass
                finally:
                    try:
                        await upstream.close()
                    except Exception:
                        pass

            async def upstream_to_client():
                try:
                    while True:
                        data = await upstream.recv()
                        if isinstance(data, (bytes, bytearray)):
                            await websocket.send_bytes(bytes(data))
                        else:
                            await websocket.send_text(str(data))
                except Exception:
                    pass
                finally:
                    try:
                        await websocket.close(code=1000)
                    except Exception:
                        pass

            await asyncio.gather(client_to_upstream(), upstream_to_client())

    except Exception as e:
        # Tell client error then close
        try:
            await websocket.send_json(
                {"error": "Upstream websocket connect failed", "service": asdict(svc), "detail": str(e)}
            )
        except Exception:
            pass
        try:
            await websocket.close(code=1011)  # Internal Error
        except Exception:
            pass


# -----------------------------
# HTTP proxy (streaming)
# -----------------------------
@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def http_proxy(full_path: str, request: Request):
    path = "/" + full_path

    if path.startswith(ADMIN_PREFIX + "/") or path == ADMIN_PREFIX:
        return JSONResponse({"error": "Not Found"}, status_code=404)

    registry: Registry = app.state.registry
    client: httpx.AsyncClient = app.state.client

    matched = await registry.match(path)
    if not matched:
        return JSONResponse(
            {"error": "No route matched", "path": path, "hint": "Register via /_admin/services"},
            status_code=404,
        )

    svc, remainder = matched
    target_url = f"{svc.target}{remainder}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    headers = _filter_request_headers(request.headers)

    # Stream request body to upstream (supports big uploads)
    async def iter_request_body():
        async for chunk in request.stream():
            yield chunk

    try:
        upstream = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=iter_request_body(),  # streaming upload
        )
    except httpx.ConnectError as e:
        return JSONResponse(
            {"error": "Upstream connect error", "service": asdict(svc), "detail": str(e)},
            status_code=502,
        )
    except httpx.ReadTimeout as e:
        return JSONResponse(
            {"error": "Upstream timeout", "service": asdict(svc), "detail": str(e)},
            status_code=504,
        )
    except Exception as e:
        return JSONResponse(
            {"error": "Upstream request failed", "service": asdict(svc), "detail": str(e)},
            status_code=502,
        )

    resp_headers = _filter_response_headers(upstream.headers)

    # Stream response body to client (supports big downloads / SSE)
    async def iter_response_body():
        async for chunk in upstream.aiter_bytes():
            yield chunk

    return StreamingResponse(
        iter_response_body(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


def main(args: dict) -> None:
    uvicorn.run(app, host="127.0.0.1", port=args["port"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run router service")
    parser.add_argument(
        "--port",
        default=6008,
        type=int
    )
    args = vars(parser.parse_args())
    main(args)
