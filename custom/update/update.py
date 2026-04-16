#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update.py

【功能】
把一条 longchain.json（messages 格式）导入到库目录 ROOT 中，并更新：
- ROOT/actions/：按“同一操作”聚合 action，并维护 variants（不同位置/不同输入/不同滑动形态）
- ROOT/tasks/：每个任务只记录 step 的 {thought + action_id}

【 PROJECT 结构（示例）】
PROJECT/
  new/
    screen/1.jpeg
    ui/1.json
  hittest.py
  longchain.json
  update.py
  淘宝/                 # 某个应用库（脚本 --root 指向它）
    actions/
    tasks/

【Qwen3-VL-plus 多图调用】
使用 DashScope compatible-mode + OpenAI SDK 方式。
你只需要填 QWEN_API_KEY。
"""

from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI  # pip install openai
import requests
import io
import time
from PIL import Image
# ✅ 你实现的命中检测函数（同目录 hittest.py）
from custom.update.hittest import same_ui_hit
from custom.update.getpackage import AppResolver
from custom.validity_check import validity_check_for_ui_match
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================================================
# 0) 你只需要改这里：填 API KEY
# =========================================================
def load_api_config(filename="api_settings.json"):
    """
    从当前脚本所在目录向上递归查找 custom/api_settings.json
    """
    # 获取当前文件所在的绝对目录
    current_path = Path(__file__).resolve().parent
    
    # 向上遍历所有父级目录 (包含当前目录)
    for parent in [current_path] + list(current_path.parents):
        # 检查是否存在 configs 文件夹下的配置文件
        config_file = parent / "custom" / filename
        if config_file.exists():
            try:
                # print(f"[Info] Loaded config from: {config_file}")
                return json.loads(config_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error loading config: {e}")
                print(f"未成功找到配置文件 {filename}，请确保其存在于 'configs' 文件夹中。")
                return {}
    
    print(f"Warning: {filename} not found in 'configs' folder of any parent directory.")
    return {}

_config = load_api_config()
API_URL = _config.get("llm_endpoints")
API_KEY = _config.get("llm_api_key")
MODEL_NAME = _config.get("llm_model")
LLM_CHECK_WAY = _config.get("llm_check_way")

'''
#这里还有可以手动设置的部分，以防出现问题无法载入。
API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "sk-1234"
MODEL_NAME = "Qwen3-VL-8B-Instruct"
'''


class LlmWrapper:
    RETRY_WAITING_SECONDS = 2
    MAX_RETRY = 3

    def __init__(self, api_key: str, model_name: str, api_url: str, check_way: str = "openai"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.check_way = check_way

    def predict(self, messages: list[dict], temperature: float = 0.01) -> str | None:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 512,
            "stream": False
        }
        if self.check_way == "openai":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        elif self.check_way == "csb":
            headers = {
                "Content-Type": "application/json",
                "csb-token": self.api_key
            }

        counter = self.MAX_RETRY
        wait_seconds = self.RETRY_WAITING_SECONDS

        while counter > 0:
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                if response.ok:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"]
                print(f"Error: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Exception: {e}")
            
            time.sleep(wait_seconds)
            wait_seconds *= 2
            counter -= 1
        return None

# 初始化全局 Wrapper
llm_wrapper = LlmWrapper(API_KEY, MODEL_NAME, API_URL, LLM_CHECK_WAY)


# =========================================================
# 1) 通用工具函数
# =========================================================

NUM_RE = re.compile(r"^\d+$")
WS_RE = re.compile(r"\s+")
QUESTION_RE = re.compile(r"<Question>(.*?)</Question>", re.DOTALL)


def now_iso_utc() -> str:
    """返回当前 UTC 时间戳（ISO8601）。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(p: Path) -> None:
    """确保目录存在。"""
    p.mkdir(parents=True, exist_ok=True)


def read_json(p: Path) -> Any:
    """读取 JSON 文件。"""
    return json.loads(p.read_text(encoding="utf-8"))


def atomic_write_json(p: Path, obj: Any) -> None:
    """原子写 JSON。"""
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False,
                              indent=2), encoding="utf-8")
    tmp.replace(p)


def is_number_string(s: str) -> bool:
    """判断字符串是否为纯数字。"""
    return bool(NUM_RE.match(s))


def normalize_text(s: str) -> str:
    """规范化文本。"""
    return WS_RE.sub(" ", s.strip())


def normalize_abstract(s: str) -> str:
    """规范化 abstract。"""
    return normalize_text(s)


def resolve_path(base_dir: Path, maybe_relative: str) -> Path:
    """解析相对路径。"""
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def is_jpeg(p: Path) -> bool:
    """判断是否为 jpeg。"""
    return p.suffix.lower() in [".jpg", ".jpeg"]


def save_screen_as_jpeg(src: Path, dst: Path) -> None:
    """保存截图到 dst。"""
    if not src.exists():
        raise FileNotFoundError(f"screen not found: {src}")

    if is_jpeg(src):
        shutil.copy2(src, dst)
        return

    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise ValueError(
            f"screen must be JPEG. got={src.name}. install Pillow to auto-convert.") from e

    with Image.open(src) as im:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(dst, format="JPEG", quality=95, optimize=True)


def read_file_as_data_url_jpeg(img_path: Path, max_size: int = 512) -> str:
    """读取图片 -> 缩放至 max_size -> 转为 data URL (节省 Token)"""
    if not img_path.exists():
        return ""
    try:
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            # 缩放逻辑
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            data = buffer.getvalue()
            
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        # 降级：如果压缩失败，读原图
        data = img_path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """提取 JSON 对象。"""
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start: end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


# =========================================================
# 2) 动作类型 & 结构化 act
# =========================================================

class ActType:
    """动作类型枚举。"""
    CLICK = "click"
    INPUT = "input"
    SWIPE = "swipe"
    LONGPRESS = "longpress"
    # 新增系统动作
    BACK = "back"
    HOME = "home"
    ENTER = "enter"
    OTHER = "other"
    # 新增打开应用动作
    OPEN = "open"


def act_tag_from_act_obj(act_obj: Dict[str, Any]) -> str:
    """
    将 act_obj 映射为归类标签用于 abstracts.json。
    """
    t = act_obj.get("act_type")
    if t in (ActType.CLICK, ActType.LONGPRESS):
        return "point"
    if t == ActType.SWIPE:
        return "pointto"
    if t == ActType.INPUT:
        return "type"
    # 新增：系统动作统一归类为 press
    if t in (ActType.BACK, ActType.HOME, ActType.ENTER):
        return "press"

    if t == ActType.OPEN:
        return "open"
    return "other"


def parse_assistant_action(a: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 assistant 原始 JSON 解析成统一结构：
    - back/home/enter: {"press": "back"}
    - click: {"point":[x,y]}
    - input: {"type":"text"} -> {"act_type":"input", "text":"text"}
    """
    # 1. 优先处理系统级按键 (back, home, enter)
    if "press" in a:
        val = str(a["press"]).strip().lower()
        if val == "back":
            return {"act_type": ActType.BACK, "press": "back"}
        if val == "home":
            return {"act_type": ActType.HOME, "press": "home"}
        if val == "enter":
            return {"act_type": ActType.ENTER, "press": "enter"}

    # 2. 处理输入
    if "type" in a and isinstance(a["type"], str):
        val = a["type"].strip()
        return {"act_type": ActType.INPUT, "text": val}

    # 3. 处理手势
    if "point" in a and "to" in a:
        return {"act_type": ActType.SWIPE, "point": a.get("point"), "to": a.get("to")}

    if "point" in a and "duration" in a:
        return {"act_type": ActType.LONGPRESS, "point": a.get("point"), "duration": a.get("duration")}

    if "point" in a:
        return {"act_type": ActType.CLICK, "point": a.get("point")}

    if "open" in a:
        return {"act_type": ActType.OPEN, "open": a.get("open")}

    return {"act_type": ActType.OTHER, "raw": a}


def get_point(act_obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    p = act_obj.get("point")
    if isinstance(p, list) and len(p) == 2:
        return float(p[0]), float(p[1])
    return None


def get_swipe_xyxy(act_obj: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    if act_obj.get("act_type") != ActType.SWIPE:
        return None
    p = act_obj.get("point")
    t = act_obj.get("to")
    if isinstance(p, list) and len(p) == 2 and isinstance(t, list) and len(t) == 2:
        return float(p[0]), float(p[1]), float(t[0]), float(t[1])
    return None


def float_close(a: float, b: float, eps: float = 1e-3) -> bool:
    return abs(a - b) <= eps


def swipe_close(s1: Tuple[float, float, float, float], s2: Tuple[float, float, float, float],
                eps: float = 1e-3) -> bool:
    return (
            float_close(s1[0], s2[0], eps)
            and float_close(s1[1], s2[1], eps)
            and float_close(s1[2], s2[2], eps)
            and float_close(s1[3], s2[3], eps)
    )


# =========================================================
# 3) 语义单元
# =========================================================

@dataclass
class SemanticUnit:
    thought: str
    abstract: str
    act_obj: Dict[str, Any]
    ui_tree: Any
    ui_path: Path
    screen_path: Path
    next_screen_path: Path
    next_ui_path: Path


# =========================================================
# 4) 命中检测
# =========================================================

def hit_same_component_on_current_ui(current_ui_tree: Any, new_act_obj: Dict[str, Any],
                                     old_act_obj: Dict[str, Any]) -> bool:
    p1 = get_point(new_act_obj)
    p2 = get_point(old_act_obj)
    if p1 is None or p2 is None:
        return False
    x1, y1 = p1
    x2, y2 = p2
    try:
        return same_ui_hit(current_ui_tree, x1, y1, x2, y2) == 1
    except Exception:
        return False


# =========================================================
# 5) Qwen-VL 语义判断
# =========================================================

def call_qwen_vl_semantic_judge(unit: SemanticUnit, ref_variant: Dict[str, Any]) -> Dict[str, Any]:
    # 1. 读取并压缩四张图 (max_size=512)
    unit_screen = read_file_as_data_url_jpeg(unit.screen_path, max_size=512)
    unit_next = read_file_as_data_url_jpeg(unit.next_screen_path, max_size=512)
    ref_screen = read_file_as_data_url_jpeg(Path(ref_variant["screen_path"]), max_size=512)
    ref_next = read_file_as_data_url_jpeg(Path(ref_variant["next_screen_path"]), max_size=512)

    # 完整保留原有的 prompt_text
    prompt_text = f"""你是移动端 UI 自动化的“操作同一性”判定器。
请判断【当前 step】与【候选 action】是否为“同一操作”。

“同一操作”的严格定义：
1) 完成的是同一件事（同一业务意图/同一功能目标）。
2) 发生在同一语境/同一页面模块中。
3) 操作完成后的结果类型一致。

你会看到：
【当前 step】
- abstract: {unit.abstract}


【候选 action】
- abstract: {ref_variant.get("abstract", "")}

请只输出严格 JSON：
{{"same": true/false, "confidence": 0.0-1.0, "brief_reason": "原因"}}
"""

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": unit_screen}},
            {"type": "image_url", "image_url": {"url": unit_next}},
            {"type": "image_url", "image_url": {"url": ref_screen}},
            {"type": "image_url", "image_url": {"url": ref_next}},
            {"type": "text", "text": prompt_text},
        ]}
    ]

    # 2. 调用本地 Wrapper
    content = llm_wrapper.predict(messages, temperature=0.01)
    
    if not content:
        return {"same": False, "confidence": 0.0, "brief_reason": "LLM call failed"}

    obj = extract_first_json_obj(content)
    if obj is None:
        return {"same": False, "confidence": 0.0, "brief_reason": "Invalid JSON"}
    return obj


def qwen_vl_semantic_same(unit: SemanticUnit, ref_variant: Dict[str, Any]) -> bool:
    result = call_qwen_vl_semantic_judge(unit, ref_variant)
    return bool(result.get("same", False))


# =========================================================
# 6) ActionsBank
# =========================================================

class ActionsBank:
    def __init__(self, root: Path):
        self.root = root
        self.actions_dir = self.root / "actions"
        self.abstracts_path = self.actions_dir / "abstracts.json"
        self.embedder = SentenceTransformer("gte", local_files_only=True)  # 语义向量模型（需预先下载好）

    def _cosine_sim(self, query_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        candidate_norms = candidate_vecs / \
            (np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-12)
        return candidate_norms @ query_norm

    def _load_abstracts_maps(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        data = read_json(self.abstracts_path)
        if not isinstance(data, dict): return {}, {}
        if isinstance(data.get("abstract_map"), dict) and isinstance(data.get("act_map"), dict):
            return data["abstract_map"], data["act_map"]
        return {}, {}

    def _save_abstracts_maps(self, abstract_map: Dict[str, str], act_map: Dict[str, str]) -> None:
        atomic_write_json(self.abstracts_path, {
            "schema_version": "1.1",
            "abstract_map": abstract_map,
            "act_map": act_map,
            "updated_at": now_iso_utc()
        })

    def _list_numeric_dirs(self, p: Path) -> List[int]:
        out: List[int] = []
        if not p.exists(): return out
        for c in p.iterdir():
            if c.is_dir() and is_number_string(c.name):
                out.append(int(c.name))
        return out

    def _next_action_id(self) -> str:
        nums = self._list_numeric_dirs(self.actions_dir)
        return str((max(nums) + 1) if nums else 1)

    def _next_variant_id(self, action_dir: Path) -> str:
        nums = self._list_numeric_dirs(action_dir)
        return str((max(nums) + 1) if nums else 1)

    # def match_action_ids_by_abstract(self, abstract: str, act_obj: Dict[str, Any]) -> List[str]:
    #     abstract_map, act_map = self._load_abstracts_maps()
    #     abs_key = normalize_abstract(abstract)
    #     tag_key = act_tag_from_act_obj(act_obj)
    #     hits = []
    #     for aid, abs_text in abstract_map.items():
    #         if normalize_abstract(abs_text) == abs_key and str(act_map.get(aid, "")) == tag_key:
    #             hits.append(aid)
    #     hits.sort(key=lambda x: int(x) if x.isdigit() else 10 ** 9)
    #     return hits

    def match_action_ids_by_abstract(self, abstract: str, act_obj: Dict[str, Any], threshold: float = 0.98) -> List[str]:
        abstract_map, act_map = self._load_abstracts_maps()
        abs_key = normalize_abstract(abstract)
        tag_key = act_tag_from_act_obj(act_obj)

        candidate_ids = []
        candidate_texts = []
        for aid, tag in act_map.items():
            if str(tag) == tag_key:
                candidate_ids.append(aid)
                candidate_texts.append(abstract_map[aid])

        if not candidate_ids:
            return []

        # 语义向量化
        # 将当前输入的 abstract 转为向量
        query_vec = self.embedder.encode(abstract, convert_to_numpy=True)
        # 将库中所有候选 abstract 转为向量 (如果库很大，建议后续引入缓存机制)
        corpus_vecs = self.embedder.encode(
            candidate_texts, convert_to_numpy=True)

        # 计算相似度
        sims = self._cosine_sim(query_vec, corpus_vecs)

        # 排序并取 Top K
        # argsort 是从小到大，所以取负号
        sorted_indices = np.argsort(-sims)
        TOP_K = 3
        top3_idx = sorted_indices[:TOP_K]
        top3_scores = sims[top3_idx]

        # 筛选超过阈值的
        candidate_global_indices = [
            idx for idx, score in zip(top3_idx, top3_scores) if score > threshold
        ]

        # 将候选索引转换为对应的 aid 列表并返回
        result_ids = [candidate_ids[idx] for idx in candidate_global_indices]
        
        # for aid, abs_text in abstract_map.items():
        #     if normalize_abstract(abs_text) == abs_key and str(act_map.get(aid, "")) == tag_key:
        #         hits.append(aid)
        # hits.sort(key=lambda x: int(x) if x.isdigit() else 10 ** 9)
        return result_ids

    def load_first_variant(self, action_id: str) -> Dict[str, Any]:
        action_dir = self.actions_dir / action_id
        vids = self._list_numeric_dirs(action_dir)
        if not vids: raise FileNotFoundError(f"action {action_id} has no variants")
        vid = str(min(vids))
        vdir = action_dir / vid
        act_json = read_json(vdir / "act.json")
        thought_json = read_json(vdir / "thought.json")
        return {
            "variant_id": vid,
            "abstract": act_json.get("abstract", ""),
            "act_obj": act_json.get("act_obj", {}),
            "thought": thought_json.get("thought", ""),
            "screen_path": str(vdir / "screen.jpeg"),
            "next_screen_path": str(vdir / "next_screen.jpeg"),
            "UI_path": str(vdir / "UI.json"),
        }

    def iter_variant_actobjs(self, action_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        action_dir = self.actions_dir / action_id
        if not action_dir.exists(): return []
        out: List[Tuple[str, Dict[str, Any]]] = []
        for n in sorted(self._list_numeric_dirs(action_dir)):
            vid = str(n)
            act_path = action_dir / vid / "act.json"
            if act_path.exists():
                out.append((vid, read_json(act_path).get("act_obj", {})))
        return out

    def count_variants(self, action_id: str) -> int:
        return len(self._list_numeric_dirs(self.actions_dir / action_id))

    def write_variant(self, action_id: str, unit: SemanticUnit) -> str:
        action_dir = self.actions_dir / action_id
        ensure_dir(action_dir)
        variant_id = self._next_variant_id(action_dir)
        vdir = action_dir / variant_id
        ensure_dir(vdir)
        atomic_write_json(vdir / "thought.json", {"thought": unit.thought, "created_at": now_iso_utc()})
        atomic_write_json(vdir / "act.json",
                          {"abstract": unit.abstract, "act_obj": unit.act_obj, "created_at": now_iso_utc()})
        shutil.copy2(unit.ui_path, vdir / "UI.json")
        # ========== 【新增：拷贝执行后的终态 UI 树】 ==========
        if hasattr(unit, 'next_ui_path') and unit.next_ui_path and unit.next_ui_path.exists():
            shutil.copy2(unit.next_ui_path, vdir / "next_UI.json")
        # ===================================================
        save_screen_as_jpeg(unit.screen_path, vdir / "screen.jpeg")
        save_screen_as_jpeg(unit.next_screen_path, vdir / "next_screen.jpeg")
        return variant_id

    def create_new_action_space(self, unit: SemanticUnit) -> Tuple[str, str]:
        action_id = self._next_action_id()
        variant_id = self.write_variant(action_id, unit)
        abstract_map, act_map = self._load_abstracts_maps()
        abstract_map[action_id] = unit.abstract
        act_map[action_id] = act_tag_from_act_obj(unit.act_obj)
        self._save_abstracts_maps(abstract_map, act_map)
        return action_id, variant_id

    def upsert_variant_in_action(self, action_id: str, unit: SemanticUnit) -> str:
        """在 action 内部复用或新增变体。"""
        new_type = unit.act_obj.get("act_type")
        existing = self.iter_variant_actobjs(action_id)

        #【新增】处理 open 动作
        if new_type == ActType.OPEN:
            # 兼容处理：有的指令是 {"open": "微信"}，有的是 {"package": "com.tencent.mm"}
            # 我们只要内容一致就复用
            target = unit.act_obj.get("package") or unit.act_obj.get("open")
            
            for vid, old_act in existing:
                if old_act.get("act_type") == ActType.OPEN:
                    old_target = old_act.get("package") or old_act.get("open")
                    # 如果打开的目标一致，直接复用旧的 variant_id
                    if old_target == target:
                        return vid
            # 没找到就存新的
            return self.write_variant(action_id, unit)

        # 1. 系统动作去重 (BACK, HOME, ENTER)
        if new_type in (ActType.BACK, ActType.HOME, ActType.ENTER):
            new_key = unit.act_obj.get("press")
            for vid, old_act in existing:
                if old_act.get("act_type") == new_type and old_act.get("press") == new_key:
                    return vid
            return self.write_variant(action_id, unit)

        # 2. 点击/长按去重
        if new_type in (ActType.CLICK, ActType.LONGPRESS):
            for vid, old_act in existing:
                if old_act.get("act_type") in (ActType.CLICK, ActType.LONGPRESS):
                    if hit_same_component_on_current_ui(unit.ui_tree, unit.act_obj, old_act):
                        return vid
            return self.write_variant(action_id, unit)

        # 3. 输入去重
        if new_type == ActType.INPUT:
            new_text = unit.act_obj.get("text")
            for vid, old_act in existing:
                if old_act.get("act_type") == ActType.INPUT and old_act.get("text") == new_text:
                    return vid
            return self.write_variant(action_id, unit)

        # 4. 滑动去重
        if new_type == ActType.SWIPE:
            new_sw = get_swipe_xyxy(unit.act_obj)
            for vid, old_act in existing:
                if old_act.get("act_type") == ActType.SWIPE:
                    old_sw = get_swipe_xyxy(old_act)
                    if new_sw and old_sw and swipe_close(new_sw, old_sw):
                        return vid
            return self.write_variant(action_id, unit)

        return self.write_variant(action_id, unit)


# =========================================================
# 7) TasksWriter & UnifiedUpdater (略，保持原有逻辑)
# =========================================================

class TasksWriter:
    def __init__(self, root: Path):
        self.root = root
        self.tasks_dir = self.root / "tasks"
        self.part_reuse_tasks_dir = self.root / "part_reuse_tasks"
        self.querys_path = self.tasks_dir / "querys.json"
        self.part_reuse_querys_path = self.part_reuse_tasks_dir / "querys.json"
        ensure_dir(self.tasks_dir)
        ensure_dir(self.part_reuse_tasks_dir)
        if not self.querys_path.exists():
            atomic_write_json(self.querys_path, [])
        if not self.part_reuse_querys_path.exists():
            atomic_write_json(self.part_reuse_querys_path, [])

    def _list_numeric_dirs(self, p: Path) -> List[int]:
        out: List[int] = []
        if not p.exists():
            return out
        for c in p.iterdir():
            if c.is_dir() and is_number_string(c.name):
                out.append(int(c.name))
        return out

    def _next_task_id(self, eval_result: bool) -> str:
        if eval_result:
            nums = self._list_numeric_dirs(self.tasks_dir)
        else: 
            nums = self._list_numeric_dirs(self.part_reuse_tasks_dir)
        return str((max(nums) + 1) if nums else 1)

    def _load_querys_list(self, eval_result: bool) -> List[Dict[str, Any]]:
        if eval_result:
            data = read_json(self.querys_path)
        else:
            data = read_json(self.part_reuse_querys_path)
        return data if isinstance(data, list) else []

    def _save_querys_list(self, items: List[Dict[str, Any]], eval_result: bool) -> None:
        if eval_result:
            atomic_write_json(self.querys_path, items)
        else:
            atomic_write_json(self.part_reuse_querys_path, items)

    def create_task(self, query: str, eval_result: bool) -> str:
        task_id = self._next_task_id(eval_result)
        if eval_result:
            ensure_dir(self.tasks_dir / task_id)
        else:
            ensure_dir(self.part_reuse_tasks_dir / task_id)
        items = self._load_querys_list(eval_result)
        items.append({"task_id": task_id, "query": query,
                     "created_at": now_iso_utc()})
        self._save_querys_list(items, eval_result)
        return task_id

    def _next_step_number(self, task_dir: Path) -> int:
        max_n = 0
        for p in task_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".json" and p.stem.isdigit():
                max_n = max(max_n, int(p.stem))
        return max_n + 1

    def write_step(self, task_id: str, thought: str, action_id: str, variant_num: int, app_id: str, variant_id: str, eval_result: bool = True) -> int:
        if eval_result:
            task_dir = self.tasks_dir / task_id
        else:
            task_dir = self.part_reuse_tasks_dir / task_id
        k = self._next_step_number(task_dir)
        atomic_write_json(
            task_dir / f"{k}.json",
            {
                "thought": thought,
                "action_id": int(action_id) if action_id.isdigit() else action_id,
                # 【新增】记录精准的变体索引
                "variant_id": int(variant_id) if str(variant_id).isdigit() else variant_id,
                "variant_num": int(variant_num),
                "app_id": int(app_id)
            },
        )
        return k

    def sync_variant_num_in_history(self, action_id: str, new_total: int, app_id: str, eval_result: bool = True):
        target_aid = int(action_id) if str(action_id).isdigit() else action_id
        if eval_result:
            task_dirs = self.tasks_dir.iterdir()
        else:
            task_dirs = self.part_reuse_tasks_dir.iterdir()
        for task_dir in task_dirs:
            if not (task_dir.is_dir() and is_number_string(task_dir.name)):
                continue
            for step_file in task_dir.glob("*.json"):
                if not step_file.stem.isdigit():
                    continue
                try:
                    data = read_json(step_file)
                    if data.get("app_id") == app_id and data.get("action_id") == target_aid and data.get("variant_num") != new_total:
                        data["variant_num"] = new_total
                        atomic_write_json(step_file, data)
                except Exception:
                    continue


class UnifiedUpdater:
    def __init__(self, root: Path, threshold: float = 0.98):
        self.root = Path(root)
        self.actions = ActionsBank(self.root)
        self.tasks = TasksWriter(self.root)
        self.apps_json = os.path.join(self.root, "apps.json")
        self.app_num = 0
        self.threshold = threshold

    def get_or_add_app_id(self, package_name: str) -> str:
        # 1. 读取 json 文件

        # 如果文件不存在，先创建
        if not os.path.exists(self.apps_json):
            with open(self.apps_json, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=4)

        with open(self.apps_json, "r", encoding="utf-8") as f:
            apps_dict = json.load(f)

        # 2. 查找是否已存在该 package_name
        for app_id, pkg in apps_dict.items():
            if pkg == package_name:
                return app_id

        # 3. 不存在：生成新编号
        if apps_dict:
            self.app_num = max(map(int, apps_dict.keys()))
        else:
            self.app_num = 0

        self.app_num += 1
        apps_dict[str(self.app_num)] = package_name

        # 4. 写回 json 文件
        with open(self.apps_json, "w", encoding="utf-8") as f:
            json.dump(apps_dict, f, ensure_ascii=False, indent=4)

        return str(self.app_num)

    def ingest_units(self, query: str, units: List[SemanticUnit], eval_result: bool = True) -> str:
        task_id = self.tasks.create_task(query, eval_result)
        for unit in units:

            appresolver = AppResolver(Path("./configs"))

            package_name = appresolver.resolve_package(unit.ui_tree, unit.screen_path, unit.thought)

            app_id = self.get_or_add_app_id(package_name)

            self.actions.root = Path(os.path.join(self.root, package_name))
            self.actions.actions_dir = Path(os.path.join(self.actions.root, "actions"))
            self.actions.abstracts_path = Path(os.path.join(self.actions.actions_dir, "abstracts.json"))
            ensure_dir(self.actions.actions_dir)
            if not self.actions.abstracts_path.exists():
                atomic_write_json(self.actions.abstracts_path, {
                    "schema_version": "1.1",
                    "abstract_map": {},
                    "act_map": {}
                })

            # 特判是否是open或press
            tag_key = act_tag_from_act_obj(unit.act_obj)
            print(unit.act_obj)
            print(tag_key)
            '''
            if tag_key == "open" or tag_key == "press":
                self.tasks.write_step(task_id, unit.thought, "0", 0, app_id)
                continue
            '''

            candidates = self.actions.match_action_ids_by_abstract(unit.abstract, unit.act_obj, threshold=self.threshold)
            chosen_action_id: Optional[str] = None
            if candidates:
                for aid in candidates:
                    ref = self.actions.load_first_variant(aid)
                    if "type" == act_tag_from_act_obj(unit.act_obj) and "type" == act_tag_from_act_obj(ref.get("act_obj", {})):
                        if unit.act_obj.get("text") != ref.get("act_obj", {}).get("text"):
                            continue
                    if "pointto" == act_tag_from_act_obj(unit.act_obj) and "pointto" == act_tag_from_act_obj(ref.get("act_obj", {})):
                        if unit.act_obj != ref.get("act_obj", {}):
                            continue
                    ui_match = validity_check_for_ui_match(
                        unit.ui_path, ref.get("UI_path"), unit, ref, threshold=self.threshold)
                    with open("ui_match_debug.txt", "a", encoding="utf-8") as f:
                        f.write(f"Comparing with action_id={aid}, ui_match={ui_match}\n")
                        f.write(f"Unit UI path: {unit.ui_path}, Ref UI path: {ref.get('UI_path')}\n")
                        f.write(f"Unit abstract: {unit.abstract}\n")
                        f.write(f"Ref abstract: {ref.get('abstract', '')}\n")
                        f.write('---\n')
                    if ui_match:
                        chosen_action_id = aid
                        break
                    if qwen_vl_semantic_same(unit, ref) or ui_match:
                        chosen_action_id = aid
                        break

            # 【修改点开始】分别捕获新建或复用时产生的 variant_id
            if chosen_action_id is None:
                action_id, variant_id = self.actions.create_new_action_space(unit)
            else:
                action_id = chosen_action_id
                variant_id = self.actions.upsert_variant_in_action(action_id, unit)
            # 【修改点结束】

            variant_num = self.actions.count_variants(action_id)
            
            # 【修改点】将 variant_id 作为新参数传入
            self.tasks.write_step(task_id, unit.thought, action_id, variant_num, app_id, variant_id, eval_result)
            self.tasks.sync_variant_num_in_history(action_id, variant_num, app_id, eval_result)
        return task_id


# =========================================================
# 8) Longchain 解析
# =========================================================

def extract_user_fields(user_msg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    question, screen_path, ui_path = None, None, None
    content = user_msg.get("content", [])
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict): continue
            t = item.get("type")
            if t == "text":
                m = QUESTION_RE.search(item.get("text", ""))
                if m: question = m.group(1).strip()
            elif t == "image_path":
                screen_path = item.get("image_path", {}).get("path")
            elif t == "ui_path":
                ui_path = item.get("ui_path", {}).get("path")
    return question, screen_path, ui_path


def parse_longchain_to_units(longchain: Any, base_dir: Path) -> Tuple[str, List[SemanticUnit]]:
    if not isinstance(longchain, list): raise ValueError("longchain must be a list")
    query, units = None, []
    pending_screen, pending_ui, pending_action = None, None, None

    for msg in longchain:
        role = msg.get("role")
        if role == "user":
            q, screen_s, ui_s = extract_user_fields(msg)
            if query is None and q: query = q
            if not screen_s or not ui_s: continue
            screen_p, ui_p = resolve_path(base_dir, screen_s), resolve_path(base_dir, ui_s)
            if pending_action:
                units.append(SemanticUnit(
                    thought=str(pending_action.get("thought", "")),
                    abstract=str(pending_action.get("abstract", "")),
                    act_obj=parse_assistant_action(pending_action),
                    ui_tree=read_json(pending_ui),
                    ui_path=pending_ui, screen_path=pending_screen, next_screen_path=screen_p,
                    next_ui_path=ui_p
                ))
                pending_action = None
            pending_screen, pending_ui = screen_p, ui_p
        elif role == "assistant":
            a = json.loads(msg.get("content", "{}"))
            if a.get("STATUS") == "finish": break
            pending_action = a
    return query or "", units


def update(root: str, longchain_json: str, eval_result: bool = True) -> str:
    longchain_path = Path(longchain_json).resolve()
    longchain = read_json(longchain_path)
    query, units = parse_longchain_to_units(longchain, longchain_path.parent)
    updater = UnifiedUpdater(root)
    return updater.ingest_units(query, units, eval_result)


if __name__ == "__main__":
    # 示例调用
    update("../../data", "../../modified_full_history.json", eval_result=True)
    pass
