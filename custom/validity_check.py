"""Helpers for determining whether two UI captures represent the same page."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
from typing import Any, Dict
from custom.detection.screenshotdetect import is_same_page, THRESHOLD
from custom.detection.keyboarddetect import detect_ime_keyboard


# ---------------------------------------------------------------------------
# Raster comparison helpers -------------------------------------------------


def _images_match(new_image: str, old_image: str, threshold: int = THRESHOLD) -> bool:
    """Return True when the two screenshots are considered equivalent."""

    same, _ = is_same_page(new_image, old_image, threshold)
    return same


# ---------------------------------------------------------------------------
# UI tree parsing and comparison (trimmed version of uicompare.py) ----------


HIT_NONE = 0
HIT_TRANSPARENT = 1
HIT_DEFAULT = 2


@dataclass(slots=True)
class _FastNode:
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    has_bounds: bool = False
    bounds_raw: str = ""
    checked: bool = False
    checkable: bool = False
    z: float = 0.0
    z_parsed: bool = False
    z_raw: str = ""

    visible: bool = True
    enabled: bool = True
    clickable: bool = False
    opacity: float = 1.0
    long_clickable: bool = False
    hit: int = HIT_DEFAULT
    hit_raw: str = ""
    backgroundColor: str = ""
    node_type: str = ""
    page_path: str = ""
    node_id: str = ""

    children: List["_FastNode"] = None


def _get_node_details(node: _FastNode) -> dict:
    """提取节点用于比对的所有关键属性"""
    if node is None:
        return {"status": "NOT_FOUND"}
    return {
        "id": node.node_id,
        "type": node.node_type,
        "bounds": node.bounds_raw,
        "page_path": node.page_path,
        "z_index": node.z_raw,
        "visible": node.visible,
        "enabled": node.enabled,
        "clickable": node.clickable,
        "long_clickable": node.long_clickable,
        "hit_behavior": node.hit_raw,
        "checked": node.checked,       # 新增
        "checkable": node.checkable,   # 新增
    }


def _to_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    return default


def _to_float(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return default


def _parse_bounds(raw: str) -> Tuple[bool, int, int, int, int]:
    if not raw or len(raw) < 11 or raw[0] != "[":
        return False, 0, 0, 0, 0

    idx = 1
    end = len(raw)

    def read_int() -> int:
        nonlocal idx
        if idx >= end:
            raise ValueError
        negative = False
        if raw[idx] == "-":
            negative = True
            idx += 1
        if idx >= end or not raw[idx].isdigit():
            raise ValueError
        value = 0
        while idx < end and raw[idx].isdigit():
            value = value * 10 + (ord(raw[idx]) - 48)
            idx += 1
        return -value if negative else value

    try:
        x1 = read_int()
        if idx >= end or raw[idx] != ",":
            return False, 0, 0, 0, 0
        idx += 1
        y1 = read_int()
        if idx >= end or raw[idx] != "]":
            return False, 0, 0, 0, 0
        idx += 1
        if idx >= end or raw[idx] != "[":
            return False, 0, 0, 0, 0
        idx += 1
        x2 = read_int()
        if idx >= end or raw[idx] != ",":
            return False, 0, 0, 0, 0
        idx += 1
        y2 = read_int()
        if idx >= end or raw[idx] != "]":
            return False, 0, 0, 0, 0
        if x2 <= x1 or y2 <= y1:
            return False, 0, 0, 0, 0
        return True, x1, y1, x2, y2
    except ValueError:
        return False, 0, 0, 0, 0


def _is_blank(value: Optional[str]) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _lower(value: Optional[str]) -> Optional[str]:
    return value.lower() if isinstance(value, str) else value


def _from_json_fast(payload: dict) -> _FastNode:
    node = _FastNode()

    attrs = payload.get("attributes", {})
    if isinstance(attrs, dict):
        node.node_type = attrs.get("type") or ""
        node.page_path = attrs.get("pagePath") or ""
        node.node_id = attrs.get("id") or ""

        bounds_raw = attrs.get("bounds")
        if isinstance(bounds_raw, str):
            node.bounds_raw = bounds_raw
            ok, x1, y1, x2, y2 = _parse_bounds(bounds_raw)
            if ok:
                node.has_bounds = True
                node.x1, node.y1, node.x2, node.y2 = x1, y1, x2, y2

        if "zIndex" in attrs:
            z_value = attrs["zIndex"]
            if isinstance(z_value, str):
                node.z_raw = z_value
                try:
                    node.z = float(z_value)
                    node.z_parsed = True
                except ValueError:
                    node.z_parsed = False
            else:
                node.z = _to_float(z_value, 0.0)
                node.z_parsed = True

        if "visible" in attrs:
            node.visible = _to_bool(attrs["visible"], True)
        if "enabled" in attrs:
            node.enabled = _to_bool(attrs["enabled"], True)
        if "clickable" in attrs:
            node.clickable = _to_bool(attrs["clickable"], False)
        if "longClickable" in attrs:
            node.long_clickable = _to_bool(attrs["longClickable"], False)
        if "opacity" in attrs:
            node.opacity = float(_to_float(attrs["opacity"], 1.0))
        if "backgroundColor" in attrs:
            node.backgroundColor = str(attrs.get("backgroundColor", ""))

        hit_behavior = attrs.get("hitTestBehavior")
        if isinstance(hit_behavior, str):
            node.hit_raw = hit_behavior
            lowered = hit_behavior.lower()
            if "none" in lowered:
                node.hit = HIT_NONE
            elif "transparent" in lowered:
                node.hit = HIT_TRANSPARENT
            else:
                node.hit = HIT_DEFAULT
        else:
            node.hit_raw = ""
            node.hit = HIT_DEFAULT

    raw_children = payload.get("children")
    if isinstance(raw_children, list):
        node.children = [_from_json_fast(child) for child in raw_children]
    else:
        node.children = []

    node.children.sort(key=lambda child: child.z)
    return node


def _is_actionable(node: _FastNode) -> bool:
    return bool(node.clickable or node.long_clickable)


def _contains(node: _FastNode, x: int, y: int) -> bool:
    return node.has_bounds and node.x1 <= x < node.x2 and node.y1 <= y < node.y2


def _hit_test(node: _FastNode, x: int, y: int) -> Optional[_FastNode]:
    if not _contains(node, x, y):
        return None
    if not node.visible or node.opacity <= 0.0:  #判定是否有效
        return None

    top: Optional[_FastNode] = None
    for child in node.children:
        candidate = _hit_test(child, x, y) 
        if candidate is not None:
            top = candidate
    if top is not None:
        return top

    if node.hit in {HIT_NONE, HIT_TRANSPARENT}:
        return None
    if node.clickable or node.hit == HIT_DEFAULT:
        return node
    return None


'''
def _hit_test_actionable(node: _FastNode, x: int, y: int) -> Optional[_FastNode]:
    if not _contains(node, x, y):
        return None
    if not node.visible or node.opacity <= 0.0:
        return None
    if node.hit in {HIT_NONE, HIT_TRANSPARENT}:
        return None

    best: Optional[_FastNode] = node if _is_actionable(node) else None

    # 继续往更深处找“可操作”的命中（更深者优先）
    for child in node.children:
        cand = _hit_test_actionable(child, x, y)
        if cand is not None:
            best = cand

    return best
'''


def _hit_test_actionable(node: _FastNode, x: int, y: int) -> Optional[_FastNode]:
    if not _contains(node, x, y):
        return None
    if not node.visible or node.opacity <= 0.0:
        return None

    # 优先递归查找子节点（深层优先）
    best_child = None
    if node.children:
        for child in node.children:
            cand = _hit_test_actionable(child, x, y)
            if cand is not None:
                best_child = cand

    if best_child is not None:
        return best_child

    # 如果子节点没有命中，再看当前节点
    # HIT_NONE 表示节点本身绝不响应命中
    if node.hit == HIT_NONE:
        return None

    # 如果是可操作节点，则返回自身
    if _is_actionable(node):
        return node

    # 如果节点是 HIT_DEFAULT 但不可点击，在 actionable 搜索中不作为终点
    return None


def _compute_screen_size(node: _FastNode, current: Tuple[int, int]) -> Tuple[int, int]:
    max_x, max_y = current
    if node.has_bounds:
        max_x = max(max_x, node.x2)
        max_y = max(max_y, node.y2)
    for child in node.children:
        max_x, max_y = _compute_screen_size(child, (max_x, max_y))
    return max_x, max_y


def _denorm_to_pixel(root: _FastNode, nx: int, ny: int) -> Tuple[int, int]:
    max_x, max_y = _compute_screen_size(root, (0, 0))
    if max_x <= 0 or max_y <= 0:
        max_x, max_y = 1280, 2800

    def to_pixel(value: int, maximum: int) -> int:
        clamped = min(1000, max(1, value))
        normalized = (clamped - 1) / 999.0
        pixel = int(round(normalized * (maximum - 1)))
        return min(max(pixel, 0), maximum - 1)

    return to_pixel(nx, max_x), to_pixel(ny, max_y)


def _eq_string_loose_raw(a: str, b: str, case_insensitive: bool = False) -> bool:
    a_blank = _is_blank(a)
    b_blank = _is_blank(b)
    if a_blank and b_blank:
        return True
    if a_blank != b_blank:
        return False
    if case_insensitive:
        return _lower(a) == _lower(b)
    return a == b


def _eq_z_loose(a: _FastNode, b: _FastNode) -> bool:
    a_blank = _is_blank(a.z_raw)
    b_blank = _is_blank(b.z_raw)
    if a_blank and b_blank:
        return True
    if a_blank != b_blank:
        return False
    if a.z_parsed and b.z_parsed:
        return abs(a.z - b.z) < 1e-9
    return _lower(a.z_raw) == _lower(b.z_raw)


def _eq_bounds_with_tol(old_node: _FastNode, new_node: _FastNode, tol_ratio: float = 0.05) -> bool:
    a_blank = _is_blank(old_node.bounds_raw)
    b_blank = _is_blank(new_node.bounds_raw)
    if a_blank and b_blank:
        return True
    if a_blank != b_blank:
        return False
    if not (old_node.has_bounds and new_node.has_bounds):
        return False

    width = max(1, old_node.x2 - old_node.x1)
    height = max(1, old_node.y2 - old_node.y1)
    tol_x = max(1.0, tol_ratio * width)
    tol_y = max(1.0, tol_ratio * height)

    def within(delta: int, tolerance: float) -> bool:
        return abs(delta) <= tolerance

    if not within(new_node.x1 - old_node.x1, tol_x):
        return False
    if not within(new_node.x2 - old_node.x2, tol_x):
        return False
    if not within(new_node.y1 - old_node.y1, tol_y):
        return False
    if not within(new_node.y2 - old_node.y2, tol_y):
        return False
    return True


@dataclass(slots=True)
class _CompareReport:
    z_index_equal: bool = False
    type_equal: bool = False
    visible_equal: bool = False
    enabled_equal: bool = False
    clickable_equal: bool = False
    hit_behavior_equal: bool = False
    page_path_equal: bool = False
    bounds_equal_with_tol: bool = False
    long_clickable_equal: bool = False  # 修复：原代码漏了这个定义
    checked_equal: bool = False         # 新增
    checkable_equal: bool = False       # 新增

    @property
    def equal(self) -> bool:
        return (
            self.z_index_equal
            and self.type_equal
            and self.visible_equal
            and self.enabled_equal
            and self.clickable_equal
            and self.hit_behavior_equal
            and self.page_path_equal
            and self.bounds_equal_with_tol
            and self.long_clickable_equal # 修复：加入判断
            and self.checked_equal        # 新增
            and self.checkable_equal      # 新增
        )


def _compare_ui_by_keys(old_node: _FastNode, new_node: _FastNode, bounds_tol_ratio: float = 0.05) -> _CompareReport:
    report = _CompareReport()
    report.z_index_equal = _eq_z_loose(old_node, new_node)
    report.type_equal = _eq_string_loose_raw(
        old_node.node_type, new_node.node_type)
    report.visible_equal = old_node.visible == new_node.visible
    report.enabled_equal = old_node.enabled == new_node.enabled
    report.clickable_equal = old_node.clickable == new_node.clickable
    report.long_clickable_equal = old_node.long_clickable == new_node.long_clickable
    report.checked_equal = old_node.checked == new_node.checked
    report.checkable_equal = old_node.checkable == new_node.checkable
    report.hit_behavior_equal = _eq_string_loose_raw(
        old_node.hit_raw, new_node.hit_raw, True)
    report.page_path_equal = _eq_string_loose_raw(
        old_node.page_path, new_node.page_path)
    report.bounds_equal_with_tol = _eq_bounds_with_tol(
        old_node, new_node, bounds_tol_ratio)
    return report


def _load_ui_tree(path: str) -> _FastNode:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _from_json_fast(payload)


'''
def _ui_matches(old_ui_path: str, new_ui_path: str, nx: int, ny: int) -> bool:
    old_root = _load_ui_tree(old_ui_path)
    new_root = _load_ui_tree(new_ui_path)

    old_x, old_y = _denorm_to_pixel(old_root, nx, ny)
    new_x, new_y = _denorm_to_pixel(new_root, nx, ny)

    # old_hit = _hit_test(old_root, old_x, old_y)
    # new_hit = _hit_test(new_root, new_x, new_y)

    old_hit = _hit_test_actionable(old_root, old_x, old_y)
    new_hit = _hit_test_actionable(new_root, new_x, new_y)

    if old_hit is None or new_hit is None:
        return False

    result = _compare_ui_by_keys(old_hit, new_hit, bounds_tol_ratio=0.05)
    return result.equal
'''


def _ui_matches(old_ui_path: str, new_ui_path: str, nx: int, ny: int) -> bool:
    old_root = _load_ui_tree(old_ui_path)
    new_root = _load_ui_tree(new_ui_path)

    old_x, old_y = _denorm_to_pixel(old_root, nx, ny)
    new_x, new_y = _denorm_to_pixel(new_root, nx, ny)

    # print("\n" + "="*80)
    # print(f"UI COMPARISON START | Coordinate: ({nx}, {ny}) -> Pixels: Old({old_x},{old_y}), New({new_x},{new_y})")
    # print("-" * 80)

    # --- 逻辑 A：尝试匹配“可操作”组件 ---
    old_hit_act = _hit_test_actionable(old_root, old_x, old_y)
    new_hit_act = _hit_test_actionable(new_root, new_x, new_y)

    if old_hit_act is not None and new_hit_act is not None:
        # print("TRYING PATH A: Actionable Node Match")
        report_act = _compare_ui_by_keys(
            old_hit_act, new_hit_act, bounds_tol_ratio=0.05)

        if report_act.equal:
            # print("PATH A RESULT: ✅ SUCCESS (Standard Actionable Match)")
            # print("="*80 + "\n")
            return True
        else:
            haha = 1  # 占位
            # print("PATH A RESULT: ❌ FAIL (Actionable nodes found but attributes differ)")
    else:
        haha = 1  # 占位
        # print("PATH A RESULT: ❌ FAIL (One or both actionable nodes NOT FOUND)")

    # --- 逻辑 B：放宽判定，比对“直接命中”的最深层组件 ---

    # _hit_test 会返回命中位置最深层的可见节点，而不强制要求 clickable 为 true
    old_hit_dir = _hit_test(old_root, old_x, old_y)
    new_hit_dir = _hit_test(new_root, new_x, new_y)

    if old_hit_dir is None or new_hit_dir is None:
        return False

    old_info = _get_node_details(old_hit_dir)
    new_info = _get_node_details(new_hit_dir)

    # 打印属性对比表（针对直接命中的节点）
    # print(f"{'Property':<15} | {'Old Direct Node':<30} | {'New Direct Node':<30}")
    # print("-" * 80)
    # for key in old_info.keys():
    # print(f"{key:<15} | {str(old_info[key]):<30} | {str(new_info[key]):<30}")

    # 获取比对报告
    report_dir = _compare_ui_by_keys(
        old_hit_dir, new_hit_dir, bounds_tol_ratio=0.05)

    # 关键修改：判定复用时，只要关键特征一致即可（忽略 clickable 状态位）
    is_key_features_match = (
        report_dir.type_equal and
        #report_dir.z_index_equal and
        report_dir.visible_equal and
        report_dir.enabled_equal and
        #report_dir.page_path_equal and
        report_dir.bounds_equal_with_tol and
        report_dir.checked_equal and      # 新增：确保状态变化被感知
        report_dir.checkable_equal        # 新增：确保组件属性一致
    )
    '''
    print("-" * 80)
    print("DETAILED KEY FEATURE RESULTS (PATH B):")
    print(f" - Type Match:      {'✅' if report_dir.type_equal else '❌'}")
    print(f" - Bounds Match:    {'✅' if report_dir.bounds_equal_with_tol else '❌'}")
    print(f" - PagePath Match:  {'✅' if report_dir.page_path_equal else '❌'}")
    print(f" - Z-Index Match:   {'✅' if report_dir.z_index_equal else '❌'}")
    '''
    if is_key_features_match:
        # print(f"\nPATH B RESULT: ✅ SUCCESS (Direct components match. Clickable status ignored.)")
        # print("="*80 + "\n")
        return True
    return False

    # print(f"\nFINAL UI MATCH: FALSE (Both Paths Failed)")
    # print("="*80 + "\n")
    return False

# ---------------------------------------------------------------------------
# Public API ---------------------------------------------------------------


def action_code(action: Dict[str, Any]) -> int:
    # 先判 TYPE（输入）
    if "type" in action:
        return 2
    # 再判 point 没有 to（点击/长按）
    if "point" in action and "to" not in action:
        return 1
    # 其他情况
    return 0


def get_xy(action: Dict[str, Any]):
    x, y = action["point"]
    x = int(x)
    y = int(y)
    return x, y


def board_check(new_screenshot_path: str, new_ui_json_path: str) -> bool:
    # 返回键盘检测结果
    if detect_ime_keyboard(new_ui_json_path):
        return True
    return False

def ui_easy_matches(old_ui_path, new_ui_path):
    old_root = _load_ui_tree(old_ui_path)
    new_root = _load_ui_tree(new_ui_path)
    return True

def validity_check(
        old_screenshot_path: str,
        new_screenshot_path: str,
        old_ui_json_path: str,
        new_ui_json_path: str,
        nx: int,
        ny: int,
) -> bool:
    """Return True when both screenshots and UI metadata agree on the same page."""
    '''
    if not _images_match(new_screenshot_path, old_screenshot_path):
        print("像素匹配失败")
        return False
    '''
    res = _ui_matches(old_ui_json_path, new_ui_json_path, int(nx), int(ny))
    if res == False:
        print("UI树匹配失败")
    return res

def validity_check_for_ui_match(old_ui_json_path: str,
                                new_ui_json_path: str,
                                unit,
                                ref_variant: Dict[str, Any],
                                threshold: float = 0.98
                                ) -> bool:

    if not old_ui_json_path or not new_ui_json_path:
        print("UI路径缺失，无法进行UI树比对")
        return False
    
    old_ui_tree = _load_ui_tree(old_ui_json_path)
    new_ui_tree = _load_ui_tree(new_ui_json_path)

    def detect_all_nodes(ui_tree: _FastNode):
        res = [ui_tree]
        if ui_tree.children:
            for child in ui_tree.children:
                res += detect_all_nodes(child)
        return res

    old_ui_nodes = detect_all_nodes(old_ui_tree)
    new_ui_nodes = detect_all_nodes(new_ui_tree)
    total_num = len(old_ui_nodes) + len(new_ui_nodes)

    same = 0

    def dfs(old_ui_tree: _FastNode, new_ui_tree: _FastNode):
        nonlocal same
        if old_ui_tree and new_ui_tree:
            report_act = _compare_ui_by_keys(
                old_ui_tree, new_ui_tree, bounds_tol_ratio=0.05)
            same += report_act.equal

            if old_ui_tree.children and new_ui_tree.children:
                for old_child, new_child in zip(old_ui_tree.children, new_ui_tree.children):
                    dfs(old_child, new_child)

    dfs(old_ui_tree, new_ui_tree)
    similarity = round(same*2 / total_num, 3)
    # print("ui_tree similarity:", round(same*2 / total_num, 2))
    with open("ui_similarity_log.txt", "a", encoding="utf-8") as log_file:
        print("old_ui_json_path: " , old_ui_json_path , " new_ui_json_path: " , new_ui_json_path , " similarity: ", similarity, file=log_file)
    return similarity > threshold

__all__ = ["validity_check"]
