import json
import difflib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# ==========================================
# 1. 核心節點數據結構 (對齊 validity_check.py)
# ==========================================
@dataclass(slots=True)
class CleanNode:
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    bounds_raw: str = ""
    
    z_raw: int = 0
    visible: bool = True
    enabled: bool = True
    clickable: bool = False
    opacity: float = 1.0
    long_clickable: bool = False
    hit_raw: str = ""
    backgroundColor: str = ""
    node_type: str = ""
    page_path: str = ""
    node_id: str = ""
    checked: bool = False
    checkable: bool = False

# ==========================================
# 2. 輔助解析與轉換函數
# ==========================================

def _to_int(value, default: int = 0) -> int:
    if isinstance(value, int): return value
    if isinstance(value, str):
        try: return int(value)
        except ValueError: return default
    return default

def _to_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)): return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}: return True
        if lowered in {"false", "0"}: return False
    return default

def _to_float(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        try: return float(value)
        except ValueError: return default
    if isinstance(value, bool): return 1.0 if value else 0.0
    return default

def _parse_bounds_simple(raw: str) -> Tuple[bool, int, int, int, int]:
    """簡化的 bounds 解析，提取絕對座標 x1, y1, x2, y2"""
    if not raw or len(raw) < 11 or raw[0] != "[":
        return False, 0, 0, 0, 0
    try:
        parts = raw.replace("][", ",").replace("[", "").replace("]", "").split(",")
        if len(parts) == 4:
            x1, y1, x2, y2 = map(int, parts)
            if x2 > x1 and y2 > y1:
                return True, x1, y1, x2, y2
    except Exception:
        pass
    return False, 0, 0, 0, 0

# ==========================================
# 3. UI 樹清洗與扁平化提取
# ==========================================
def extract_key_nodes_from_tree(payload: Dict[str, Any]) -> List[CleanNode]:
    """遞歸遍歷 UI 樹，清洗無效節點並將其拍扁為一維列表"""
    valid_nodes: List[CleanNode] = []
    
    attrs = payload.get("attributes", {})
    if isinstance(attrs, dict):
        # 處理默認值
        visible_raw = attrs.get("visible")
        visible = _to_bool(visible_raw, default=True) if visible_raw is not None else True
        
        bg_color = str(attrs.get("backgroundColor", "")).strip()
        if not bg_color:
            bg_color = "#00000000"
            
        opacity = float(_to_float(attrs.get("opacity"), 1.0))
        
        # 判定有效性：可見 且 不透明 且 背景色不以 #00 開頭
        is_transparent_bg = bg_color.startswith("#00")
        is_valid = visible and (opacity > 0.0) and (not is_transparent_bg)
        
        if is_valid:
            bounds_raw = str(attrs.get("bounds", ""))
            _, x1, y1, x2, y2 = _parse_bounds_simple(bounds_raw)
            
            node = CleanNode(
                x1=x1, y1=y1, x2=x2, y2=y2, bounds_raw=bounds_raw,
                z_raw=_to_int(attrs.get("zIndex")),
                visible=visible,
                enabled=_to_bool(attrs.get("enabled"), True),
                clickable=_to_bool(attrs.get("clickable"), False),
                opacity=opacity,
                long_clickable=_to_bool(attrs.get("longClickable"), False),
                hit_raw=str(attrs.get("hitTestBehavior", "")),
                checked=_to_bool(attrs.get("checked"), False),      # 新增
                checkable=_to_bool(attrs.get("checkable"), False),  # 新增
                backgroundColor=bg_color,
                node_type=str(attrs.get("type", "")),
                page_path=str(attrs.get("pagePath", "")),
                node_id=str(attrs.get("id", ""))
            )
            valid_nodes.append(node)

    # ==========================================
    # 兄弟節點遮擋計算與遞歸過濾
    # ==========================================
    raw_children = payload.get("children", [])
    if isinstance(raw_children, list) and len(raw_children) > 0:
        # 1. 預先提取所有兄弟節點的邊界與 Z-Index 資訊
        siblings_info = []
        for idx, child in enumerate(raw_children):
            attrs = child.get("attributes", {}) if isinstance(child, dict) else {}
            bounds_raw = str(attrs.get("bounds", ""))
            is_valid_bounds, x1, y1, x2, y2 = _parse_bounds_simple(bounds_raw)
            # 這裡依賴我們上一步加好的 _to_int 函數
            z_index = _to_int(attrs.get("zIndex"))
            
            siblings_info.append({
                "child": child,
                "idx": idx,
                "z_index": z_index,
                "is_valid_bounds": is_valid_bounds,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
            
        # 2. 互相校對遮擋關係
        for i, info_i in enumerate(siblings_info):
            is_occluded = False
            
            # 只有當前節點本身有有效坐標，才需要去判斷是否被遮擋
            if info_i["is_valid_bounds"]:
                for j, info_j in enumerate(siblings_info):
                    # 自己不跟自己比，且只跟有有效坐標的兄弟比
                    if i == j or not info_j["is_valid_bounds"]:
                        continue
                        
                    # 判斷 j 是否覆蓋在 i 的上面
                    # 規則：Z-Index 大的在上面；若相同，則原始結構排後面的 (j > i) 在上面
                    is_j_above_i = (info_j["z_index"] > info_i["z_index"]) or \
                                   (info_j["z_index"] == info_i["z_index"] and j > i)
                                   
                    if is_j_above_i:
                        # 判斷 j 的矩形是否完全包裹 (遮蓋) 了 i 的矩形
                        if (info_j["x1"] <= info_i["x1"] and 
                            info_j["y1"] <= info_i["y1"] and 
                            info_j["x2"] >= info_i["x2"] and 
                            info_j["y2"] >= info_i["y2"]):
                            is_occluded = True
                            break # 被任意一個上層兄弟完全遮擋，直接宣佈無效，停止比對
            
            # 3. 如果沒有被完全遮擋，才繼續遞歸，提取它自己和它的後代
            if not is_occluded:
                valid_nodes.extend(extract_key_nodes_from_tree(info_i["child"]))
            
    return valid_nodes

def load_and_clean_ui(json_path: str) -> List[CleanNode]:
    """讀取 JSON 並返回清洗後的扁平化關鍵節點列表"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return extract_key_nodes_from_tree(payload)
    except Exception as e:
        print(f"解析 UI 樹出錯 {json_path}: {e}")
        return []

# ==========================================
# 4. 指紋生成與序列相似度計算
# ==========================================
def _get_node_signature(node: CleanNode) -> str:
    # 将 checked 和 checkable 加入签名串
    return f"{node.node_type}|{node.backgroundColor}|{node.enabled}|{node.clickable}|{node.long_clickable}|{node.opacity}|{node.checked}|{node.checkable}"

def calculate_ui_similarity_ordered(old_ui_path: str, new_ui_path: str) -> float:
    """
    端到端介面：傳入新舊 UI 樹的文件路徑，直接返回 0.0 ~ 1.0 的結構相似度
    """
    old_nodes = load_and_clean_ui(old_ui_path)
    new_nodes = load_and_clean_ui(new_ui_path)
    
    if not old_nodes and not new_nodes:
        return 0.0
        
    # 提取順序指紋列表
    old_signatures = [_get_node_signature(n) for n in old_nodes]
    new_signatures = [_get_node_signature(n) for n in new_nodes]
    
    # 基於 difflib 進行最長公共子序列 (LCS) 匹配，嚴格保證 DOM 結構順序
    matcher = difflib.SequenceMatcher(None, old_signatures, new_signatures)
    return matcher.ratio()

if __name__ == "__main__":
    # 测试用例：请替换成你的实际 JSON 文件路径
    old_ui_path = "old_1.json"  
    new_ui_path = "new_1.json"
    similarity = calculate_ui_similarity_ordered(old_ui_path, new_ui_path)
    print(f"UI 结构相似度: {similarity}")

