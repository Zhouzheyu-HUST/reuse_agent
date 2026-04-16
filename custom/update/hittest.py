import json

HIT_NONE = 0
HIT_TRANSPARENT = 1
HIT_DEFAULT = 2  # Default / Block

class FastNode:
    __slots__ = (
        "x1", "y1", "x2", "y2", "has_bounds", "bounds_raw",
        "z", "z_parsed", "z_raw",
        "visible", "enabled", "clickable", "long_clickable", "opacity",
        "hit", "hit_raw",
        "type", "page_path", "id",
        "children", "parent"  # 关键：增加 parent 引用
    )

    def __init__(self):
        self.x1 = self.y1 = self.x2 = self.y2 = 0
        self.has_bounds = False
        self.bounds_raw = ""
        self.z = 0.0
        self.z_parsed = False
        self.z_raw = ""
        self.visible = True
        self.enabled = True
        self.clickable = False
        self.long_clickable = False
        self.opacity = 1.0
        self.hit = HIT_DEFAULT
        self.hit_raw = ""
        self.type = ""
        self.page_path = ""
        self.id = ""
        self.children = []
        self.parent = None # 初始化父节点为空

def _to_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1"):
            return True
        if s in ("false", "0"):
            return False
    return default

def _to_float(v, default=0.0):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return default
    return default

def parse_bounds_fast(s: str):
    if not s or len(s) < 11 or s[0] != "[":
        return False, 0, 0, 0, 0
    p = 1
    n = len(s)

    def read_int():
        nonlocal p
        if p >= n: raise ValueError
        neg = False
        if s[p] == "-":
            neg = True
            p += 1
        val = 0
        while p < n and s[p].isdigit():
            val = val * 10 + (ord(s[p]) - 48)
            p += 1
        return -val if neg else val

    try:
        x1 = read_int()
        p += 1 # skip ,
        y1 = read_int()
        p += 2 # skip ][
        x2 = read_int()
        p += 1 # skip ,
        y2 = read_int()
        return True, x1, y1, x2, y2
    except Exception:
        return False, 0, 0, 0, 0

def from_json_fast(j, parent=None):
    """
    解析 JSON 构造 FastNode 树，并建立 parent 关联。
    """
    node = FastNode()
    node.parent = parent

    attrs = j.get("attributes", {})
    if isinstance(attrs, dict):
        node.type = attrs.get("type", "") or ""
        node.page_path = attrs.get("pagePath", "") or ""
        node.id = attrs.get("id", "") or ""

        b = attrs.get("bounds")
        if isinstance(b, str):
            node.bounds_raw = b
            ok, x1, y1, x2, y2 = parse_bounds_fast(b)
            if ok:
                node.has_bounds = True
                node.x1, node.y1, node.x2, node.y2 = x1, y1, x2, y2

        # 针对模板 JSON 中字符串形式的布尔值进行转换
        node.clickable = _to_bool(attrs.get("clickable"), False)
        node.long_clickable = _to_bool(attrs.get("longClickable"), False)
        node.visible = _to_bool(attrs.get("visible"), True)
        node.enabled = _to_bool(attrs.get("enabled"), True)
        node.opacity = _to_float(attrs.get("opacity"), 1.0)

        # zIndex 处理
        zv = attrs.get("zIndex")
        if zv is not None and zv != "":
            node.z = _to_float(zv, 0.0)
            node.z_parsed = True

        htb = attrs.get("hitTestBehavior", "")
        if isinstance(htb, str):
            node.hit_raw = htb
            h = htb.lower()
            if "none" in h: node.hit = HIT_NONE
            elif "transparent" in h: node.hit = HIT_TRANSPARENT
            else: node.hit = HIT_DEFAULT

    children_json = j.get("children")
    if isinstance(children_json, list):
        # 递归创建子节点并传入当前节点作为 parent
        node.children = [from_json_fast(cj, node) for cj in children_json]
        node.children.sort(key=lambda c: (0 if not c.z_parsed else 1, c.z if c.z_parsed else 0.0))
    else:
        node.children = []

    return node

def contains(node: FastNode, x: int, y: int) -> bool:
    return node.has_bounds and (node.x1 <= x < node.x2) and (node.y1 <= y < node.y2)

def hit_test(node: FastNode, x: int, y: int):
    """
    深度优先查找视觉上命中最深层的元素。
    """
    if not contains(node, x, y): return None
    if not node.visible or node.opacity <= 0.0: return None

    # 从上层（zIndex 大）的子节点开始找
    for c in reversed(node.children):
        h = hit_test(c, x, y)
        if h is not None: return h

    # 自身逻辑判断
    if node.hit == HIT_NONE or node.hit == HIT_TRANSPARENT:
        return None
    return node

def find_clickable_target(node: FastNode):
    """
    向上回溯：如果当前节点不可点击，则寻找最近的一个可点击/长按的父容器。
    这是解决“点击图标还是点击按钮文字”问题的核心。
    """
    curr = node
    while curr is not None:
        if curr.clickable or curr.long_clickable:
            return curr
        curr = curr.parent
    return node # 如果往上都没找到，返回叶子节点本身

def compute_screen_size(node: FastNode, mx_my=(0, 0)):
    max_x, max_y = mx_my
    if node.has_bounds:
        max_x = max(max_x, node.x2)
        max_y = max(max_y, node.y2)
    for c in node.children:
        max_x, max_y = compute_screen_size(c, (max_x, max_y))
    return max_x, max_y

def denorm_to_pixel(root: FastNode, nx: int, ny: int):
    max_x, max_y = compute_screen_size(root)
    # 适配不同分辨率 fallback
    if max_x <= 0 or max_y <= 0:
        max_x, max_y = 1276, 2848 

    def to_pix(n, maxv):
        n = max(1, min(n, 1000))
        t = (n - 1) / 999.0
        return int(round(t * (maxv - 1)))

    return to_pix(nx, max_x), to_pix(ny, max_y)

def same_ui_hit(ui_json, nx1: int, ny1: int, nx2: int, ny2: int) -> int:
    """
    判断归一化坐标 (nx1, ny1) 和 (nx2, ny2) 是否命中同一个逻辑 UI 组件。
    """
    try:
        if isinstance(ui_json, str):
            ui_json = json.loads(ui_json)
        if not isinstance(ui_json, dict):
            return 0

        # 1. 构建带 parent 指针的树
        root = from_json_fast(ui_json)

        # 2. 坐标转换
        x1, y1 = denorm_to_pixel(root, nx1, ny1)
        x2, y2 = denorm_to_pixel(root, nx2, ny2)

        # 3. 查找最深层叶子节点
        leaf1 = hit_test(root, x1, y1)
        leaf2 = hit_test(root, x2, y2)

        if leaf1 is None or leaf2 is None:
            print("未命中任何组件")
            return 0

        # 4. 向上追溯响应链
        target1 = find_clickable_target(leaf1)
        target2 = find_clickable_target(leaf2)

        # 5. 比较是否为同一个内存对象
        if target1 is target2:
            print(f"是 (命中组件类型: {target1.type}, ID: {target1.id})")
            return 1
        else:
            print(f"不是 (目标1: {target1.type}, 目标2: {target2.type})")
            return 0
    except Exception as e:
        print(f"HitTest 发生异常: {e}")
        return 0