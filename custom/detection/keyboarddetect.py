#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_ime_keyboard_module.py

A small module to detect whether an HDC UI-tree JSON contains an on-screen IME keyboard window.

Public API:
  - detect_ime_keyboard(obj_or_path) -> int   # returns 1 (found) or 0 (not found)
  - has_ime_keyboard(obj_or_path) -> bool

The function accepts either:
  - a Python dict/list parsed from JSON, OR
  - a filesystem path (str/pathlib.Path) to a JSON file.

Example:
  from detect_ime_keyboard_module import detect_ime_keyboard
  print(detect_ime_keyboard("ui_tree.json"))   # 1 or 0
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union


# Strong signal used by HarmonyOS SceneBoard keyboard panel window nodes
_RE_KEYBOARD_PANEL = re.compile(r"^keyboardpanel\d+$", re.IGNORECASE)

# Optional extra hints inside IME trees (kept broad but safe)
_RE_KEYBOARD_ID_HINT = re.compile(
    r"(keycanvaskeyboard|softkeyboard|keyboard(view)?|imekeyboard|inputmethodkeyboard)",
    re.IGNORECASE,
)


JsonLike = Union[Dict[str, Any], List[Any]]
InputType = Union[JsonLike, str, Path]


def _iter_nodes(root: Any) -> Iterator[Dict[str, Any]]:
    """Iterate dict nodes in a UI tree. Expected shape: {"attributes": {...}, "children": [...]}."""
    stack: List[Any] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            yield node
            children = node.get("children")
            if isinstance(children, list):
                stack.extend(children)
        elif isinstance(node, list):
            stack.extend(node)


def _load_if_path(obj_or_path: InputType) -> Any:
    """If input is a path, load JSON from it; otherwise return input unchanged."""
    if isinstance(obj_or_path, (str, Path)):
        p = Path(obj_or_path)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return obj_or_path


def has_ime_keyboard(obj_or_path: InputType) -> bool:
    """
    Return True if a keyboard/IME window is detected, else False.

    Detection rule (practical + robust for HDC trees):
      True if any of these is met:
        A) bundleName == "com.ohos.sceneboard" AND id matches ^keyboardPanel\\d+$
        B) bundleName contains "inputmethod" AND id hints keyboard (e.g., "KeyCanvasKeyboard" / contains "keyboard")
        C) id/key matches ^keyboardPanel\\d+$ (vendor variants)
    """
    data = _load_if_path(obj_or_path)

    for node in _iter_nodes(data):
        attrs = node.get("attributes") if isinstance(node.get("attributes"), dict) else {}
        bundle = str(attrs.get("bundleName", "") or "")
        node_id = str(attrs.get("id", "") or "")
        node_key = str(attrs.get("key", "") or "")

        bundle_l = bundle.lower()
        node_id_l = node_id.lower()
        node_key_l = node_key.lower()

        # A) SceneBoard keyboard panel window (very strong signal)
        if bundle_l == "com.ohos.sceneboard" and _RE_KEYBOARD_PANEL.match(node_id):
            return True

        # C) keyboardPanel* regardless of bundle (fallback)
        if _RE_KEYBOARD_PANEL.match(node_id) or _RE_KEYBOARD_PANEL.match(node_key):
            return True

        # B) IME bundle + keyboard-ish ids (strong)
        if ("inputmethod" in bundle_l) and (
            node_id_l == "keycanvaskeyboard"
            or "keycanvaskeyboard" in node_id_l
            or "keyboard" in node_id_l
            or _RE_KEYBOARD_ID_HINT.search(node_id_l) is not None
        ):
            return True

    return False


def detect_ime_keyboard(obj_or_path: InputType) -> int:
    """Return 1 if a keyboard/IME window is detected, otherwise 0."""
    try:
        return 1 if has_ime_keyboard(obj_or_path) else 0
    except Exception:
        # Conservative: on malformed input or unexpected structures, treat as not found.
        return 0


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Detect whether an HDC UI-tree JSON contains an IME keyboard window.")
    ap.add_argument("json_path", help="Path to UI tree JSON")
    args = ap.parse_args()

    sys.stdout.write(str(detect_ime_keyboard(args.json_path)))
    sys.stdout.flush()
