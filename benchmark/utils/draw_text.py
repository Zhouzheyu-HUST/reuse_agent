# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import math
import os
import re
import shutil
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from utils import read_json


Point = Tuple[int, int]
Gesture = Tuple[Point, Point]

FRAME_PATTERN = re.compile(r"frame_(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)
SCROLL_PATTERN = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

RED_DOT_COLOR = (255, 0, 0, 200)
SCROLL_COLOR = (0, 122, 255, 210)
SCROLL_WIDTH = 12
STRIP_BG_COLOR = "white"
STRIP_ACCENT_COLOR = "#0062ff"

ACTION_LABELS = {
    "open_app": "Open App",
    "click": "Tap",
    "longclick": "Long Press",
    "scroll": "Scroll",
    "input_text": "Input Text",
    "back": "Back",
    "home": "Home",
    "retry": "Retry",
}


@dataclass
class ActionDescription(object):
    step: int
    name: str
    label: str
    detail: str
    tap_point: Optional[Point] = None
    gesture: Optional[Gesture] = None


def parse_point(value: object) -> Optional[Point]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            x = int(value[0])
            y = int(value[1])
            return x, y
        except (TypeError, ValueError):
            return None
    return None


def parse_scroll_points(value: object) -> Optional[Gesture]:
    if isinstance(value, str):
        match = SCROLL_PATTERN.fullmatch(value.strip())
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            return (x1, y1), (x2, y2)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            x1, y1, x2, y2 = map(int, value)
            return (x1, y1), (x2, y2)
        except (TypeError, ValueError):
            return None
    return None


def format_point(point: Optional[Point]) -> str:
    if point is None:
        return "(unknown)"
    return f"({point[0]}, {point[1]})"


def build_action_description(entry: dict) -> ActionDescription:
    action = entry.get("action") or "unknown_action"
    label = ACTION_LABELS.get(action, action.replace("_", " ").title())
    payload = (entry.get("original_item") or {}).get("text")
    step = entry.get("step_number", -1)
    detail = ""
    tap_point: Optional[Point] = None
    gesture: Optional[Gesture] = None

    if action in {"click", "longclick"}:
        tap_point = parse_point(payload)
        verb = "Tap" if action == "click" else "Long press"
        detail = f"{verb} at {format_point(tap_point)}"
    elif action == "open_app":
        detail = f"Open app: {payload or 'Unknown'}"
    elif action == "scroll":
        gesture = parse_scroll_points(payload)
        if gesture:
            start, end = gesture
            detail = f"Scroll from {format_point(start)} to {format_point(end)}"
        else:
            detail = "Scroll on screen"
    elif action == "input_text":
        detail = f"Input text: {payload}"
    elif action == "back":
        detail = "Go back to previous screen"
    elif action == "home":
        detail = "Return to the home screen"
    elif action == "retry":
        detail = "Retry this reasoning step"
    else:
        detail = f"Action payload: {payload}"

    return ActionDescription(
        step=step,
        name=action,
        label=label,
        detail=detail,
        tap_point=tap_point,
        gesture=gesture
    )


def load_trace(trace_path: str) -> List[ActionDescription]:
    entries = read_json(trace_path)
    descriptions = [build_action_description(entry) for entry in entries]
    return descriptions


def extract_frame_index(filename: str) -> Optional[int]:
    match = FRAME_PATTERN.match(filename)
    if match:
        return int(match.group(1))
    return None


def collect_frame_files(image_dir: str) -> List[Tuple[int, str]]:
    frames: List[Tuple[int, str]] = []
    for name in os.listdir(image_dir):
        frame_index = extract_frame_index(name)
        if frame_index is not None:
            frames.append((frame_index, name))
    return sorted(frames, key=lambda item: item[0])


def ensure_fresh_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def draw_tap_overlay(
    image: Image.Image, 
    coordinates: Point, 
    radius: int
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x, y = coordinates
    left_up = (x - radius, y - radius)
    right_down = (x + radius, y + radius)
    draw.ellipse([left_up, right_down], fill=RED_DOT_COLOR)
    return Image.alpha_composite(base, overlay)


def draw_scroll_overlay(
    image: Image.Image, 
    gesture: Gesture
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    start, end = gesture
    draw.line([start, end], fill=SCROLL_COLOR, width=SCROLL_WIDTH)
    draw_arrow_head(draw, start, end, SCROLL_COLOR)
    draw.ellipse(
        [
            (start[0] - 15, start[1] - 15),
            (start[0] + 15, start[1] + 15)
        ],
        outline=SCROLL_COLOR,
        width=6
    )
    return Image.alpha_composite(base, overlay)


def draw_arrow_head(
    draw: ImageDraw.ImageDraw, 
    start: Point, 
    end: Point, 
    color: Tuple[int, int, int, int], 
    size: int = 40
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    left = (end[0] - ux * size - uy * size / 2, end[1] - uy * size + ux * size / 2)
    right = (end[0] - ux * size + uy * size / 2, end[1] - uy * size - ux * size / 2)
    draw.polygon([end, left, right], fill=color)


def draw_action_overlay(
    image_path: str, 
    action: ActionDescription, 
    dot_radius: int
) -> None:    
    image = Image.open(image_path)
    modified = False

    if action.tap_point:
        image = draw_tap_overlay(image, action.tap_point, dot_radius)
        modified = True

    if action.gesture:
        image = draw_scroll_overlay(image, action.gesture)
        modified = True

    if modified:
        rgb_image = image.convert("RGB")
        rgb_image.save(image_path)


def load_font(font_size: int) -> ImageFont.ImageFont:
    font_path = "benchmark/font/NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(font_path, font_size)


def calculate_characters_per_line(
    image_width: int, 
    font: ImageFont.ImageFont
) -> int:
    sample_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    total_width = 0
    for char in sample_text:
        bbox = font.getbbox(char)
        total_width += bbox[2] - bbox[0]
    average_char_width = max(1, total_width // len(sample_text))
    return max(1, image_width // average_char_width)


def add_description_strip(
    image_path: str,
    action: ActionDescription,
    output_path: str,
    font_size: int,
    line_spacing: int
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    font = load_font(font_size)
    characters_per_line = calculate_characters_per_line(width, font)

    segments = [
        {"text": f"Action: {action.label}", "color": "red"},
        {"text": f"Detail: {action.detail}", "color": "black"}
    ]

    wrapped_lines: List[Tuple[str, str]] = []
    wrapper = textwrap.TextWrapper(width=characters_per_line, break_long_words=True, break_on_hyphens=False)
    for segment in segments:
        lines = segment["text"].splitlines() or [segment["text"]]
        for line in lines:
            wrapped = wrapper.wrap(line) or [line]
            for wrapped_line in wrapped:
                wrapped_lines.append((wrapped_line, segment["color"]))

    text_height = 0
    for line, _ in wrapped_lines:
        bbox = font.getbbox(line)
        text_height += (bbox[3] - bbox[1]) + line_spacing

    strip_height = text_height + 20
    accent_height = 10
    new_image = Image.new("RGB", (width, height + accent_height + strip_height), STRIP_BG_COLOR)
    new_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_image)
    draw.rectangle([(0, height), (width, height + accent_height)], fill=STRIP_ACCENT_COLOR)

    y = height + accent_height + 10
    for line, color in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y), line, font=font, fill=color)
        y += (bbox[3] - bbox[1]) + line_spacing

    new_image.save(output_path)


def add_action_2_screenshot(
    folder_path: str,
    include_text: bool = True,
    font_size: int = 60,
    line_spacing: int = 10,
    dot_radius: int = 20
) -> None:
    trace_path = os.path.join(folder_path, "trace.json")
    image_dir = os.path.join(folder_path, "ImageInfo")

    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    descriptions = load_trace(trace_path)
    action_map: Dict[int, ActionDescription] = {desc.step: desc for desc in descriptions if desc.step >= 0}
    frames = collect_frame_files(image_dir)
    if not frames:
        raise RuntimeError(f"No frames found in {image_dir}")
    
    if not include_text:
        dst_dir = os.path.join(folder_path, "tap_only")
    else:
        dst_dir = os.path.join(folder_path, "tap_and_text")
    ensure_fresh_dir(dst_dir)

    for frame_index, filename in frames:
        src = os.path.join(image_dir, filename)
        dst = os.path.join(dst_dir, filename)
        shutil.copy(src, dst)
        action = action_map.get(frame_index)
        if action:
            draw_action_overlay(dst, action, dot_radius)
            if include_text:
                add_description_strip(dst, action, dst, font_size, line_spacing)
