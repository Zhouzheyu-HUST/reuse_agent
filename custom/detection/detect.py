#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化脚本（无判稳版，截图判挡 + 模糊哈希）：
- 每一步只 dump 一次 UI 树 json
- 利用“感知哈希 + 汉明距离阈值”做 BlockingMonitor，检查 Agent 运行期间屏幕是否变化

变化点：
  之前用的是 MD5 完全一致判断：
    baseline_hash == new_hash 才认为“没变”
  现在改为感知哈希（aHash）+ 汉明距离：
    hamming_distance(baseline_hash, new_hash) <= MAX_HASH_DISTANCE 认为“基本没变”
"""

import subprocess
import time
import json
import threading
import os
from typing import List, Tuple, Dict, Any, Optional

from PIL import Image  # 需要：pip install pillow

# ====================== 全局配置 ======================

# hdc 可执行文件名字或路径（如果不在 PATH，可以改绝对路径）
HDC = "hdc"

# Agent 一轮推理的模拟时长（秒）
AGENT_DURATION = 3.0

# BlockingMonitor 截图间隔（秒），越小越灵敏，越大越省资源
MONITOR_INTERVAL = 0.3

# 截图/哈希失败后的重试等待（秒）
MONITOR_RETRY_DELAY = 0.3

# 感知哈希允许的最大差异（bit 位数）
# 0 表示完全一样，值越大表示允许的差异越大
MAX_HASH_DISTANCE = 20


# =====================================================================
#                           hdc 工具函数
# =====================================================================

def run_hdc(args: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    """
    调用 hdc 命令。

    示例：
        run_hdc(["shell", "input", "tap", "500", "1200"])
        run_hdc(["shell", "uitest", "dumpLayout"], capture_output=True)
    """
    cmd = [HDC] + args
    if capture_output:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        return subprocess.run(cmd, text=True)


# =====================================================================
#                      dumpLayout & 截图 操作
# =====================================================================

def dump_layout_json() -> Optional[Dict[str, Any]]:
    """
    使用 hdc 从手机拉取当前 UI 树（JSON）。

    返回：
        dict  - 解析成功的 json
        None  - 失败（命令失败 / JSON 解析失败 / 空输出）
    """
    proc = run_hdc(["shell", "uitest", "dumpLayout"], capture_output=True)
    if proc.returncode != 0:
        print("[PC] dumpLayout failed:", proc.stderr.strip())
        return None

    text = proc.stdout.strip()
    if not text:
        return None

    try:
        data: Dict[str, Any] = json.loads(text)
        return data
    except json.JSONDecodeError:
        print("[PC] JSON decode error, raw head:", text[:200])
        return None


def save_json(data: Dict[str, Any], filename: str) -> None:
    """将 UI 树 JSON 内容保存为本地文件。"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[PC] saved json → {filename}")


def capture_screenshot(step_id: int, suffix: str = "") -> str:
    """
    使用 snapshot_display 截图，并保存到本地。

    参数：
        step_id : 当前步骤编号，用于文件命名
        suffix  : 可选后缀，用于区分 Agent 重试（如 "_agent2"）

    返回：
        本地图片文件名
    """
    base = f"page_{step_id}{suffix}"
    remote_path = f"/data/local/tmp/{base}.png"
    local_path = f"{base}.png"

    print(f"[PC] capturing screenshot → {local_path}")

    # 注意：不同设备 snapshot_display 语法可能不同，需要按实际情况调整
    run_hdc(["shell", f"snapshot_display -f {remote_path}"])
    run_hdc(["file", "recv", remote_path, local_path])
    run_hdc(["shell", "rm", "-f", remote_path])

    return local_path


# =====================================================================
#                  感知哈希（aHash）+ 汉明距离
# =====================================================================

def compute_perceptual_hash(path: str, size: int = 16) -> Optional[int]:
    """
    计算图片的感知哈希（aHash），返回一个整数（size*size bit）。

    步骤：
      1. 打开图片，转为灰度
      2. 缩放为 size x size（默认 16x16，共 256 个像素）
      3. 计算所有像素的平均灰度值
      4. 对每个像素：
           灰度 > 均值 → bit = 1
           否则        → bit = 0
      5. 最终得到 size*size 个 bit，拼成一个整数（位 0 是最低位）

    返回：
      int      - 感知哈希值
      None     - 失败（文件不存在或无法打开）
    """
    if not os.path.exists(path):
        print(f"[PC] image not found for pHash: {path}")
        return None

    try:
        with Image.open(path) as img:
            img = img.convert("L")        # 转灰度
            img = img.resize((size, size), Image.BILINEAR)
            pixels = list(img.getdata())
    except OSError as e:
        print(f"[PC] failed to open image for pHash: {e}")
        return None

    if not pixels:
        return None

    avg = sum(pixels) / len(pixels)

    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= (1 << i)

    return bits


def hamming_distance(a: int, b: int) -> int:
    """
    计算两个整数（位串）的汉明距离（bit 不同的个数）。
    """
    x = a ^ b
    dist = 0
    while x:
        x &= x - 1
        dist += 1
    return dist


# =====================================================================
#                     BlockingMonitor：截图判挡
# =====================================================================

class BlockingMonitor(threading.Thread):
    """
    BlockingMonitor：在 Agent 阶段运行，用“感知哈希 + 汉明距离”监控屏幕是否变化。

    - baseline_hash: 基线截图的感知哈希（整数）。
    - 周期性抓新截图（覆盖 monitor_step{step_id}.png），计算新哈希 new_hash：
        * 若 hamming_distance(baseline_hash, new_hash) > MAX_HASH_DISTANCE：
            认为“屏幕发生明显变化”，调用 change_event.set()
    - 主线程在重新截新图后，可以调用 update_baseline(new_hash)，更新 baseline。
    """

    def __init__(self,
                 step_id: int,
                 baseline_hash: int,
                 change_event: "threading.Event",
                 stop_event: "threading.Event"):
        super().__init__(daemon=True)
        self.step_id = step_id
        self._lock = threading.Lock()
        self._baseline_hash = baseline_hash
        self.change_event = change_event
        self.stop_event = stop_event

    def update_baseline(self, new_hash: int) -> None:
        """主线程在重截新图后调用，更新基线哈希。"""
        with self._lock:
            self._baseline_hash = new_hash

    def _get_baseline(self) -> int:
        with self._lock:
            return self._baseline_hash

    def _capture_monitor_screenshot_hash(self) -> Optional[int]:
        """
        专供 BlockingMonitor 使用的截图函数：

        - 远端文件固定为 /data/local/tmp/monitor_step{step_id}.png
        - 本地文件固定为 monitor_step{step_id}.png（每次覆盖写）

        返回：
            感知哈希（int）或 None（失败）
        """
        remote_path = f"/data/local/tmp/monitor_step{self.step_id}.png"
        local_path = f"monitor_step{self.step_id}.png"

        # 抓新截图
        run_hdc(["shell", f"snapshot_display -f {remote_path}"])
        run_hdc(["file", "recv", remote_path, local_path])
        run_hdc(["shell", "rm", "-f", remote_path])

        # 计算感知哈希
        return compute_perceptual_hash(local_path)

    def run(self) -> None:
        print(f"[PC] BlockingMonitor started for step {self.step_id}")
        while not self.stop_event.is_set():
            if self.change_event.is_set():
                # 已经检测到变化了，等主线程处理，避免空转
                time.sleep(MONITOR_RETRY_DELAY)
                continue

            new_hash = self._capture_monitor_screenshot_hash()
            if new_hash is None:
                time.sleep(MONITOR_RETRY_DELAY)
                continue

            baseline = self._get_baseline()
            dist = hamming_distance(baseline, new_hash)

            if dist > MAX_HASH_DISTANCE:
                print(f"[PC] WARNING: screen changed for step {self.step_id}, "
                      f"hamming_distance={dist} > {MAX_HASH_DISTANCE}")
                print("[PC]          Agent may need a new screenshot.")
                self.change_event.set()
                time.sleep(MONITOR_RETRY_DELAY)
                continue

            # 屏幕仍与 baseline 足够相似，下一轮再检查
            time.sleep(MONITOR_INTERVAL)

        print(f"[PC] BlockingMonitor stopped for step {self.step_id}")


# =====================================================================
#                       初次比对的占位函数
# =====================================================================

def mock_compare(json_path: str, image_path: str) -> bool:
    """
    初次比对的占位函数。

    实际中：
      - 这里应该调用 C++ 动态库，对 json + 截图进行比对，
        包括判挡、关键区域是否被盖住等等。

    返回值语义：
      True  - 页面与“期望模板”一致，可以直接执行下一步动作
      False - 页面与“期望模板”不一致，需要进入 Agent 阶段

    目前实现：
      - sleep(1) 模拟比对耗时
      - 固定返回 False，测试 Agent + BlockingMonitor 的逻辑
    """
    print(f"[PC] mock_compare(json='{json_path}', image='{image_path}')")
    time.sleep(1.0)
    return False  # 调试阶段统一当作“不一致”


# =====================================================================
#                   动作发送 & 单步流程（核心逻辑）
# =====================================================================

def send_tap(x: int, y: int) -> float:
    """
    发送一次点击动作：hdc shell input tap x y

    返回：
        动作发送完成时的时间戳（time.time）
    """
    print(f"[PC] tap ({x}, {y})")
    run_hdc(["shell", "input", "tap", str(x), str(y)])
    return time.time()


def run_step(step_id: int,
             action: Tuple[str, int, int]) -> bool:
    """
    执行单步动作（step）的完整流程。

    参数：
        step_id : 当前是第几步
        action  : ("tap", x, y)

    返回：
        True  - 本步成功，可以继续下一步
        False - 本步失败，主流程应中断
    """
    kind, x, y = action
    print(f"\n===== STEP {step_id}: {kind}({x}, {y}) =====")

    if kind != "tap":
        print(f"[PC] unsupported action kind: {kind}")
        return False

    # 1. 发送点击动作
    _ = send_tap(x, y)  # 目前只是发送，不用时间戳

    # 2. 直接 dump 一次 UI 树（无判稳）
    layout = dump_layout_json()
    if layout is None:
        print(f"[PC] step {step_id}: failed to dump layout, aborting.")
        return False

    json_name = f"page_{step_id}.json"
    save_json(layout, json_name)

    # 3. 截一次图，作为 baseline screenshot
    png_name = capture_screenshot(step_id)

    # 4. 初次比对（占位）
    is_match = mock_compare(json_name, png_name)
    if is_match:
        # 一致：本步结束，直接下一步
        print(
            f"[PC] step {step_id}: page matches expectation, proceed to next step.")
        return True

    print(
        f"[PC] step {step_id}: page does NOT match expectation, entering Agent + BlockingMonitor phase.")

    # 5. Agent + BlockingMonitor 阶段
    baseline_hash = compute_perceptual_hash(png_name)
    if baseline_hash is None:
        print(
            f"[PC] step {step_id}: failed to compute baseline pHash, aborting.")
        return False

    change_event = threading.Event()
    stop_event = threading.Event()
    monitor = BlockingMonitor(step_id, baseline_hash, change_event, stop_event)
    monitor.start()

    agent_round = 0

    try:
        while True:
            agent_round += 1
            change_event.clear()

            # 5.1 模拟 Agent 推理过程
            print(
                f"[PC] step {step_id}: Agent round {agent_round} running (sleep {AGENT_DURATION}s) ...")
            time.sleep(AGENT_DURATION)

            # 5.2 检查期间 BlockingMonitor 是否检测到屏幕变化
            if change_event.is_set():
                print(
                    f"[PC] step {step_id}: Agent round {agent_round} aborted - screen changed during decision.")
                print(
                    "[PC]          Re-capturing baseline json + screenshot and restarting Agent.")

                # 重新 dump 一份 UI 树作为新的 baseline json
                new_layout = dump_layout_json()
                if new_layout is None:
                    print(
                        f"[PC] step {step_id}: failed to dump layout after change, aborting step.")
                    return False

                # 新的 json 文件（加后缀方便调试；生产中可按需覆盖）
                json_name = f"page_{step_id}_agent{agent_round}.json"
                save_json(new_layout, json_name)

                # 重新截屏，作为新的 baseline screenshot
                png_name = capture_screenshot(
                    step_id, suffix=f"_agent{agent_round}")
                new_hash = compute_perceptual_hash(png_name)
                if new_hash is None:
                    print(
                        f"[PC] step {step_id}: failed to compute new baseline pHash, aborting.")
                    return False

                # 更新 BlockingMonitor 的 baseline
                monitor.update_baseline(new_hash)

                # 然后继续下一轮 Agent（再 sleep 3 秒）
                continue

            # 屏幕在整轮 Agent 过程中保持与 baseline 截图足够相似
            print(
                f"[PC] step {step_id}: Agent round {agent_round} finished with stable screen.")
            # 这里本来应该根据 Agent 的输出决定下一步动作；
            # 当前只是结构骨架，认为本步成功，结束 Agent 阶段。
            break

    finally:
        # 5.3 停止 BlockingMonitor
        stop_event.set()
        monitor.join(timeout=2.0)

    print(f"[PC] step {step_id}: finished successfully after Agent phase.")
    return True


# =====================================================================
#                             主流程 main
# =====================================================================

def main() -> None:
    """
    主流程：
      - 定义一串动作脚本（steps）
      - 逐步执行 run_step
      - 一旦某一步返回 False，则中断后续执行
    """

    steps: List[Tuple[str, int, int]] = [
        ("tap", 500, 1600),
        ("tap", 200, 300),
        ("tap", 800, 300),
    ]

    for idx, act in enumerate(steps, start=1):
        ok = run_step(idx, act)
        if not ok:
            print(f"[PC] stopping sequence due to failure at step {idx}")
            break

    print("[PC] script finished.")


if __name__ == "__main__":
    main()
