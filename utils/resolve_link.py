# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


import os
from pathlib import Path
import argparse


def _resolve_link_target(link_path: Path) -> Path:
    """
    Resolve symlink target to an absolute real path.
    """
    # readlink returns the raw target path (may be relative)
    raw = os.readlink(str(link_path))
    t = Path(raw)
    if not t.is_absolute():
        t = (link_path.parent / t)
    # resolve() will normalize .. and return absolute path
    return t.resolve()


def mirror_hardlinks(src: str, 
                     dst: str) -> None:
    src_p = Path(src)
    dst_p = Path(dst)

    if not src_p.exists():
        raise FileNotFoundError(f"Src not found: {src_p}")

    # Walk all files under src
    for p in src_p.rglob("*"):
        if not p.is_file():
            continue

        rel = p.relative_to(src_p)
        out = dst_p / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        # Determine real target (follow symlink if needed)
        if p.is_symlink():
            target = _resolve_link_target(p)
        else:
            target = p.resolve()

        # Hardlink requires same drive/volume on Windows
        if target.drive.lower() != out.drive.lower():
            raise RuntimeError(f"Hardlink requires same drive. target={target} dst={out}")

        # If target already exists, remove it (file or dangling link)
        if out.exists() or out.is_symlink():
            out.unlink()

        # Create hardlink
        os.link(str(target), str(out))


def _main(args: dict) -> None:
    if args["src_repo"]:
        mirror_hardlinks(args["src_repo"], args["dst"])
    if args["src_weights"]:
        mirror_hardlinks(args["src_weights"], args["dst"])

    print(f'Done. Hardlink folder created at: {args["dst"]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transform relink")
    parser.add_argument(
        "--src_repo", 
        default=r"", 
        type=str
    )
    parser.add_argument(
        "--src_weights", 
        default="", 
        type=str
    )
    parser.add_argument(
        "--dst", 
        default="", 
        type=str
    )
    args = vars(parser.parse_args())

    _main(args)
