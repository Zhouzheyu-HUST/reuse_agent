from PIL import Image
import sys
import os

TARGET_SIZE = (256, 256)

def resize_to_256(src_path, dst_path=None):
    # 打开图片
    img = Image.open(src_path)

    # 直接缩放到 256x256（会拉伸/压扁，不保持比例）
    img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)

    # 目标路径
    if dst_path is None:
        root, ext = os.path.splitext(src_path)
        dst_path = f"{root}_256x256{ext}"

    # 保存
    img_resized.save(dst_path)
    print(f"已保存到: {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python resize_256.py your_screenshot.png [输出路径可选]")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) >= 3 else None

    resize_to_256(src, dst)
