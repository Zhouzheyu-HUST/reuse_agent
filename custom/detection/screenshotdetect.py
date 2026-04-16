from PIL import Image
import numpy as np
import sys

# 目标尺寸
TARGET_SIZE = (256, 256)

# 阈值：先用 30000
THRESHOLD = 30000


def load_gray_resized(path, size=TARGET_SIZE):
    """
    读取图片 -> 转灰度 -> 缩放到指定大小
    返回 numpy 数组（int16，避免做差时溢出）
    """
    img = Image.open(path).convert("L")  # "L" = 灰度
    img = img.resize(size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    return arr


def diff_with_early_exit(gray1, gray2, threshold):
    """
    逐像素计算灰度差的绝对值之和，带早退：
    一旦累计和 > threshold，立刻返回（不是同一页面）

    返回: (is_same_page: bool, diff_sum: int)
    """
    if gray1.shape != gray2.shape:
        raise ValueError(f"shape 不一致: {gray1.shape} vs {gray2.shape}")

    # 拉平成一维，遍历更简单
    flat1 = gray1.ravel()
    flat2 = gray2.ravel()

    diff_sum = 0
    # 用 range 循环 + 手动累加，可以控制早退
    for i in range(flat1.size):
        # 注意 int(...)，避免 Python 把 numpy.int16 搞出意外行为
        d = abs(int(flat1[i]) - int(flat2[i]))
        if d >= 50:
            diff_sum += 1

        # 早退：一旦超过阈值，就已经可以判定为“不是同一页面”
        if diff_sum > threshold:
            return False, diff_sum

    # 能跑完说明总差没有超过阈值
    return True, diff_sum


def is_same_page(new_path, old_path, threshold=THRESHOLD):
    """
    对外暴露的接口：
    给新旧两张图路径，返回 (是否同一页面, 差值和)
    """
    g1 = load_gray_resized(new_path)
    g2 = load_gray_resized(old_path)

    return diff_with_early_exit(g1, g2, threshold)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python compare_gray_early_exit.py new.png old.png")
        sys.exit(1)

    new_path = sys.argv[1]
    old_path = sys.argv[2]

    same, diff_value = is_same_page(new_path, old_path, THRESHOLD)

    print(f"总像素差异值: {diff_value}")
    if same:
        print("判定：是同一页面")
    else:
        print("判定：不是同一页面")
