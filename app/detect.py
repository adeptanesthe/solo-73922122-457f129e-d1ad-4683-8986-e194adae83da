"""相邻原帧背景灰度相关 —— 机位切点检测。

对每对相邻帧 (i, i+1)：
- 有效背景 = 不落在任一帧必留框内的像素；
- 有效背景不足全帧 1/4，或任一帧在这些位置灰度恒定 -> 待核对（不给相关系数）；
- 否则给出皮尔逊相关系数，网页按所设阈值筛出建议切点。
"""

from __future__ import annotations

import numpy as np

MIN_BG_RATIO = 0.25  # 有效背景不足全帧四分之一 -> 待核对


def background_mask(boxes, width: int, height: int) -> np.ndarray:
    """不在任何必留框内的像素为 True。框为 [x0, x1) x [y0, y1)。"""
    mask = np.ones((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        xa, xb = int(np.floor(x0)), int(np.ceil(x1))
        ya, yb = int(np.floor(y0)), int(np.ceil(y1))
        mask[max(0, ya):min(height, yb), max(0, xa):min(width, xb)] = False
    return mask


def pair_stats(grays: list[np.ndarray], frames_boxes: list) -> list[dict]:
    """计算每对相邻帧的背景相关统计。

    grays:        长度 n 的 (H, W) 灰度数组
    frames_boxes: 每帧必留框列表

    返回长度 n-1 的列表，第 i 项描述相邻帧 i 与 i+1：
    - cut:    若在此切开，新镜头从第 cut 帧（0 基）开始，即 cut = i + 1
    - bg:     有效背景像素数
    - ratio:  有效背景占全帧比例
    - corr:   背景灰度皮尔逊相关系数；待核对时为 None
    - review: 是否待核对
    - reason: 待核对原因（非待核对为 None）
    """
    h, w = grays[0].shape
    total = h * w
    masks = [background_mask(b, w, h) for b in frames_boxes]
    stats = []
    for i in range(len(grays) - 1):
        m = masks[i] & masks[i + 1]
        count = int(m.sum())
        entry = {
            "cut": i + 1,
            "bg": count,
            "ratio": count / total,
            "corr": None,
            "review": False,
            "reason": None,
        }
        if count * 4 < total:  # 剩下不到全帧四分之一
            entry["review"] = True
            entry["reason"] = "有效背景不足全帧 1/4"
        else:
            a = grays[i][m].astype(np.float64)
            b = grays[i + 1][m].astype(np.float64)
            sa, sb = a.std(), b.std()
            if sa == 0.0 or sb == 0.0:  # 任一帧在这些位置灰度恒定
                entry["review"] = True
                entry["reason"] = "任一帧背景灰度恒定"
            else:
                entry["corr"] = float(
                    ((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
        stats.append(entry)
    return stats
