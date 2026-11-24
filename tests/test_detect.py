"""切点检测（相邻帧背景灰度相关）测试。

运行：python3 tests/test_detect.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import detect  # noqa: E402


def noise(seed, w=16, h=16):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w)).astype(np.uint8)


def test_identical_frames():
    g = noise(0)
    s = detect.pair_stats([g, g.copy()], [[], []])[0]
    assert not s["review"] and abs(s["corr"] - 1.0) < 1e-9, s
    assert s["cut"] == 1 and s["bg"] == 256 and abs(s["ratio"] - 1.0) < 1e-9
    print("相同帧：相关系数 1，不核对")


def test_independent_frames():
    s = detect.pair_stats([noise(1), noise(2)], [[], []])[0]
    assert not s["review"] and abs(s["corr"]) < 0.2, s
    print("独立噪点：相关系数接近 0")


def test_boxes_excluded_from_either_frame():
    a = noise(3)
    b = a.copy()
    b[2:6, 2:6] = 255 - b[2:6, 2:6]          # 仅框内像素不同
    box = [[2, 2, 6, 6]]
    # 两帧都有该框 -> 排除后相关为 1
    s = detect.pair_stats([a, b], [box, box])[0]
    assert abs(s["corr"] - 1.0) < 1e-9, s
    # 框只在其中一帧 -> 同样排除（落在任一帧必留框里的像素都排除）
    s = detect.pair_stats([a, b], [[], box])[0]
    assert abs(s["corr"] - 1.0) < 1e-9, s
    s = detect.pair_stats([a, b], [box, []])[0]
    assert abs(s["corr"] - 1.0) < 1e-9, s
    # 不排除时相关显著下降
    s = detect.pair_stats([a, b], [[], []])[0]
    assert s["corr"] < 0.99, s
    print("必留框像素：任一帧的框都被排除")


def test_quarter_boundary():
    # 16x16 = 256 像素，1/4 = 64
    a, b = noise(4), noise(4)
    # 恰好剩 64（=1/4，不是"不到"）-> 不核对
    box = [[0, 0, 12, 16]]                    # 盖住 192，剩 64
    s = detect.pair_stats([a, b], [box, box])[0]
    assert s["bg"] == 64 and not s["review"], s
    # 剩 63（< 1/4）-> 待核对
    box2 = [[0, 0, 12, 16], [12, 0, 13, 1]]   # 盖住 193，剩 63
    s = detect.pair_stats([a, b], [box2, box2])[0]
    assert s["bg"] == 63 and s["review"] and s["corr"] is None, s
    assert "1/4" in s["reason"]
    print("四分之一边界：恰好 1/4 不核对，不足 1/4 待核对")


def test_constant_gray():
    a = np.full((16, 16), 128, dtype=np.uint8)   # 灰度恒定
    b = noise(5)
    s = detect.pair_stats([a, b], [[], []])[0]
    assert s["review"] and s["corr"] is None and "恒定" in s["reason"], s
    s = detect.pair_stats([b, a], [[], []])[0]   # 另一帧恒定同样核对
    assert s["review"] and s["corr"] is None, s
    print("灰度恒定：任一帧恒定即待核对")


def test_negative_correlation():
    a = noise(6)
    b = (255 - a.astype(np.int16)).astype(np.uint8)  # 反相 -> 强负相关
    s = detect.pair_stats([a, b], [[], []])[0]
    assert not s["review"] and s["corr"] < -0.99, s
    print("反相帧：强负相关（仍给出数值，由阈值判断）")


def test_multiple_pairs():
    gs = [noise(10), noise(10), noise(11)]
    stats = detect.pair_stats(gs, [[], [], []])
    assert len(stats) == 2
    assert stats[0]["cut"] == 1 and stats[1]["cut"] == 2
    assert abs(stats[0]["corr"] - 1.0) < 1e-9 and abs(stats[1]["corr"]) < 0.2
    print("多对相邻帧：逐对统计，cut 为后一帧下标")


if __name__ == "__main__":
    test_identical_frames()
    test_independent_frames()
    test_boxes_excluded_from_either_frame()
    test_quarter_boundary()
    test_constant_gray()
    test_negative_correlation()
    test_multiple_pairs()
    print("全部通过")
