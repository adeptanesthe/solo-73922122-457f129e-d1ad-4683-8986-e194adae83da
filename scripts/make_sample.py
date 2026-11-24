"""生成演示样片：sample_a（可解）与 sample_b（不可解演示）。

样片：24 帧 512x288（高为 16 的倍数，裁切窗 162x288）。
画面：渐变底 + 静态噪点纹理 + 棋盘格高纹理区 + 移动的"工件"亮块。
必留框逐帧跟踪亮块。样片 B 在第 12 帧让工件瞬移 40px，使轨迹接不上。

样片 C / D 演示机位切换：第 12 帧起换机位（背景整体不同，相邻帧背景
相关系数骤降 -> 建议切点），工件轨迹也换到新机位取景。C 两个镜头各自
可解；D 在第二个镜头内第 18 帧再让工件瞬移，使镜头 2 不可解（演示按
原片帧号报告首个接不上的帧）。

用法：python3 scripts/make_sample.py [输出目录，默认 sample/]
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 512, 288
N = 24
OBJ = 44          # 工件边长
X0, V = 20, 7     # 工件起始 x 与每帧速度
OY = 120          # 工件 y


def base_frame(rng):
    yy, xx = np.mgrid[0:H, 0:W]
    g = 60 + 40 * np.sin(xx / 90.0) + 30 * np.cos(yy / 70.0)
    g += rng.normal(0, 18, (H, W))            # 静态噪点（同一随机场）
    # 棋盘格高纹理条带
    chk = ((xx // 6 + yy // 6) % 2) * 90
    band = (xx >= 300) & (xx < 380)
    g = np.where(band, 40 + chk, g)
    return np.clip(g, 0, 255)


def make(out: Path, teleport_at=None):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    bg = base_frame(rng)
    boxes_all = []
    for i in range(N):
        ox = X0 + V * i
        if teleport_at is not None and i >= teleport_at:
            ox += 160                           # 瞬移 -> 相邻帧窗口可行区间错开 >8px，必接不上
        oy = OY + int(6 * np.sin(i / 3.0))
        frame = bg.copy()
        # 暗色圆（干扰物，缓慢左移）
        cx = 430 - 3 * i
        yy, xx = np.mgrid[0:H, 0:W]
        frame[(xx - cx) ** 2 + (yy - 200) ** 2 < 26 ** 2] = 30
        img = Image.fromarray(frame.astype(np.uint8), "L").convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([ox, oy, ox + OBJ - 1, oy + OBJ - 1], fill=(235, 220, 180))
        d.line([ox, oy + OBJ // 2, ox + OBJ - 1, oy + OBJ // 2], fill=(120, 60, 40), width=3)
        img.save(out / f"frame_{i:03d}.png")
        boxes_all.append([[ox, oy, ox + OBJ, oy + OBJ]])
    (out / "boxes.json").write_text(
        json.dumps({"frames": boxes_all}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out}: {N} 帧 {W}x{H}")


def base_frame2(rng):
    """机位 2 的背景：不同的渐变、噪点场与棋盘条带位置。"""
    yy, xx = np.mgrid[0:H, 0:W]
    g = 50 + 35 * np.sin(xx / 110.0 + 1.3) + 25 * np.cos(yy / 55.0 + 0.4)
    g += rng.normal(0, 18, (H, W))            # 另一组静态噪点
    chk = ((xx // 6 + yy // 6) % 2) * 90
    band = (xx >= 60) & (xx < 140)
    g = np.where(band, 40 + chk, g)
    return np.clip(g, 0, 255)


def draw_frame(frame, ox, oy, i):
    """在背景上画干扰圆与工件，保存 PNG 并返回该帧必留框。"""
    cx = 430 - 3 * i
    yy, xx = np.mgrid[0:H, 0:W]
    frame[(xx - cx) ** 2 + (yy - 200) ** 2 < 26 ** 2] = 30
    img = Image.fromarray(frame.astype(np.uint8), "L").convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([ox, oy, ox + OBJ - 1, oy + OBJ - 1], fill=(235, 220, 180))
    d.line([ox, oy + OBJ // 2, ox + OBJ - 1, oy + OBJ // 2], fill=(120, 60, 40), width=3)
    return img, [[ox, oy, ox + OBJ, oy + OBJ]]


def make_cut(out: Path, cut_at, cam2_x0, cam2_v, cam2_oy, teleport_at=None):
    """双机位样片：第 cut_at 帧起换机位（背景与工件取景都变）。

    teleport_at: 在机位 2 内再让工件瞬移 160px（演示镜头内不可解）。
    """
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    bg1 = base_frame(rng)
    bg2 = base_frame2(rng)
    boxes_all = []
    for i in range(N):
        if i < cut_at:
            ox = X0 + V * i
            oy = OY + int(6 * np.sin(i / 3.0))
            frame = bg1.copy()
        else:
            ox = cam2_x0 + cam2_v * (i - cut_at)
            if teleport_at is not None and i >= teleport_at:
                ox += 160                       # 镜头内瞬移 -> 该镜头接不上
            oy = cam2_oy + int(6 * np.sin(i / 3.0))
            frame = bg2.copy()
        img, boxes = draw_frame(frame, ox, oy, i)
        img.save(out / f"frame_{i:03d}.png")
        boxes_all.append(boxes)
    (out / "boxes.json").write_text(
        json.dumps({"frames": boxes_all}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out}: {N} 帧 {W}x{H}，第 {cut_at} 帧切换机位")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample")
    make(root / "sample_a")
    make(root / "sample_b", teleport_at=12)
    make_cut(root / "sample_c", cut_at=12, cam2_x0=320, cam2_v=6, cam2_oy=100)
    make_cut(root / "sample_d", cut_at=12, cam2_x0=240, cam2_v=6, cam2_oy=100,
             teleport_at=18)


if __name__ == "__main__":
    main()
