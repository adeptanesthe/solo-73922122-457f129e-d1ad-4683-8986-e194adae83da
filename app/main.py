"""手工课机位切换：切点确认 + 分镜头竖屏 9:16 裁切 —— FastAPI 后端。

流程：上传有序 PNG 帧 + 逐帧必留框 JSON -> 计算相邻帧背景灰度相关 ->
网页按阈值得到建议切点并确认 -> 按已确认切点拆成镜头，各自求解最优裁切轨迹
（改切点或锁定只重算受影响的镜头，其余命中缓存）-> 逐帧查看/拖动/锁定。
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import detect, solver

MAX_FRAMES = 80
MAX_SIDE = 512
MAX_CACHE = 128  # 镜头结果缓存上限，超出即清空（缓存只是加速，可安全丢弃）

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DIR", BASE_DIR.parent / "sample"))
SAMPLES = ("a", "b", "c", "d")

app = FastAPI(title="手工课切点确认与竖屏取景")

# 单会话演示：服务端只保存一份当前工程
SESSION: dict = {}
_MAT = {"v": 0}  # 素材版本号：每次载入素材递增，用于识别迟到结果


# ---------------------------------------------------------------- 工具

def _read_png(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"无法解码 PNG：{exc}") from exc
    if img.format != "PNG":
        raise HTTPException(400, "只接受 PNG 帧")
    return np.asarray(img.convert("L"), dtype=np.uint8)


def _parse_boxes(payload) -> list[list[list[float]]]:
    """接受 {"frames": [...]} 或裸数组；每帧为 [[x0,y0,x1,y1], ...]。"""
    if isinstance(payload, dict):
        payload = payload.get("frames")
    if not isinstance(payload, list):
        raise HTTPException(400, "必留框 JSON 须为数组或含 frames 数组的对象")
    frames = []
    for i, entry in enumerate(payload):
        if entry is None:
            entry = []
        if not isinstance(entry, list):
            raise HTTPException(400, f"第 {i} 帧的必留框不是数组")
        boxes = []
        for b in entry:
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                raise HTTPException(400, f"第 {i} 帧存在非法必留框：{b!r}")
            try:
                boxes.append([float(v) for v in b])
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, f"第 {i} 帧必留框坐标非数字：{b!r}") from exc
        frames.append(boxes)
    return frames


def _load_session(png_blobs: list[bytes], boxes_payload) -> dict:
    if not 1 <= len(png_blobs) <= MAX_FRAMES:
        raise HTTPException(400, f"帧数须在 1..{MAX_FRAMES} 之间，收到 {len(png_blobs)}")
    grays = [_read_png(b) for b in png_blobs]
    H, W = grays[0].shape
    for i, g in enumerate(grays):
        if g.shape != (H, W):
            raise HTTPException(400, f"第 {i} 帧尺寸 {g.shape[1]}x{g.shape[0]} 与首帧 {W}x{H} 不一致")
    if W > MAX_SIDE or H > MAX_SIDE:
        raise HTTPException(400, f"帧边长至多为 {MAX_SIDE} 像素，收到 {W}x{H}")
    if H % 16 != 0:
        raise HTTPException(400, f"帧高须为 16 的倍数，收到 {H}")
    cw = solver.crop_width(H)
    if cw > W:
        raise HTTPException(400, f"原图宽 {W} 放不下 9:16 竖屏窗口（需 ≥ {cw}）")

    frames_boxes = _parse_boxes(boxes_payload)
    if len(frames_boxes) != len(grays):
        raise HTTPException(400, f"必留框帧数 {len(frames_boxes)} 与 PNG 帧数 {len(grays)} 不一致")
    for i, boxes in enumerate(frames_boxes):
        for b in boxes:
            x0, y0, x1, y1 = b
            if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
                raise HTTPException(400, f"第 {i} 帧必留框 {b} 超出画面或非法")

    energies = [solver.window_energies(g) for g in grays]
    feasibles = [solver.feasible_mask(b, W, H) for b in frames_boxes]
    pairs = detect.pair_stats(grays, frames_boxes)
    X = W - cw + 1
    _MAT["v"] += 1
    session = {
        "n": len(grays), "w": W, "h": H, "cw": cw, "x_max": X - 1,
        "png": png_blobs, "boxes": frames_boxes,
        "energies": energies, "feasibles": feasibles,
        "pairs": pairs, "mat": _MAT["v"], "cache": {},
    }
    SESSION.clear()
    SESSION.update(session)
    return session


def _meta(session: dict) -> dict:
    feas = []
    for f in session["feasibles"]:
        idx = np.flatnonzero(f)
        feas.append([int(idx[0]), int(idx[-1])] if idx.size else None)
    return {
        "n": session["n"], "w": session["w"], "h": session["h"],
        "cw": session["cw"], "x_max": session["x_max"],
        "boxes": session["boxes"], "feasible": feas,
        "mat": session["mat"], "pairs": session["pairs"],
    }


def solve_shots(session: dict, ranges: list[tuple[int, int]],
                locks: dict[int, int]) -> list[dict]:
    """逐镜头求解裁切轨迹；范围与锁定未变的镜头直接命中缓存（不重算）。

    ranges: [(start, end), ...]，原片帧号区间，end 不含。
    locks:  {原片帧号: x}，按镜头范围折算为镜头内下标交给求解器。
    返回与 ranges 对齐的结果列表；失败时 fail_frame 为原片（全局）帧号。
    """
    cache = session["cache"]
    results = []
    for s, e in ranges:
        sub = {i - s: x for i, x in locks.items() if s <= i < e}
        key = (s, e, tuple(sorted(sub.items())))
        hit = cache.get(key)
        if hit is None:
            traj, fail, total = solver.solve(
                session["energies"][s:e], session["feasibles"][s:e], sub)
            if traj is None:
                hit = {"start": s, "end": e, "ok": False, "fail_frame": s + fail}
            else:
                hit = {"start": s, "end": e, "ok": True,
                       "trajectory": traj, "energy": total}
            if len(cache) >= MAX_CACHE:
                cache.clear()
            cache[key] = hit
        results.append(hit)
    return results


# ---------------------------------------------------------------- 接口

@app.post("/api/upload")
async def api_upload(frames: list[UploadFile] = File(...),
                     boxes: UploadFile = File(...)):
    png_blobs = [await f.read() for f in frames]
    try:
        payload = json.loads((await boxes.read()).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"必留框 JSON 解析失败：{exc}") from exc
    session = _load_session(png_blobs, payload)
    return _meta(session)


@app.post("/api/load_sample")
async def api_load_sample(name: str = "a"):
    if name not in SAMPLES:
        raise HTTPException(400, "未知样片")
    d = SAMPLE_DIR / f"sample_{name}"
    if not d.is_dir():
        raise HTTPException(404, f"样片目录不存在：{d}")
    paths = sorted(d.glob("frame_*.png"))
    boxes_path = d / "boxes.json"
    if not paths or not boxes_path.is_file():
        raise HTTPException(404, "样片文件不完整")
    png_blobs = [p.read_bytes() for p in paths]
    payload = json.loads(boxes_path.read_text(encoding="utf-8"))
    session = _load_session(png_blobs, payload)
    return _meta(session)


class ShotRange(BaseModel):
    start: int
    end: int


class SolveShotsRequest(BaseModel):
    mat: int                       # 发起请求时的素材版本
    shots: list[ShotRange]         # 需要（重）求解的镜头范围
    locks: dict[int, int] = {}     # 原片帧号 -> 锁定 x


@app.post("/api/solve_shots")
async def api_solve_shots(req: SolveShotsRequest):
    if not SESSION:
        raise HTTPException(400, "请先上传帧或载入样片")
    if req.mat != SESSION["mat"]:
        # 素材已变：迟到请求直接拒绝，结果不得覆盖当前预览
        raise HTTPException(409, "素材已更新，该请求结果已过期，请按当前素材重新求解")
    n = SESSION["n"]
    locks = {}
    for k, v in (req.locks or {}).items():
        i = int(k)
        if not 0 <= i < n:
            raise HTTPException(400, f"锁定帧号越界：{i}")
        x = int(v)
        if not 0 <= x <= SESSION["x_max"]:
            raise HTTPException(400, f"锁定位置越界：帧 {i} x={x}")
        locks[i] = x
    ranges = []
    for r in req.shots:
        if not (0 <= r.start < r.end <= n):
            raise HTTPException(400, f"镜头范围非法：[{r.start}, {r.end})")
        ranges.append((r.start, r.end))
    results = solve_shots(SESSION, ranges, locks)
    return {"mat": req.mat, "results": results}


@app.get("/api/frame/{idx}.png")
async def api_frame(idx: int):
    if not SESSION:
        raise HTTPException(400, "请先上传帧或载入样片")
    if not 0 <= idx < SESSION["n"]:
        raise HTTPException(404, "帧号越界")
    return Response(content=SESSION["png"][idx], media_type="image/png")


@app.get("/api/health")
async def api_health():
    return {"ok": True, "loaded": bool(SESSION)}


# 静态页面
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
