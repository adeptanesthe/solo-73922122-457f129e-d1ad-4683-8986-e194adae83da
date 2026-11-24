"""接口级测试：分镜头求解、增量重算（只重算受影响镜头）、迟到结果拒绝。

运行：python3 tests/test_api.py
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app.main as m  # noqa: E402

client = TestClient(m.app)

H = W = 64          # 高 64 -> 裁切窗宽 36，x ∈ [0, 28]


def png_bytes(gray):
    buf = io.BytesIO()
    Image.fromarray(gray, "L").save(buf, "PNG")
    return buf.getvalue()


def upload(obj_xs):
    """造一组 64x64 帧：噪点底 + 8x8 工件块，必留框跟踪工件。"""
    rng = np.random.RandomState(0)
    frames = [rng.randint(0, 256, (H, W)).astype(np.uint8) for _ in obj_xs]
    boxes = []
    files = []
    for i, (g, ox) in enumerate(zip(frames, obj_xs)):
        g = g.copy()
        g[20:28, ox:ox + 8] = 240
        boxes.append([[ox, 20, ox + 8, 28]])
        files.append(("frames", (f"f{i:03d}.png", png_bytes(g), "image/png")))
    files.append(("boxes", ("boxes.json", json.dumps({"frames": boxes}),
                             "application/json")))
    r = client.post("/api/upload", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def solve_shots(mat, shots, locks=None):
    body = {"mat": mat, "shots": [{"start": s, "end": e} for s, e in shots],
            "locks": locks or {}}
    return client.post("/api/solve_shots", json=body)


# 工件在帧 3->4 之间从 x=8 跳到 x=50：整段不可解（第 4 帧接不上），
# 切成 [0,4) 与 [4,6) 后各自可解。
OBJ_XS = [2, 4, 6, 8, 50, 50]


def test_split_and_global_fail_frame():
    meta = upload(OBJ_XS)
    assert meta["n"] == 6 and len(meta["pairs"]) == 5 and meta["mat"] >= 1
    # 整段：失败帧为原片第 4 帧（全局帧号）
    r = solve_shots(meta["mat"], [(0, 6)])
    res = r.json()["results"][0]
    assert not res["ok"] and res["fail_frame"] == 4, res
    # 切开：两个镜头各自可解
    r = solve_shots(meta["mat"], [(0, 4), (4, 6)])
    res = r.json()["results"]
    assert all(x["ok"] for x in res), res
    assert res[0]["trajectory"] and res[1]["trajectory"]
    print("拆分镜头：整段失败于全局帧 4；切开后两镜头各自可解")


def test_incremental_recompute():
    meta = upload(OBJ_XS)
    mat = meta["mat"]
    calls = []
    orig = m.solver.solve

    def counting(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)

    m.solver.solve = counting
    try:
        solve_shots(mat, [(0, 4), (4, 6)])
        assert len(calls) == 2, calls
        solve_shots(mat, [(0, 4), (4, 6)])          # 原样再来 -> 全命中缓存
        assert len(calls) == 2, calls
        # 只改镜头 1 内的锁定 -> 只重算镜头 1
        r = solve_shots(mat, [(0, 4), (4, 6)], locks={2: 5})
        assert len(calls) == 3, calls
        assert r.json()["results"][0]["ok"]
        # 再改镜头 2 内的锁定 -> 只重算镜头 2
        r = solve_shots(mat, [(0, 4), (4, 6)], locks={2: 5, 4: 25})
        assert len(calls) == 4, calls
        res = r.json()["results"]
        assert res[1]["trajectory"][0] == 25        # 锁定按镜头内下标生效
        assert res[0]["trajectory"]                 # 镜头 1 结果来自缓存
    finally:
        m.solver.solve = orig
    print("增量重算：改锁定只重算受影响镜头，其余命中缓存")


def test_stale_mat_rejected():
    meta = upload(OBJ_XS)
    r = solve_shots(meta["mat"] + 1, [(0, 6)])
    assert r.status_code == 409, r.status_code
    # 换素材后旧 mat 也被拒
    meta2 = upload(OBJ_XS)
    assert meta2["mat"] != meta["mat"]
    r = solve_shots(meta["mat"], [(0, 6)])
    assert r.status_code == 409, r.status_code
    print("迟到结果：素材版本不符返回 409，不得覆盖当前预览")


def test_fail_frame_global_in_second_shot():
    # 镜头 2 内帧 5 工件瞬移回 0 -> 该镜头失败，报告原片帧号 5
    meta = upload([2, 4, 6, 8, 50, 0])
    r = solve_shots(meta["mat"], [(0, 4), (4, 6)])
    res = r.json()["results"]
    assert res[0]["ok"], res[0]
    assert not res[1]["ok"] and res[1]["fail_frame"] == 5, res[1]
    print("镜头内不可解：报告原片（全局）首个接不上的帧")


def test_validation():
    meta = upload(OBJ_XS)
    mat = meta["mat"]
    assert solve_shots(mat, [(4, 4)]).status_code == 400        # 空范围
    assert solve_shots(mat, [(-1, 3)]).status_code == 400       # 越界
    assert solve_shots(mat, [(0, 7)]).status_code == 400        # 越界
    assert solve_shots(mat, [(0, 4)], locks={9: 3}).status_code == 400   # 锁定帧越界
    assert solve_shots(mat, [(0, 4)], locks={1: 999}).status_code == 400  # 锁定位置越界
    print("参数校验：非法镜头范围与锁定被拒")


if __name__ == "__main__":
    test_split_and_global_fail_frame()
    test_incremental_recompute()
    test_stale_mat_rejected()
    test_fail_frame_global_in_second_shot()
    test_validation()
    print("全部通过")
