"use strict";

const $ = (id) => document.getElementById(id);

let meta = null;        // {n,w,h,cw,x_max,boxes,feasible,mat,pairs}
let images = [];        // 每帧 Image
let cuts = [];          // 已确认切点：新镜头起始帧号（0 基），升序
let locks = {};         // 原片帧号 -> 锁定 x（锁定留在原帧，不随镜头变化）
let shots = [];         // {start,end,key,status:'pending'|'ok'|'fail'|'err',traj,failFrame,energy}
let threshold = 0.9;    // 建议阈值（只影响建议列表）
let cur = 0;
let shownX = 0;         // 当前帧画面上显示的裁切窗横坐标
let dragging = false;
let grabDX = 0;
let loadToken = 0;      // 会话版本，用于帧图片缓存刷新

// ------------------------------------------------------------ 状态提示

function setStatus(msg, kind) {
  const el = $("status");
  el.textContent = msg;
  el.className = "status " + (kind || "");
}

// ------------------------------------------------------------ 数据加载

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return handle(r);
}

async function handle(r) {
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || ("HTTP " + r.status));
  return data;
}

function naturalSort(files) {
  return [...files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
}

async function doUpload() {
  const fs = naturalSort($("fileFrames").files);
  const bf = $("fileBoxes").files[0];
  if (!fs.length || !bf) { setStatus("请选择 PNG 帧和必留框 JSON", "err"); return; }
  const fd = new FormData();
  for (const f of fs) fd.append("frames", f, f.name);
  fd.append("boxes", bf, bf.name);
  setStatus("上传中…");
  try {
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    await afterLoad(await handle(r));
  } catch (e) { setStatus("上传失败：" + e.message, "err"); }
}

async function loadSample(name) {
  setStatus("载入样片…");
  try {
    const r = await fetch("/api/load_sample?name=" + name, { method: "POST" });
    await afterLoad(await handle(r));
  } catch (e) { setStatus("载入样片失败：" + e.message, "err"); }
}

async function afterLoad(m) {
  meta = m;
  cuts = [];
  locks = {};
  shots = [];
  cur = 0;
  shownX = 0;
  loadToken++;
  images = await Promise.all([...Array(m.n).keys()].map(loadFrame));
  $("slider").max = m.n - 1;
  $("slider").value = 0;
  $("cvMain").width = m.w; $("cvMain").height = m.h;
  $("cvPreview").width = m.cw; $("cvPreview").height = m.h;
  $("cutPanel").hidden = false;
  $("shotPanel").hidden = false;
  $("viewer").hidden = false;
  const nReview = m.pairs.filter((p) => p.review).length;
  setStatus(`已载入 ${m.n} 帧（${m.w}×${m.h}）：${m.pairs.length} 对相邻帧已算背景相关，其中 ${nReview} 对待核对。求解中…`);
  await recompute();
}

function loadFrame(i) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("帧 " + i + " 加载失败"));
    img.src = "/api/frame/" + i + ".png?t=" + loadToken;
  });
}

// ------------------------------------------------------------ 镜头拆分与增量重算

function shotRanges() {
  const bounds = [0, ...cuts, meta.n];
  return bounds.slice(0, -1).map((s, i) => [s, bounds[i + 1]]);
}

function locksIn(s, e) {
  const o = {};
  for (const k of Object.keys(locks)) {
    const i = +k;
    if (i >= s && i < e) o[i] = locks[i];
  }
  return o;
}

// 镜头结果键：素材版本 + 帧范围 + 范围内锁定。任一变化才需要重算。
function shotKey(s, e) {
  return meta.mat + "|" + s + "-" + e + "|" + JSON.stringify(locksIn(s, e));
}

function shotOf(f) {
  return shots.find((sh) => f >= sh.start && f < sh.end);
}

async function recompute() {
  const ranges = shotRanges();
  // 范围与锁定未变的镜头保留原结果，照常预览；其余标为待求解
  shots = ranges.map(([s, e]) => {
    const key = shotKey(s, e);
    const old = shots.find((sh) => sh.start === s && sh.end === e && sh.key === key
      && (sh.status === "ok" || sh.status === "fail"));
    return old || { start: s, end: e, key, status: "pending" };
  });
  render();
  const pending = shots.filter((sh) => sh.status === "pending");
  if (!pending.length) { afterSolve(); return; }
  // 记录本次请求发出的 key，响应回来时逐项核对，迟到结果不得覆盖当前状态
  const sentKeys = new Map(pending.map((sh) => [sh.start + "-" + sh.end, sh.key]));
  try {
    const r = await postJSON("/api/solve_shots", {
      mat: meta.mat,
      shots: pending.map(({ start, end }) => ({ start, end })),
      locks,
    });
    for (const res of r.results) {
      const id = res.start + "-" + res.end;
      const curShot = shots.find((sh) => sh.start === res.start && sh.end === res.end);
      if (!curShot || curShot.key !== sentKeys.get(id)) continue; // 镜头范围/锁定/素材已变
      if (res.ok) {
        Object.assign(curShot, { status: "ok", traj: res.trajectory, energy: res.energy, failFrame: null });
      } else {
        Object.assign(curShot, { status: "fail", traj: null, failFrame: res.fail_frame });
      }
    }
  } catch (e) {
    if (/已过期|已更新/.test(e.message)) return; // 素材已变，新的一次重算会跟上
    for (const sh of shots) if (sh.status === "pending") sh.status = "err";
    setStatus("求解失败：" + e.message + "（可再次操作触发重算）", "err");
  }
  afterSolve();
}

function afterSolve() {
  const sh = shotOf(cur);
  // 当前所在镜头无解时，跳到该镜头在原片中首个接不上的帧
  if (sh && sh.status === "fail") gotoFrame(sh.failFrame);
  const failed = shots.filter((s) => s.status === "fail");
  if (shots.some((s) => s.status === "pending")) {
    setStatus("求解中…");
  } else if (failed.length) {
    const idx = shots.indexOf(failed[0]);
    setStatus(`镜头 ${idx + 1} 从原片第 ${failed[0].failFrame + 1} 帧起接不上，已撤下该镜头预览；其余镜头照常预览。`, "err");
  } else if (shots.length) {
    setStatus(`全部 ${shots.length} 个镜头求解成功。可拖动裁切窗并锁定。`, "ok");
  }
  render();
}

// ------------------------------------------------------------ 切点操作

function addCut(c) {
  if (!cuts.includes(c)) {
    cuts = [...cuts, c].sort((a, b) => a - b);
    recompute();
  }
}

function removeCut(c) {
  cuts = cuts.filter((x) => x !== c);
  recompute();
}

// ------------------------------------------------------------ 绘制

function render() {
  if (!meta) return;
  renderCuts();
  renderShots();
  renderLocks();

  const cv = $("cvMain"), ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(images[cur], 0, 0);

  // 必留框（红）
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#e33";
  for (const b of meta.boxes[cur] || []) {
    ctx.strokeRect(b[0] + 0.5, b[1] + 0.5, b[2] - b[0] - 1, b[3] - b[1] - 1);
  }

  const sh = shotOf(cur);
  const failed = sh && sh.status === "fail";
  // 无解镜头：撤下竖屏预览
  $("previewPane").style.visibility = failed ? "hidden" : "visible";

  // 裁切窗（锁定绿 / 未锁定蓝；失败时也可拖动以便重新锁定）
  const locked = locks[cur] !== undefined;
  ctx.strokeStyle = locked ? "#0a0" : "#28f";
  ctx.lineWidth = 2;
  ctx.strokeRect(shownX + 1, 1, meta.cw - 2, meta.h - 2);
  ctx.fillStyle = locked ? "#0a0" : "#28f";
  ctx.font = "12px sans-serif";
  ctx.fillText((locked ? "🔒 " : "") + "x=" + shownX, Math.min(shownX + 4, meta.w - 60), 14);

  if (!failed) {
    const pv = $("cvPreview").getContext("2d");
    pv.imageSmoothingEnabled = false;
    pv.drawImage(images[cur], shownX, 0, meta.cw, meta.h, 0, 0, meta.cw, meta.h);
  } else {
    ctx.fillStyle = "rgba(200,0,0,0.85)";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText(`本镜头从原片第 ${sh.failFrame + 1} 帧起无法接续`, 8, 18);
  }

  $("frameLabel").textContent = `第 ${cur + 1}/${meta.n} 帧`;
  const si = shots.indexOf(sh);
  $("shotLabel").textContent = sh
    ? `镜头 ${si + 1}/${shots.length}（第 ${sh.start + 1}–${sh.end} 帧）` + (failed ? " · 不可解，预览已撤下" : "")
    : "";
  $("btnCut").textContent = cuts.includes(cur)
    ? `撤销第 ${cur}–${cur + 1} 帧之间切点`
    : `在第 ${cur}–${cur + 1} 帧之间标记切点`;
  $("btnCut").disabled = cur === 0;
  $("btnLock").textContent = locked ? "更新锁定为当前位置" : "锁定本帧位置";
}

function renderCuts() {
  // 建议切点：相关系数低于阈值（调阈值只更新这里）
  const sug = meta.pairs.filter((p) => !p.review && p.corr !== null
    && p.corr < threshold && !cuts.includes(p.cut));
  $("suggest").innerHTML = `<b>建议切点（${sug.length}）</b>` + (sug.length ? "" : "：无")
    + sug.map((p) => candHtml(p, `相关系数 ${p.corr.toFixed(3)}`, "")).join("");
  // 待核对：有效背景不足或灰度恒定，给不出相关系数
  const rev = meta.pairs.filter((p) => p.review && !cuts.includes(p.cut));
  $("review").innerHTML = `<b>待核对（${rev.length}）</b>` + (rev.length ? "" : "：无")
    + rev.map((p) => candHtml(p, `${p.reason}（有效背景 ${(p.ratio * 100).toFixed(1)}%）`, " review")).join("");
  // 已确认切点
  $("cuts").innerHTML = cuts.length
    ? "已确认切点：" + cuts.map((c) =>
      `<span class="chip cut">第 ${c}–${c + 1} 帧之间<button class="x" data-uncut="${c}" title="撤销切点">✕</button></span>`).join("")
    : "已确认切点：无（整段素材为一个镜头）。";

  document.querySelectorAll("[data-go]").forEach((el) => {
    el.onclick = () => gotoFrame(+el.dataset.go); // 跳到切点后一帧
  });
  document.querySelectorAll("[data-mark]").forEach((el) => {
    el.onclick = (ev) => { ev.stopPropagation(); addCut(+el.dataset.mark); };
  });
  document.querySelectorAll("[data-uncut]").forEach((el) => {
    el.onclick = () => removeCut(+el.dataset.uncut);
  });
}

function candHtml(p, extra, cls) {
  return `<span class="cand${cls}" data-go="${p.cut}" title="点击跳到切点后一帧（第 ${p.cut + 1} 帧）">`
    + `第 ${p.cut}–${p.cut + 1} 帧之间 · ${extra}`
    + `<button class="mini" data-mark="${p.cut}">标为切点</button></span>`;
}

function renderShots() {
  $("shots").innerHTML = shots.map((sh, i) => {
    const label = `镜头 ${i + 1} · 第 ${sh.start + 1}–${sh.end} 帧`;
    if (sh.status === "pending") return `<span class="chip shot pending">${label} · 求解中…</span>`;
    if (sh.status === "err") return `<span class="chip shot fail" data-shot="${i}">${label} · 求解失败，点此重试</span>`;
    if (sh.status === "fail") return `<span class="chip shot fail" data-shot="${i}">${label} · ✗ 第 ${sh.failFrame + 1} 帧起接不上</span>`;
    return `<span class="chip shot ok" data-shot="${i}">${label} · ✓ 能量 ${Math.round(sh.energy).toLocaleString()}</span>`;
  }).join("");
  document.querySelectorAll("[data-shot]").forEach((el) => {
    el.onclick = () => {
      const sh = shots[+el.dataset.shot];
      if (sh.status === "err") { sh.status = "pending"; recompute(); return; }
      gotoFrame(sh.status === "fail" ? sh.failFrame : sh.start);
    };
  });
}

function renderLocks() {
  const el = $("locks");
  const keys = Object.keys(locks).map(Number).sort((a, b) => a - b);
  el.innerHTML = keys.length
    ? "已锁定：" + keys.map((i) => `<span class="chip">帧 ${i + 1} → x=${locks[i]}</span>`).join("")
    : "未锁定任何帧。";
}

// ------------------------------------------------------------ 交互

function clampX(x) {
  return Math.max(0, Math.min(meta.x_max, Math.round(x)));
}

function defaultX(i) {
  if (locks[i] !== undefined) return locks[i];
  const f = meta.feasible[i];
  return f ? Math.round((f[0] + f[1]) / 2) : 0;
}

function gotoFrame(i) {
  cur = Math.max(0, Math.min(meta.n - 1, i));
  $("slider").value = cur;
  const sh = shotOf(cur);
  shownX = sh && sh.status === "ok" ? sh.traj[cur - sh.start] : defaultX(cur);
  render();
}

$("btnUpload").onclick = doUpload;
$("btnSampleA").onclick = () => loadSample("a");
$("btnSampleB").onclick = () => loadSample("b");
$("btnSampleC").onclick = () => loadSample("c");
$("btnSampleD").onclick = () => loadSample("d");
$("btnPrev").onclick = () => gotoFrame(cur - 1);
$("btnNext").onclick = () => gotoFrame(cur + 1);
$("slider").oninput = (e) => gotoFrame(+e.target.value);

// 调阈值只更新建议列表，不触碰已确认切点，也不重新求解
$("thr").oninput = (e) => {
  threshold = +e.target.value;
  $("thrVal").textContent = threshold.toFixed(2);
  renderCuts();
};

$("btnCut").onclick = () => {
  if (cur === 0) return;
  if (cuts.includes(cur)) removeCut(cur); else addCut(cur);
};
$("btnLock").onclick = () => { locks[cur] = shownX; recompute(); };
$("btnUnlock").onclick = () => { delete locks[cur]; recompute(); };

const cvMain = $("cvMain");
cvMain.addEventListener("mousedown", (e) => {
  if (!meta) return;
  dragging = true;
  grabDX = e.offsetX - shownX;
});
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const rect = cvMain.getBoundingClientRect();
  shownX = clampX(e.clientX - rect.left - grabDX);
  render();
});
window.addEventListener("mouseup", async () => {
  if (!dragging) return;
  dragging = false;
  if (locks[cur] !== undefined && locks[cur] !== shownX) {
    locks[cur] = shownX;   // 拖动已锁定帧 = 改锁定位置，只重算该帧所在镜头
    await recompute();
  }
});
