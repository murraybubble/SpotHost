# spot_algorithms.py
import cv2
import numpy as np

# ---------- 错误码 ----------
(ERR_IMG_NONE, ERR_IMG_CHANNEL, ERR_IMG_TOO_SMALL, ERR_IMG_BLACK,
 ERR_BINARY_ALL_ZERO, ERR_OPEN_ALL_ZERO, ERR_NO_LOCAL_MAX,
 ERR_ALL_RADIUS_TOO_SMALL, ERR_NO_VALID_SPOT) = range(51, 60)

def _die(code: int, msg: str):
    # 统一异常出口，返回 None 让上层弹窗
    raise RuntimeError(f"【检测错误 {code}】{msg}")

# ---------- 通用预处理 ----------
def _pre_check(img):
    if img is None:                       _die(ERR_IMG_NONE, "读取图片失败（None）")
    if len(img.shape) != 3 or img.shape[2] != 3:
        _die(ERR_IMG_CHANNEL, "图片通道数≠3，请确认 BGR 图像")
    h, w = img.shape[:2]
    if min(h, w) < 20:                    _die(ERR_IMG_TOO_SMALL, f"图片尺寸过小 ({w}×{h})")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.max(gray) == 0:                 _die(ERR_IMG_BLACK, "整幅图全黑")
    return gray

# ---------- A：标准多光斑 ----------
def _algo_A(img, max_spots=3):
    gray = _pre_check(img)
    thresh_val = int(np.max(gray) * 0.85)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    if not np.count_nonzero(binary):      _die(ERR_BINARY_ALL_ZERO, "二值化后全黑")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    if not np.count_nonzero(opening):     _die(ERR_OPEN_ALL_ZERO, "开运算后全黑")
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    local_max = (dist == cv2.dilate(dist, kernel)) & (dist > 0)
    coords = np.column_stack(np.where(local_max))
    if not len(coords):                   _die(ERR_NO_LOCAL_MAX, "无局部极大值")
    radii = dist[coords[:, 0], coords[:, 1]]
    idx = np.argsort(-radii)[:20]
    coords, radii = coords[idx], radii[idx]
    out, used = img.copy(), np.zeros_like(opening, bool)
    det = 0
    for (y, x), r in zip(coords, radii):
        if det >= max_spots: break
        r = int(r)
        if r < 3 or r > 3000:          # 半径异常直接跳过
            continue
        mask = np.zeros_like(opening, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 1, -1)
        overlap = np.logical_and(used, mask.astype(bool)).sum() / (np.pi * r * r + 1e-6)
        if overlap > 0.5: continue
        if cv2.mean(gray, mask=mask * 255)[0] < thresh_val: continue
        cv2.circle(out, (x, y), r, (0, 0, 255), 2)
        used |= mask.astype(bool); det += 1
    if not det: _die(ERR_NO_VALID_SPOT, "最终可画光斑数为 0")
    return out

# ---------- B：双光斑（先完全复用 A，后续可改参） ----------
def _algo_B(img, max_spots=2):   # 仅减少 max_spots
    return _algo_A(img, max_spots)

# ---------- C：单光斑 + 去噪 ----------
def _algo_C(img, max_spots=1):
    gray = _pre_check(img)
    thresh_val = int(np.max(gray) * 0.85)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    if not np.count_nonzero(binary): _die(ERR_BINARY_ALL_ZERO, "二值化后全黑")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    if not np.count_nonzero(opening): _die(ERR_OPEN_ALL_ZERO, "开运算后全黑")
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 95% 掩膜
    thr = np.percentile(dist[dist > 0], 95)
    mask = (dist >= thr).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for i in range(1, n_labels):
        x, y = centroids[i][:2]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3: continue
        r = max(3, int(np.sqrt(area / np.pi) * 3.6))
        candidates.append((int(x), int(y), r))
    if not candidates: _die(ERR_NO_LOCAL_MAX, "无有效候选光斑")
    candidates.sort(key=lambda x: x[2], reverse=True)
    # 简单 NMS
    keep = []
    for x, y, r in candidates:
        for (x2, y2, r2) in keep:
            if np.hypot(x - x2, y - y2) < min(r, r2): break
        else: keep.append((x, y, r))
    keep = keep[:max_spots]
    # 画圈
    out, used = img.copy(), np.zeros_like(opening, bool)
    det = 0
    for x, y, r in keep:
        mask = np.zeros_like(opening, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 1, -1)
        if cv2.mean(gray, mask=mask * 255)[0] < thresh_val: continue
        cv2.circle(out, (x, y), r, (0, 0, 255), 2)
        used |= mask.astype(bool); det += 1
    if not det: _die(ERR_NO_VALID_SPOT, "最终可画光斑数为 0")
    return out

# ---------- D：框选后 + 二次亮度校验 ----------
def _algo_D(img, max_spots=3):
    # 先走 A 流程
    out = _algo_A(img, max_spots)
    # D 额外再做一次整体亮度过滤（已在 A 中 mean_val<thresh_val 完成）
    # 如需更严可把阈值提高，这里保持与 A 一致即可
    return out

# ---------- 统一对外接口 ----------
def detect_spots(img: np.ndarray, algo_type: str = "A", max_spots=3):
    """
    algo_type:  "A" 标准
                "B" 双光斑
                "C" 单光斑去噪
                "D" 框选识别
    """
    algo_map = {"A": _algo_A, "B": _algo_B, "C": _algo_C, "D": _algo_D}
    if algo_type not in algo_map:
        raise ValueError(f"未知算法类型 {algo_type}")
    return algo_map[algo_type](img, max_spots)