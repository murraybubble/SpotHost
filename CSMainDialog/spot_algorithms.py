# spot_algorithms.py
import cv2
import numpy as np

# ---------------- 错误码 ----------------
(ERR_IMG_NONE, ERR_IMG_CHANNEL, ERR_IMG_TOO_SMALL, ERR_IMG_BLACK,
 ERR_BINARY_ALL_ZERO, ERR_OPEN_ALL_ZERO, ERR_NO_LOCAL_MAX,
 ERR_ALL_RADIUS_TOO_SMALL, ERR_NO_VALID_SPOT) = range(51, 60)

# ---------------- 仅提示，不终止 ----------------
def _die(code: int, msg: str):
    print(f"【检测错误 {code}】{msg}")
    return None   # 不再抛异常

# ---------------- 通用预处理 ----------------
def _pre_check(img):
    if img is None:
        _die(ERR_IMG_NONE, "读取图片失败（None）")
        return None
    if len(img.shape) != 3 or img.shape[2] != 3:
        _die(ERR_IMG_CHANNEL, "图片通道数≠3，请确认 BGR 图像")
        return None
    h, w = img.shape[:2]
    if min(h, w) < 20:
        _die(ERR_IMG_TOO_SMALL, f"图片尺寸过小 ({w}×{h})")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.max(gray) == 0:
        _die(ERR_IMG_BLACK, "整幅图全黑")
        return None
    return gray

# ================== A：标准多光斑 ==================
def _algo_A(img, max_spots=3):
    gray = _pre_check(img)
    if gray is None: return img         # 预处理失败，直接返原图，修改1
    thresh_val = int(np.max(gray) * 0.85)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    if not np.count_nonzero(binary):
        print(f"【检测错误 {ERR_BINARY_ALL_ZERO}】二值化后全黑")
        return img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    if not np.count_nonzero(opening):
        print(f"【检测错误 {ERR_OPEN_ALL_ZERO}】开运算后全黑")
        return img
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    local_max = (dist == cv2.dilate(dist, kernel)) & (dist > 0)
    coords = np.column_stack(np.where(local_max))
    if not len(coords):
        print(f"【检测错误 {ERR_NO_LOCAL_MAX}】无局部极大值")
        return img
    radii = dist[coords[:, 0], coords[:, 1]]
    idx = np.argsort(-radii)[:20]
    coords, radii = coords[idx], radii[idx]
    out, used = img.copy(), np.zeros_like(opening, bool)
    det = 0
    areas = []      # 保存面积
    centers = []    # 保存圆心坐标
    for (y, x), r in zip(coords, radii):
        if det >= max_spots: break
        r = int(r)
        # ---------- 半径钳位 ----------
        if r < 3 or r > 3000: continue
        mask = np.zeros_like(opening, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 1, -1)
        overlap = np.logical_and(used, mask.astype(bool)).sum() / (np.pi * r * r + 1e-6)
        if overlap > 0.5: continue
        if cv2.mean(gray, mask=mask * 255)[0] < thresh_val: continue
        # 画圆
        cv2.circle(out, (x, y), r, (0, 0, 255), 2)
        # 画圆心
        cv2.circle(out, (x, y), 3, (255, 0, 0), -1)
        # 计算面积
        area = int(np.sum(mask > 0))
        areas.append(area)
        centers.append((int(x), int(y)))
        # 写文字：面积 + 坐标
        text = f"{area}px  ({x},{y})"
        cv2.putText(out, text, (x + r + 5, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        used |= mask.astype(bool)
        det += 1
    if not det:
        print(f"【检测错误 {ERR_NO_VALID_SPOT}】最终可画光斑数为 0")
        return img #修改2
    # 控制台打印
    print("【光斑面积】", areas)
    print("【圆心坐标】", centers)
    return out #返回光斑中心，修改3

# ================== B：双光斑 ==================
def _algo_B(img, max_spots=2):
    return _algo_A(img, max_spots)

# ================== C：单光斑 + 去噪 ==================
def _algo_C(img, max_spots=1):
    gray = _pre_check(img)
    if gray is None: return img
    thresh_val = int(np.max(gray) * 0.85)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    if not np.count_nonzero(binary):
        print(f"【检测错误 {ERR_BINARY_ALL_ZERO}】二值化后全黑")
        return img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    if not np.count_nonzero(opening):
        print(f"【检测错误 {ERR_OPEN_ALL_ZERO}】开运算后全黑")
        return img
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
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
    if not candidates:
        print(f"【检测错误 {ERR_NO_LOCAL_MAX}】无有效候选光斑")
        return img
    candidates.sort(key=lambda x: x[2], reverse=True)
    keep = []
    for x, y, r in candidates:
        for (x2, y2, r2) in keep:
            if np.hypot(x - x2, y - y2) < min(r, r2): break
        else: keep.append((x, y, r))
    keep = keep[:max_spots]
    out, used = img.copy(), np.zeros_like(opening, bool)
    det = 0
    areas = []
    centers = []
    for x, y, r in keep:
        mask = np.zeros_like(opening, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 1, -1)
        if cv2.mean(gray, mask=mask * 255)[0] < thresh_val: continue
        cv2.circle(out, (x, y), r, (0, 0, 255), 2)
        cv2.circle(out, (x, y), 3, (255, 0, 0), -1)
        area = int(np.sum(mask > 0))
        areas.append(area)
        centers.append((int(x), int(y)))
        text = f"{area}px  ({x},{y})"
        cv2.putText(out, text, (x + r + 5, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        used |= mask.astype(bool); det += 1
    if not det:
        print(f"【检测错误 {ERR_NO_VALID_SPOT}】最终可画光斑数为 0")
        return img #修改4
    print("【光斑面积】", areas)
    print("【圆心坐标】", centers)
    return out,centers #修改5

# ================== D：框选后 + 二次亮度校验 ==================
def _algo_D(img, max_spots=1):
    # 直接走 A，已含圆心、面积输出
    return _algo_A(img, max_spots)

# ================== 统一对外接口 ==================
def detect_spots(img: np.ndarray, algo_type: str = "A", max_spots=3):
    algo_map = {"A": _algo_A, "B": _algo_B, "C": _algo_C, "D": _algo_D}
    if algo_type not in algo_map:
        raise ValueError(f"未知算法类型 {algo_type}")
    return algo_map[algo_type](img, max_spots)