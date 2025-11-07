import cv2
import numpy as np

def preprocess_image_cv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    return gray, blur


def detect_and_draw_spots(
    img: np.ndarray,
    intensity_ratio: float = 0.85,
    max_spots: int = 3,
    morph_kernel_size: int = 5,
    log_func=None  # ← 新增：日志回调函数
) -> np.ndarray:
    """
    自动检测图像中亮度较高的光斑区域，并返回绘制结果。
    """



    if img is None or not isinstance(img, np.ndarray):
        log_func("输入图像无效，请传入OpenCV格式的numpy数组。")
        raise ValueError("输入图像无效，请传入OpenCV格式的numpy数组。")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_val = np.max(gray)
    threshold_val = int(max_val * intensity_ratio)
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    local_max = (dist_transform == cv2.dilate(dist_transform, dilate_kernel)) & (dist_transform > 0)
    spot_coords = np.column_stack(np.where(local_max))

    if len(spot_coords) == 0:
        log_func("未检测到光斑。")
        return img.copy()

    spot_radii = dist_transform[spot_coords[:, 0], spot_coords[:, 1]]
    sorted_idx = np.argsort(-spot_radii)
    spot_coords = spot_coords[sorted_idx][:20]
    spot_radii = spot_radii[sorted_idx][:20]

    output = img.copy()
    used_mask = np.zeros_like(cleaned, dtype=bool)
    detected_spots = []

    h, w = gray.shape[:2]

    for i in range(len(spot_coords)):
        if len(detected_spots) >= max_spots:
            break

        y, x = spot_coords[i]
        radius = int(spot_radii[i])
        if radius < 3:
            continue

        radius = int(np.clip(radius, 1, min(h, w) // 2))
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        spot_mask = np.zeros_like(cleaned, dtype=np.uint8)

        try:
            cv2.circle(spot_mask, (x, y), radius, 1, -1)
        except Exception as e:
            log_func(f"[警告] 绘制光斑失败: (x={x}, y={y}, r={radius}), 错误: {e}")
            continue

        overlap = np.logical_and(used_mask, spot_mask.astype(bool)).sum() / (np.pi * radius * radius + 1e-6)
        if overlap > 0.5:
            continue

        mean_intensity = cv2.mean(gray, mask=spot_mask)[0]
        if mean_intensity < threshold_val:
            continue

        cv2.circle(output, (x, y), radius, (0, 0, 255), 2)
        used_mask = np.logical_or(used_mask, spot_mask.astype(bool))
        detected_spots.append((x, y, radius))

    if len(detected_spots) > 0:
        msg = "检测到光斑位置（x, y, radius）：\n"
        for i, (x, y, r) in enumerate(detected_spots):
            msg += f"  光斑 {i + 1}: ({x}, {y}), 半径 = {r}\n"
        if log_func:
            log_func(msg)
        else:
            print(msg)
    else:
        if log_func:
            log_func("未检测到符合条件的光斑。")
        else:
            print("未检测到符合条件的光斑。")

    return output


def energy_distribution(gray):
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heatmap
