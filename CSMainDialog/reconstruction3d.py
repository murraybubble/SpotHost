import numpy as np
import matplotlib
matplotlib.use('Agg')  # 禁止弹出窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import cv2

def generate_3d_image(gray_img):
    """
    根据灰度图生成伪3D表面重构图像（返回OpenCV格式BGR图）
    """
    if gray_img is None or len(gray_img.shape) != 2:
        raise ValueError("输入图像必须是灰度图。")

    # 缩放图像，避免太大造成绘图卡顿
    h, w = gray_img.shape
    scale = 400 / max(h, w) if max(h, w) > 400 else 1.0
    if scale < 1.0:
        gray_small = cv2.resize(gray_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray_img

    # 构建3D坐标
    X = np.arange(0, gray_small.shape[1])
    Y = np.arange(0, gray_small.shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = gray_small.astype(np.float32)

    # 归一化高度
    Z = cv2.GaussianBlur(Z, (5, 5), 0)
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z) + 1e-6)

    # 绘制3D表面
    fig = plt.figure(figsize=(4, 3), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap='viridis', linewidth=0, antialiased=True)

    ax.set_axis_off()
    ax.view_init(elev=60, azim=45)
    plt.tight_layout(pad=0)

    # 将matplotlib图像渲染为numpy数组
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 适配到label4大小（保持比例）
    img = cv2.resize(img, (400, 300), interpolation=cv2.INTER_AREA)

    return img
