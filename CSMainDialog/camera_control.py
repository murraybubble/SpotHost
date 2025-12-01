import sys
import os
import platform
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox

# 隐藏 tkinter 主窗口
root = tk.Tk()
root.withdraw()

# 全局变量
g_expValue = -1.0
g_gain = -1.0
g_autoAdjust = False

# 设置文件夹
SETTINGS_DIR = "camera_setting"
os.makedirs(SETTINGS_DIR, exist_ok=True)

if platform.system() == 'Windows':
    sys.path.append(os.environ.get('IPX_CAMSDK_ROOT', '') + '/bin/win64_x64/')
    sys.path.append(os.environ.get('IPX_CAMSDK_ROOT', '') + '/bin/win32_i86/')
    import IpxCameraApiPy
else:
    import libIpxCameraApiPy as IpxCameraApiPy


# ============================================================================
#  自动调节曝光 & 增益（曝光强制限制在 [exp_lo, exp_hi] 默认 10000–30000 µs）
# ============================================================================
def AutoAdjustExposureGain(camera,
                           target=200.0,
                           tol=10.0,
                           max_iter=10,
                           percentile=99.0,
                           min_bright_threshold=50.0,
                           exp_lo=10000.0,
                           exp_hi=30000.0):
    """
    自动调节曝光和增益，使图像中第 percentile% 最亮的像素平均亮度接近 target。
    曝光被强制约束在 [exp_lo, exp_hi] 区间内，剩余亮度差用增益补。
    特别适合：有局部光斑/激光点、背景黑暗的场景（避免过曝噪声）
    """
    pars = camera.GetCameraParameters()
    parExp = pars.GetFloat("ExposureTimeRaw") or pars.GetInt("ExposureTimeRaw")
    parG = pars.GetFloat("GainRaw") or pars.GetInt("GainRaw")
    if parExp is None or parG is None:
        print("错误：相机不支持 ExposureTimeRaw 或 GainRaw 参数")
        return False

    expMin, expMax = parExp.GetMin()[1], parExp.GetMax()[1]
    gainMin, gainMax = parG.GetMin()[1], parG.GetMax()[1]
    # 与相机物理范围再取交集，防止相机本身范围更小
    exp_lo = max(exp_lo, expMin)
    exp_hi = min(exp_hi, expMax)

    # 关闭自动曝光/增益
    try:
        parAec = pars.GetEnum("ExposureAuto")
        if parAec:
            parAec.SetValueStr("Off")
    except Exception as e:
        print(f"警告：关闭自动曝光失败: {e}")

    try:
        parAgc = pars.GetEnum("GainAuto")
        if parAgc:
            parAgc.SetValueStr("Off")
    except Exception as e:
        print(f"警告：关闭自动增益失败: {e}")

    # 启动采集
    try:
        acqMode = pars.GetEnum("AcquisitionMode")
        if acqMode:
            acqMode.SetValueStr("Continuous")
    except:
        pass

    stream = camera.GetStreamByIndex(0)
    if stream is None:
        print("错误：无法获取数据流")
        return False

    buffers = []
    try:
        stream.FlushBuffers(stream.Flush_AllDiscard)
        bufSize = stream.GetBufferSize()
        numBuf = max(4, stream.GetMinNumBuffers())
        buffers = [stream.CreateBuffer(bufSize) for _ in range(numBuf)]
        for b in buffers:
            stream.QueueBuffer(b)

        stream.StartAcquisition()
        pars.ExecuteCommand("AcquisitionStart")
        time.sleep(0.5)

        # 先抓取一帧初始图像进行光斑检测
        buf = stream.GetBuffer(2000)
        if not hasattr(buf, 'GetBufferPtr') or buf.GetBufferPtr() is None:
            print("初始缓冲区无效或为空")
            stream.QueueBuffer(buf)
            return False

        img = np.frombuffer(buf.GetBufferPtr(), dtype=np.uint8).reshape(
            buf.GetHeight(), buf.GetWidth())
        initial_bright_value = np.percentile(img, percentile)
        print(f"初始检测: 第 {percentile}% 亮度 = {initial_bright_value:.2f}")

        if initial_bright_value < min_bright_threshold:
            print(f"初始亮度 {initial_bright_value:.2f} < {min_bright_threshold}，无光斑检测到，不进行调节")
            stream.QueueBuffer(buf)
            return True

        stream.QueueBuffer(buf)

        # 读取当前值
        current_exp = parExp.GetValue()[1]
        current_gain = parG.GetValue()[1]

        for i in range(max_iter):
            buf = stream.GetBuffer(2000)
            if buf is None or not hasattr(buf, 'GetBufferPtr') or buf.GetBufferPtr() is None:
                if buf: stream.QueueBuffer(buf)
                continue

            img = np.frombuffer(buf.GetBufferPtr(), dtype=np.uint8).reshape(
                buf.GetHeight(), buf.GetWidth())
            bright_value = np.percentile(img, percentile)
            print(f"迭代 {i + 1}: 第 {percentile}% 亮度 = {bright_value:.2f}  "
                  f"(曝光={current_exp:.1f}, 增益={current_gain:.2f})")

            if abs(bright_value - target) <= tol:
                print("已达到目标亮度范围，停止调节")
                stream.QueueBuffer(buf)
                break

            factor = target / (bright_value + 1e-6)

            # 1. 计算理想曝光
            new_exp_raw = current_exp * factor
            # 2. 硬裁剪到合理区间
            new_exp = np.clip(new_exp_raw, exp_lo, exp_hi)
            # 3. 计算剩余比例
            residual_factor = 1.0
            if new_exp != new_exp_raw:
                residual_factor = new_exp_raw / new_exp
            # 4. 应用曝光
            parExp.SetValue(new_exp)
            current_exp = new_exp
            # 5. 应用补偿增益
            new_gain = np.clip(current_gain * residual_factor, gainMin, gainMax)
            parG.SetValue(new_gain)
            current_gain = new_gain

            stream.QueueBuffer(buf)
            time.sleep(0.12)

        else:
            print("达到最大迭代次数，自动调节结束（可能未完全收敛）")

    except Exception as e:
        print(f"自动调节异常: {e}")
        return False
    finally:
        try:
            pars.ExecuteCommand("AcquisitionStop")
            stream.StopAcquisition(1)
            for b in buffers:
                stream.RevokeBuffer(b)
            stream.FlushBuffers(stream.Flush_AllDiscard)
            print("自动调节完成，资源已释放")
        except Exception as e:
            print(f"清理资源失败: {e}")
    return True


# ---------------- 以下函数保持原样 ----------------
def SetupExposure(camera, expValue):
    pars = camera.GetCameraParameters()
    if pars is None:
        print("错误：无法获取相机参数")
        return False
    try:
        parAec = pars.GetEnum("ExposureAuto")
        if parAec:
            parAec.SetValueStr("Off")
    except Exception as e:
        print(f"警告：关闭自动曝光失败: {e}")

    parExp = pars.GetFloat("ExposureTimeRaw") or pars.GetInt("ExposureTimeRaw")
    if parExp is None:
        print('错误：找不到 ExposureTimeRaw 参数')
        return False
    expMin, expMax = parExp.GetMin()[1], parExp.GetMax()[1]
    if not (expMin <= expValue <= expMax):
        print(f'积分时间 {expValue} 超出范围 [{expMin}, {expMax}]')
        return False
    try:
        parExp.SetValue(expValue)
        print(f'积分时间设置为 {expValue}')
        return True
    except Exception as e:
        print(f'设置积分时间失败: {e}')
        return False


def SetupGain(camera, gainValue):
    pars = camera.GetCameraParameters()
    if pars is None:
        print("错误：无法获取相机参数")
        return False
    try:
        parAgc = pars.GetEnum("GainAuto")
        if parAgc:
            parAgc.SetValueStr("Off")
    except Exception as e:
        print(f"警告：关闭自动增益失败: {e}")

    parG = pars.GetInt("GainRaw") or pars.GetFloat("GainRaw")
    if parG is None:
        print('错误：找不到 GainRaw 参数')
        return False
    gainMin, gainMax = parG.GetMin()[1], parG.GetMax()[1]
    if not (gainMin <= gainValue <= gainMax):
        print(f'增益 {gainValue} 超出范围 [{gainMin}, {gainMax}]')
        return False
    try:
        parG.SetValue(gainValue)
        print(f'增益设置为 {gainValue}')
        return True
    except Exception as e:
        print(f'设置增益失败: {e}')
        return False


def SaveExposureAndGain(camera):
    pars = camera.GetCameraParameters()
    if pars is None:
        messagebox.showerror("错误", "无法获取相机参数")
        return False
    parExp = pars.GetFloat("ExposureTimeRaw") or pars.GetInt("ExposureTimeRaw")
    parG = pars.GetFloat("GainRaw") or pars.GetInt("GainRaw")
    if parExp is None or parG is None:
        messagebox.showerror("错误", "无法获取 ExposureTimeRaw 或 GainRaw 参数")
        return False
    expValue = parExp.GetValue()[1]
    gainValue = parG.GetValue()[1]

    file_path = filedialog.asksaveasfilename(
        title="保存相机设置",
        initialdir=SETTINGS_DIR,
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        initialfile="camera_config.txt")
    if not file_path:
        print("保存取消")
        return False
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"exposure: {expValue}\n")
            f.write(f"gain: {gainValue}\n")
        print(f"当前设置已保存到: {file_path}")
        messagebox.showinfo("成功", f"设置已保存到:\n{file_path}")
        return True
    except Exception as e:
        messagebox.showerror("保存失败", f"保存文件失败:\n{str(e)}")
        return False


def LoadExposureAndGain(camera):
    file_path = filedialog.askopenfilename(
        title="加载相机设置",
        initialdir=SETTINGS_DIR,
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if not file_path:
        print("加载取消")
        return False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        expValue = None
        gainValue = None
        for line in lines:
            line = line.strip()
            if line.startswith("exposure:"):
                expValue = float(line.split(":", 1)[1].strip())
            elif line.startswith("gain:"):
                gainValue = float(line.split(":", 1)[1].strip())
        if expValue is None or gainValue is None:
            messagebox.showerror("格式错误", "文件中缺少 exposure 或 gain 值")
            return False
        if not SetupExposure(camera, expValue):
            return False
        if not SetupGain(camera, gainValue):
            return False
        print(f"从 {file_path} 加载并应用设置成功")
        messagebox.showinfo("成功", f"已加载设置:\n曝光={expValue}\n增益={gainValue}")
        return True
    except Exception as e:
        messagebox.showerror("加载失败", f"读取或应用文件失败:\n{str(e)}")
        return False