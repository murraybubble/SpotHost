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
root.withdraw()  # 隐藏主窗口

# 全局变量
g_expValue = -1.0
g_gain = -1.0
g_autoAdjust = False

# 设置文件夹
SETTINGS_DIR = "camera_setting"
os.makedirs(SETTINGS_DIR, exist_ok=True)  # 自动创建文件夹

if platform.system() == 'Windows':
    sys.path.append(os.environ.get('IPX_CAMSDK_ROOT', '') + '/bin/win64_x64/')
    sys.path.append(os.environ.get('IPX_CAMSDK_ROOT', '') + '/bin/win32_i86/')
    import IpxCameraApiPy
else:
    import libIpxCameraApiPy as IpxCameraApiPy



def AutoAdjustExposureGain(camera,
                           target=200,
                           tol=10,
                           max_iter=10,
                           percentile=99.0,
                           min_bright_threshold=30):
    """
    最稳固的自动曝光/增益调节函数。
    关键点：
    - Stop → StartAcq → QueueBuffer → ExecStart → GetBuffer
    - 避免任何 None buffer
    """
    pars = camera.GetCameraParameters()
    parExp = pars.GetFloat("ExposureTimeRaw") or pars.GetInt("ExposureTimeRaw")
    parGain = pars.GetFloat("GainRaw")        or pars.GetInt("GainRaw")
    if parExp is None or parGain is None:
        print("相机不支持曝光或增益参数")
        return False

    expMin, expMax = parExp.GetMin()[1], parExp.GetMax()[1]
    gainMin, gainMax = parGain.GetMin()[1], parGain.GetMax()[1]

    stream = camera.GetStreamByIndex(0)
    if stream is None:
        print("无法获取 stream")
        return False

    # -----------------------
    #   停止采集，清空缓冲
    # -----------------------
    try:
        pars.ExecuteCommand("AcquisitionStop")
    except:
        pass
    try:
        stream.StopAcquisition(1)
    except:
        pass

    stream.FlushBuffers(stream.Flush_AllDiscard)

    # -----------------------
    #   重新启动采集
    # -----------------------
    stream.StartAcquisition()     # 顺序必须是先 StartAcq
    time.sleep(0.05)

    try:
        pars.ExecuteCommand("AcquisitionStart")
    except:
        pass

    # -----------------------
    #   建立缓冲区
    # -----------------------
    buf_size = stream.GetBufferSize()
    num_buf = max(4, stream.GetMinNumBuffers())
    buffers = []
    for _ in range(num_buf):
        b = stream.CreateBuffer(buf_size)
        stream.QueueBuffer(b)
        buffers.append(b)

    time.sleep(0.12)

    # -----------------------
    #   读取初始图像
    # -----------------------
    buf = stream.GetBuffer(3000)
    if buf is None:
        print("❌ 初始帧抓取超时（GetBuffer = None）")
        return False

    ptr = buf.GetBufferPtr()
    if ptr is None:
        print("❌ 初始缓冲区为空")
        stream.QueueBuffer(buf)
        return False

    img = np.frombuffer(ptr, np.uint8).reshape(buf.GetHeight(), buf.GetWidth())
    initial_bright = np.percentile(img, percentile)
    stream.QueueBuffer(buf)

    print(f"初始亮度 P{percentile} = {initial_bright:.2f}")

    if initial_bright < min_bright_threshold:
        print("⚠ 图像太暗，没有光斑，自动调节终止")
        return False

    # -----------------------
    #   调节循环
    # -----------------------
    exp = parExp.GetValue()[1]
    gain = parGain.GetValue()[1]

    ok = False

    for i in range(max_iter):
        buf = stream.GetBuffer(2000)
        if buf is None or buf.GetBufferPtr() is None:
            print(f"第 {i+1} 次抓取失败，继续下一轮")
            continue

        img = np.frombuffer(buf.GetBufferPtr(), np.uint8).reshape(buf.GetHeight(), buf.GetWidth())
        stream.QueueBuffer(buf)

        bright = np.percentile(img, percentile)
        print(f"[{i+1}] 当前亮度: {bright:.2f} (exp={exp:.1f}, gain={gain:.2f})")

        if abs(bright - target) <= tol:
            print("亮度达到目标范围，调节完成")
            ok = True
            break

        factor = target / (bright + 1e-6)

        # 先调曝光
        new_exp = float(np.clip(exp * factor, expMin, expMax))
        parExp.SetValue(new_exp)

        # 曝光触顶则调增益
        if new_exp >= expMax * 0.98:
            new_gain = float(np.clip(gain * factor, gainMin, gainMax))
            parGain.SetValue(new_gain)
            gain = new_gain

        exp = new_exp
        time.sleep(0.1)

    # -----------------------
    #   清理资源
    # -----------------------
    try:
        pars.ExecuteCommand("AcquisitionStop")
    except:
        pass
    try:
        stream.StopAcquisition(1)
    except:
        pass

    for b in buffers:
        stream.RevokeBuffer(b)

    stream.FlushBuffers(stream.Flush_AllDiscard)
    print("自动调节完成，资源已释放")

    return ok



def SetupExposure(camera, expValue):
    """
    设置积分时间（ExposureTimeRaw），使用提供的 expValue。
    """
    pars = camera.GetCameraParameters()
    if pars is None:
        print("错误：无法获取相机参数")
        return False

    parAec = pars.GetEnum("ExposureAuto")
    if parAec:
        try:
            parAec.SetValueStr("Off")
        except Exception as e:
            print(f"警告：关闭自动曝光失败: {e}")

    parExp = pars.GetFloat("ExposureTimeRaw")
    if parExp is None:
        parExp = pars.GetInt("ExposureTimeRaw")
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
    """
    设置增益（GainRaw），使用提供的 gainValue。
    """
    pars = camera.GetCameraParameters()
    if pars is None:
        print("错误：无法获取相机参数")
        return False

    parAgc = pars.GetEnum("GainAuto")
    if parAgc:
        try:
            parAgc.SetValueStr("Off")
        except Exception as e:
            print(f"警告：关闭自动增益失败: {e}")

    parG = pars.GetInt("GainRaw")
    if parG is None:
        parG = pars.GetFloat("GainRaw")
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
    """
    保存当前相机的积分时间和增益到用户指定的文件（弹窗选择文件名）
    文件保存在 camera_setting 文件夹下
    """
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

    # 弹窗让用户选择保存路径和文件名
    file_path = filedialog.asksaveasfilename(
        title="保存相机设置",
        initialdir=SETTINGS_DIR,
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        initialfile="camera_config.txt"
    )

    if not file_path:  # 用户取消
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
    """
    从用户选择的文件中读取积分时间和增益并应用到相机上（弹窗选择文件）
    """
    file_path = filedialog.askopenfilename(
        title="加载相机设置",
        initialdir=SETTINGS_DIR,
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    if not file_path:  # 用户取消
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

        # 应用设置
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