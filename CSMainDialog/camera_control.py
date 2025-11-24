import sys
import os
import platform
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox

# 隐藏 tkinter 主窗口123456
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

def AutoAdjustExposureGain(camera, target=170.0, tol=5.0, max_iter=8):
    # （保持原函数不变）
    pars = camera.GetCameraParameters()
    parExp = pars.GetFloat("ExposureTimeRaw") or pars.GetInt("ExposureTimeRaw")
    parG = pars.GetFloat("GainRaw") or pars.GetInt("GainRaw")
    if parExp is None or parG is None:
        print("错误：相机不支持 ExposureTimeRaw 或 GainRaw 参数（无法自动调节）")
        return False

    expMin, expMax = parExp.GetMin()[1], parExp.GetMax()[1]
    gainMin, gainMax = parG.GetMin()[1], parG.GetMax()[1]

    try:
        acqMode = pars.GetEnum("AcquisitionMode")
        if acqMode:
            acqMode.SetValueStr("Continuous")
            print("采集模式设置为 Continuous")
    except Exception as e:
        print("设置 Continuous 模式失败:", e)

    stream = camera.GetStreamByIndex(0)
    if stream is None:
        print("错误：无法获取数据流")
        return False

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

        for i in range(max_iter):
            buf = stream.GetBuffer(1000)
            if buf is None:
                print(f"第 {i} 次调节时未获取到 buffer（超时）")
                continue

            img = np.frombuffer(buf.GetBufferPtr(), dtype=np.uint8).reshape(buf.GetHeight(), buf.GetWidth())
            brightness = float(np.mean(img))
            print(f"迭代 {i}: 平均亮度 = {brightness:.2f}")

            if abs(brightness - target) <= tol:
                print("亮度已在目标范围内，停止调节。")
                stream.QueueBuffer(buf)
                break

            factor = target / (brightness + 1e-6)
            newExp = min(max(parExp.GetValue()[1] * factor, expMin), expMax)
            parExp.SetValue(newExp)

            if (brightness < target and newExp >= expMax * 0.98) or (brightness > target and newExp <= expMin * 1.02):
                newG = min(max(parG.GetValue()[1] * factor, gainMin), gainMax)
                parG.SetValue(newG)

            print(f"设置后：曝光={parExp.GetValue()[1]:.1f}, 增益={parG.GetValue()[1]:.2f}")
            stream.QueueBuffer(buf)
            time.sleep(0.1)

    except Exception as e:
        print(f"自动调节过程中发生错误: {str(e)}")
        return False
    finally:
        try:
            pars.ExecuteCommand("AcquisitionStop")
            stream.StopAcquisition(1)
            for b in buffers:
                stream.RevokeBuffer(b)
            stream.FlushBuffers(stream.Flush_AllDiscard)
            print("自动调节结束，缓冲区已清理。")
        except Exception as e:
            print(f"清理缓冲区或停止采集失败: {str(e)}")
            return False

    return True

def SetupExposure(camera, expValue):
    # （保持原函数不变）
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
        print(f'快门时间 {expValue} 超出范围 [{expMin}, {expMax}]')
        return False

    try:
        parExp.SetValue(expValue)
        print(f'快门时间设置为 {expValue}')
        return True
    except Exception as e:
        print(f'设置快门时间失败: {e}')
        return False

def SetupGain(camera, gainValue):
    # （保持原函数不变）
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
    保存当前相机的快门时间和增益到用户指定的文件（弹窗选择文件名）
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
    从用户选择的文件中读取快门时间和增益并应用到相机上（弹窗选择文件）
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
