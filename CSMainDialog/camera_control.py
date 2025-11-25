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



def AutoAdjustExposureGain(camera, target=200.0, tol=10.0, max_iter=10, percentile=99.0, min_bright_threshold=50.0):
    """
    自动调节曝光和增益，使图像中第 percentile% 最亮的像素平均亮度接近 target。
    特别适合：有局部光斑/激光点、背景黑暗的场景（避免过曝噪声）
    修改：如果初始图像中第 percentile% 最亮的像素平均亮度低于 min_bright_threshold，则认为没有光斑，不进行调节。

    参数:
        target: 目标亮度 (建议 180~220 for 8bit)
        tol: 容差
        max_iter: 最大迭代次数
        percentile: 使用第几个百分位的亮度作为判断依据，推荐 99.0 ~ 99.9
                    99.0  → 最亮的 1% 像素的平均值
                    99.9  → 最亮的 0.1% 像素（更严格，只看最亮区域）
        min_bright_threshold: 如果初始亮度低于此阈值，则认为没有光斑，不调节（默认50.0）
    """
    pars = camera.GetCameraParameters()
    parExp = pars.GetFloat("ExposureTimeRaw") or pars.GetInt("ExposureTimeRaw")
    parG = pars.GetFloat("GainRaw") or pars.GetInt("GainRaw")
    if parExp is None or parG is None:
        print("错误：相机不支持 ExposureTimeRaw 或 GainRaw 参数")
        return False

    expMin, expMax = parExp.GetMin()[1], parExp.GetMax()[1]
    gainMin, gainMax = parG.GetMin()[1], parG.GetMax()[1]

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
        if buf is None:
            print("初始图像抓取失败，无法检测光斑")
            return False

        # 转为 numpy 图像
        img = np.frombuffer(buf.GetBufferPtr(), dtype=np.uint8).reshape(
            buf.GetHeight(), buf.GetWidth())

        # 计算初始高百分位亮度
        initial_bright_value = np.percentile(img, percentile)
        print(f"初始检测: 第 {percentile}% 亮度 = {initial_bright_value:.2f}")

        stream.QueueBuffer(buf)

        # 如果初始亮度低于阈值，认为没有光斑，不调节
        if initial_bright_value < min_bright_threshold:
            print(f"初始亮度 {initial_bright_value:.2f} < {min_bright_threshold}，无光斑检测到，不进行调节")
            return True  # 返回True表示不调节，但操作成功（保持不变）

        # 如果有光斑，继续调节
        current_exp = parExp.GetValue()[1]
        current_gain = parG.GetValue()[1]

        for i in range(max_iter):
            buf = stream.GetBuffer(2000)
            if buf is None:
                print(f"第 {i + 1} 次未抓到图像")
                continue

            # 转为 numpy 图像
            img = np.frombuffer(buf.GetBufferPtr(), dtype=np.uint8).reshape(
                buf.GetHeight(), buf.GetWidth())

            # 关键修改：使用高百分位亮度，而不是全局平均
            bright_value = np.percentile(img, percentile)
            print(f"迭代 {i + 1}: 第 {percentile}% 亮度 = {bright_value:.2f}  "
                  f"(曝光={current_exp:.1f}, 增益={current_gain:.2f})")

            # 判断是否满足目标
            if abs(bright_value - target) <= tol:
                print(f"已达到目标亮度范围 [{target - tol}, {target + tol}]，停止调节")
                stream.QueueBuffer(buf)
                break

            # 计算调节倍率（避免除零）
            factor = target / (bright_value + 1e-6)

            # 先调节曝光（优先）
            new_exp = np.clip(current_exp * factor, expMin, expMax)
            parExp.SetValue(new_exp)

            # 如果曝光已到上限但仍太暗，才动增益；如果过亮则先降曝光再降增益
            if bright_value < target - tol and new_exp >= expMax * 0.98:
                # 还暗，且曝光已近上限 → 增加增益
                new_gain = np.clip(current_gain * factor * 1.2, gainMin, gainMax)  # 稍微多补一点
                parG.SetValue(new_gain)
                current_gain = new_gain
            elif bright_value > target + tol and new_exp <= expMin * 1.02:
                # 太亮了，且曝光已近下限 → 降低增益
                new_gain = np.clip(current_gain / max(factor, 1.1), gainMin, gainMax)
                parG.SetValue(new_gain)
                current_gain = new_gain

            current_exp = new_exp
            stream.QueueBuffer(buf)
            time.sleep(0.12)  # 给相机响应时间

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