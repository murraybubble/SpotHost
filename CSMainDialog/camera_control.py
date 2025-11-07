import sys
import os
import platform
import cv2
import numpy as np
import time

# 全局变量用于快门时间和增益设置
g_expValue = -1.0
g_gain = -1.0
g_autoAdjust = False

if platform.system() == 'Windows':
    sys.path.append(os.environ.get('IPX_CAMSDK_ROOT', '') + '/bin/win64_x64/')
    sys.path.append(os.environ.get('IPX_CAMSDK_ROOT', '') + '/bin/win32_i86/')
    import IpxCameraApiPy
else:
    import libIpxCameraApiPy as IpxCameraApiPy

def AutoAdjustExposureGain(camera, target=170.0, tol=5.0, max_iter=8):
    """
    自动调节曝光（ExposureTimeRaw）和增益（GainRaw），使图像整体平均亮度靠近 target。
    """
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
        # 清空现有缓冲区
        stream.FlushBuffers(stream.Flush_AllDiscard)
        # 创建并入队缓冲区
        bufSize = stream.GetBufferSize()
        numBuf = max(4, stream.GetMinNumBuffers())
        buffers = [stream.CreateBuffer(bufSize) for _ in range(numBuf)]
        for b in buffers:
            stream.QueueBuffer(b)

        stream.StartAcquisition()
        pars.ExecuteCommand("AcquisitionStart")
        time.sleep(0.5)  # 增加等待时间以确保相机稳定

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
        # 停止采集并清理缓冲区
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
    """
    设置快门时间（ExposureTimeRaw），使用提供的 expValue。
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