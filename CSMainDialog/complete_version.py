import sys
import subprocess
import time
import os
import re
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QHBoxLayout, QLineEdit, QFileDialog,QTextEdit
)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
import pyqtgraph as pg



# ----------------------------
# XDMA / 命令封装
# ----------------------------
def run_cmd(cmd):
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, encoding='gbk')
    except Exception as e:
        print(f"[ERR] 启动命令失败: {e}")
        return None
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("[ERR] 命令返回非0", stderr)
        return None
    if not stdout:
        print("[WARN] 没有输出")
        return None
    return stdout.strip()


def write_reg(addr, value):
    return run_cmd([r"xdma_rw.exe", "user", "write", f"{addr}", f"{value}"])


def read_reg(addr, length=1):
    return run_cmd([r"xdma_rw.exe", "user", "read", f"{addr}", "-l", str(length)])


# ----------------------------
# 数据解析
# ----------------------------

def convert_signed_14bit(bin_str):
    unsigned_val = int(bin_str, 2)
    if bin_str[0] == '1':
        return unsigned_val - (1 << 14)
    else:
        return unsigned_val


def read_and_process_data(length_bytes):
    txt = run_cmd([r"xdma_rw.exe", "c2h_0", "read", "0", "-l", str(length_bytes)])
    if not txt:
        return []
    channel_data = []
    for line in txt.splitlines():
        if not line.strip():
            continue
# 去掉行中所有 “0x????:” 地址段
        line_no_addr = re.sub(r'0x[0-9A-Fa-f]+:\s*', '', line)
        hex_bytes = line_no_addr.split()[:16]
        if len(hex_bytes) < 16:
            continue
        for i in range(0, 16, 4):
            group = hex_bytes[i:i+4]
            rev = group[::-1]
            merged = ''.join(rev)
            if not (merged.startswith(('F','f')) ):
                continue
            remaining_hex = merged[1:]
            bin_str = bin(int(remaining_hex, 16))[2:].zfill(28)
            a_bin = bin_str[:14]
            b_bin = bin_str[14:]
            a_val = convert_signed_14bit(a_bin)
            b_val = convert_signed_14bit(b_bin)
            channel_data.append((a_val, b_val))
    return channel_data


# ----------------------------
# 滤波
# ----------------------------

def butter_lowpass(cutoff, fs=250e6, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs=250e6, order=5):
    if data is None:
        return data
    if len(data) < (order*3):
        return data
    b, a = butter_lowpass(cutoff, fs, order)
    try:
        return filtfilt(b, a, data)
    except Exception as e:
        print(f"滤波失败，返回原始数据: {e}")
        return data

# ----------------------------
# 计算平均重频和脉宽
# ----------------------------

def calc_freq_pulse(signal, fs=250e6, threshold=None):
    import numpy as np
    from scipy.signal import medfilt

    if signal is None or len(signal) == 0:
        return 0.0, 0.0, np.array([])

    if threshold is None:
        threshold = 0.5 * (np.max(signal) - np.min(signal)) + np.min(signal)

    sig = medfilt((signal > threshold).astype(int), kernel_size=3)
    edges = np.diff(sig)
    rising = np.where(edges == 1)[0]
    falling = np.where(edges == -1)[0]

    # 对齐长度
    if len(falling) > 0 and len(rising) > 0 and falling[0] < rising[0]:
        falling = falling[1:]
    n_edges = min(len(rising), len(falling))
    rising = rising[:n_edges]
    falling = falling[:n_edges]
    mask = falling > rising
    rising = rising[mask]
    falling = falling[mask]

    if len(rising) < 2:
        return 0.0, 0.0, np.array([])

    pulse_widths = (falling - rising) / fs
    periods = np.diff(rising) / fs
    freqs = 1.0 / periods  # 每个脉冲的瞬时频率
    avg_width = np.mean(pulse_widths)
    avg_freq = np.mean(freqs)
    return avg_freq, avg_width, freqs


# 缩放系数
signed_14bit_scale = 3.5 / 16383


# ----------------------------
# PyQt GUI
# ----------------------------
class ADCWindow(QWidget):
    def __init__(self):
        super().__init__()
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adc_icon", "2.ico")
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("ADC 实时采样（A/B + 重频/脉宽）")
        self.resize(1200, 900)

        main_layout = QVBoxLayout()
        ctrl_layout = QHBoxLayout()

        # Buttons
        self.start_btn = QPushButton("开始采样")
        self.stop_btn = QPushButton("停止采样")
        self.single_btn = QPushButton("单次采样")
        self.save_btn = QPushButton("保存数据")
        self.load_btn = QPushButton("导入数据")


        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(self.single_btn)
        ctrl_layout.addWidget(self.save_btn)
        ctrl_layout.addWidget(self.load_btn)

        ctrl_layout.addWidget(QLabel("滤波截止(Hz):"))
        self.cutoff_edit = QLineEdit("100000000")
        ctrl_layout.addWidget(self.cutoff_edit)

        self.auto_btn = QPushButton("AUTO")
        # self.auto_btn.setStyleSheet("font-size:14px; font-family:Microsoft YaHei;") 
        ctrl_layout.addWidget(self.auto_btn)

        main_layout.addLayout(ctrl_layout)

        # Plots
        self.plotA = pg.PlotWidget(title="A 通道")
        self.curveA = self.plotA.plot(pen='y')
        self.plotA.showGrid(x=True, y=True)
        self.plotA.setYRange(-1.75, 1.75)
        self.plotA.setMouseEnabled(x=True, y=False)

        self.plotB = pg.PlotWidget(title="B 通道")
        self.curveB = self.plotB.plot(pen='c')
        self.plotB.showGrid(x=True, y=True)
        self.plotB.setYRange(-1.75, 1.75)
        self.plotB.setMouseEnabled(x=True, y=False)

        main_layout.addWidget(self.plotA)
        main_layout.addWidget(self.plotB)

        a_layout = QVBoxLayout()
        self.infoA = QLabel("A: 重频=0 Hz, 脉宽=0 us")
        a_layout.addWidget(self.infoA)

        self.freq_hist_plot_A = pg.PlotWidget(title="A 通道频率分布")
        self.freq_hist_plot_A.showGrid(x=True, y=True)
        self.freq_hist_plot_A.setLabel('left', '次数')
        self.freq_hist_plot_A.setLabel('bottom', '频率 (Hz)')
        self.freq_hist_curve_A = self.freq_hist_plot_A.plot(
            pen='g', stepMode=False, fillLevel=0, brush=(0,255,0,80)
        )
        a_layout.addWidget(self.freq_hist_plot_A)

        # ----------------------------
        # B 通道 info + 频率分布
        # ----------------------------
        b_layout = QVBoxLayout()
        self.infoB = QLabel("B: 重频=0 Hz, 脉宽=0 us")
        b_layout.addWidget(self.infoB)

        self.freq_hist_plot_B = pg.PlotWidget(title="B 通道频率分布")
        self.freq_hist_plot_B.showGrid(x=True, y=True)
        self.freq_hist_plot_B.setLabel('left', '次数')
        self.freq_hist_plot_B.setLabel('bottom', '频率 (Hz)')

        self.freq_hist_curve_B = self.freq_hist_plot_B.plot(
            pen='b', stepMode=False, fillLevel=0, brush=(0,0,255,80)
        )
        b_layout.addWidget(self.freq_hist_plot_B)

        # ----------------------------
        # 总水平布局，将 A/B 两个布局并排
        # ----------------------------
        freq_layout = QHBoxLayout()
        freq_layout.addLayout(a_layout)
        freq_layout.addLayout(b_layout)

        # 添加到主布局
        main_layout.addLayout(freq_layout)
        self.clear_btn = QPushButton("清空统计")
        ctrl_layout.addWidget(self.clear_btn)
        # 连接信号
        self.clear_btn.clicked.connect(self.clear_stats)


                # ---- 日志控件 ----
        main_layout.addWidget(self.infoB)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        self.log_view.setStyleSheet("background:black; color:lime; font-family:Courier; font-size:12px;")
        # 打开日志目录按钮
        self.open_log_btn = QPushButton("打开日志目录")
        self.open_log_btn.setFixedSize(140, 70)
        self.open_log_btn.setStyleSheet("font-size:14px; font-weight:bold;")
        self.log_layout = QHBoxLayout()
        self.log_layout .addWidget(self.log_view)        # 左侧日志窗口
        self.log_layout .addWidget(self.open_log_btn)    # 右侧按钮
        self.setLayout(main_layout)
        main_layout.addLayout(self.log_layout)



        self.freq_history_A = []  # 用于存储A通道每次采样频率
        self.freq_history_B = []  # B通道

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer)

        self.running = False
        self.last_dataA = None
        self.last_dataB = None
        self.last_interp_x = None
        self.last_interp_A = None
        self.last_interp_B = None
        self.last_freqA = 0.0
        self.last_freqB = 0.0
        self.last_widthA = 0.0
        self.last_widthB = 0.0

        # Connect signals
        self.start_btn.clicked.connect(self.start_acq)
        self.stop_btn.clicked.connect(self.stop_acq)
        self.single_btn.clicked.connect(self.single_acq)
        self.save_btn.clicked.connect(self.save_data)
        self.load_btn.clicked.connect(self.load_data)
        self.auto_btn.clicked.connect(self.on_auto)
        self.open_log_btn.clicked.connect(self.open_log_directory)

    # ----------------------------
    # AUTO 按钮
    # ----------------------------
    def on_auto(self):
        self.plotA.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        self.plotB.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        if hasattr(self, 'freq_hist_plot_A') and hasattr(self, 'freq_hist_plot_B'):
        # 重新绘制频率分布图（历史频率）
            self.update_freq_hist()

        if hasattr(self, 'freq_hist_plot_A'):
            self.freq_hist_plot_A.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
            self.freq_hist_plot_A.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        if hasattr(self, 'freq_hist_plot_B'):
            self.freq_hist_plot_B.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
            self.freq_hist_plot_B.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    def log(self, msg):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        line =f"{timestamp} {msg}"
        self.log_view.append(line)
        self.save_log_to_file(line)

    def save_log_to_file(self, text):
        try:
            # 日志文件夹
            log_dir = "adc_logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        
            # 日志文件名：按日期自动切分
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_filename = f"log_{date_str}.txt"

            log_path = os.path.join(log_dir, log_filename)

            # 写入文件
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

        except Exception as e:
            print("日志保存失败:", e)

    def open_log_directory(self):
        log_dir = os.path.abspath("adc_logs")
        try:
            if sys.platform.startswith("win"):
                os.startfile(log_dir)
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", log_dir])
            else:
                subprocess.Popen(["xdg-open", log_dir])
            self.log("已打开日志所在文件夹")
        except Exception as e:
            self.log(f"打开日志目录失败: {e}")
    # ----------------------------
    # 连续采样
    # ----------------------------
    def start_acq(self):
        self.running = True
        self.timer.start(int(1000/15))  # 15 FPS

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.single_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        print("连续采样开始")
        self.log("连续采样开始")

    def stop_acq(self):
        self.running = False
        self.timer.stop()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.single_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        print("连续采样停止")
        self.log("连续采样停止")

    # ----------------------------
    # 单次采样
    # ----------------------------
    def single_acq(self):
        print("执行单次采样")
        self.log("执行单次采样")
        self.timer.stop()
        self.running = False

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.single_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        self.acquire_once()

    def update_freq_hist(self):
        # ------------------ A 通道 ------------------
        if hasattr(self, 'freq_history_A') and len(self.freq_history_A) > 0:
            counts, bins = np.histogram(self.freq_history_A, bins=100)  # 增加 bins 数量
            bins_step = np.zeros(len(counts)+1)
            bins_step[:] = bins
            self.freq_hist_curve_A.setData(
                x=bins_step, y=counts, stepMode=True,
                fillLevel=0, brush=(0,255,0,80)
            )
        else:
            self.freq_hist_curve_A.clear()

        # ------------------ B 通道 ------------------
        if hasattr(self, 'freq_history_B') and len(self.freq_history_B) > 0:
            counts, bins = np.histogram(self.freq_history_B, bins=100)  # 增加 bins 数量
            bins_step = np.zeros(len(counts)+1)
            bins_step[:] = bins
            self.freq_hist_curve_B.setData(
                x=bins_step, y=counts, stepMode=True,
                fillLevel=0, brush=(0,0,255,80)
            )
        else:
            self.freq_hist_curve_B.clear()


    def clear_stats(self):
        # 清空 info 标签
        self.infoA.setText("A: 重频=0 Hz, 脉宽=0 us")
        self.infoB.setText("B: 重频=0 Hz, 脉宽=0 us")

        # 清空内部存储的频率数据
        self.freq_history_A = []
        self.freq_history_B = []

        # 清空绘图
        self.freq_hist_curve_A.clear()
        self.freq_hist_curve_B.clear()

        # 强制刷新绘图区域
        self.freq_hist_plot_A.plotItem.update()
        self.freq_hist_plot_B.plotItem.update()

        # 处理事件循环，保证界面立即更新
        QApplication.processEvents()

        self.log("已清空重频脉宽统计")

    # ----------------------------
    # 采一次数据（给连续或单次调用）
    # ----------------------------
    def acquire_once(self):
        # 向 FPGA 发起采样
        write_reg("0x0000", "1")
        data = read_and_process_data(32768)
        if not data:
            print("无数据或读取失败")
            self.log("无数据或读取失败")
            return

        arr = np.array(data)
        if arr.size == 0:
            print("解析后数据为空")
            self.log("解析后数据为空")
            return

        chA = arr[:, 0].astype(np.float64) * signed_14bit_scale
        chB = arr[:, 1].astype(np.float64) * signed_14bit_scale

        try:
            cutoff = float(self.cutoff_edit.text())
            if cutoff <= 0:
                cutoff = 1e8
        except:
            cutoff = 1e8

        chA_f = lowpass_filter(chA, cutoff)
        chB_f = lowpass_filter(chB, cutoff)

        # interp for plotting
        n = len(chA_f)
        interp_factor = 6
        x = np.arange(n)
        x2 = np.linspace(0, n-1, n * interp_factor)
        try:
            a_interp = CubicSpline(x, chA_f)(x2)
            b_interp = CubicSpline(x, chB_f)(x2)
        except Exception as e:
            print("插值失败，使用原始数据", e)
            self.log(f"插值失败: {e}")
            a_interp = chA_f
            b_interp = chB_f
            x2 = np.arange(len(a_interp))

        freqA, widthA, freqsA_array = calc_freq_pulse(chA_f)
        freqB, widthB, freqsB_array = calc_freq_pulse(chB_f)

        # 保存瞬时频率数据，用于绘制频率分布
        self.last_freqs_A_array = freqsA_array
        self.last_freqs_B_array = freqsB_array
        
        if hasattr(self, 'freq_history_A'):
            self.freq_history_A.extend(freqsA_array.tolist())  # 累积
        else:
            self.freq_history_A = freqsA_array.tolist()

        if hasattr(self, 'freq_history_B'):
            self.freq_history_B.extend(freqsB_array.tolist())
        else:
            self.freq_history_B = freqsB_array.tolist()

        # 保存到实例以便保存/导入使用
        self.last_dataA = chA_f
        self.last_dataB = chB_f
        self.last_interp_x = x2
        self.last_interp_A = a_interp
        self.last_interp_B = b_interp
        self.last_freqA = freqA
        self.last_freqB = freqB
        self.last_widthA = widthA
        self.last_widthB = widthB
        
        # update UI
        self.infoA.setText(f"A: 重频={freqA:.1f} Hz, 脉宽={widthA*1e6:.2f} us")
        self.infoB.setText(f"B: 重频={freqB:.1f} Hz, 脉宽={widthB*1e6:.2f} us")

        self.curveA.setData(self.last_interp_x, self.last_interp_A)
        self.curveB.setData(self.last_interp_x, self.last_interp_B)
        self.update_freq_hist()

    # ----------------------------
    # 连续采样回调
    # ----------------------------
    def on_timer(self):
        self.acquire_once()

    # ----------------------------
# 保存功能（NPZ）
# ----------------------------
    def save_data(self):
        if self.last_dataA is None or self.last_dataB is None:
            print("无数据可保存")
            self.log("无数据可保存")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "保存数据", "adc_capture.npz", "NumPy 压缩文件 (*.npz)")
        if not filename:
            return

        meta = {
            'fs': 250e6,
            'cutoff': float(self.cutoff_edit.text()) if self.cutoff_edit.text() else None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }

        # 保存最后一次采样的瞬时频率数组
        np.savez_compressed(filename,
                            A_raw=self.last_dataA,
                            B_raw=self.last_dataB,
                            A_interp=self.last_interp_A,
                            B_interp=self.last_interp_B,
                            interp_x=self.last_interp_x,
                            last_freqs_A=self.last_freqs_A_array,
                            last_freqs_B=self.last_freqs_B_array,
                            freqA=self.last_freqA,
                            freqB=self.last_freqB,
                            widthA=self.last_widthA,
                            widthB=self.last_widthB,
                            meta=meta)

        print(f"保存成功 -> {filename}")
        self.log(f"保存成功 -> {filename}")


    # ----------------------------
    # 导入功能
    # ----------------------------
    def load_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, '打开数据文件', '', 'NumPy 压缩文件 (*.npz)')
        if not filename:
            return
        try:
            d = np.load(filename, allow_pickle=True)
        except Exception as e:
            print("加载文件失败:", e)
            self.log(f"加载文件失败: {e}")
            return

        try:
            # 原始和插值数据
            self.last_dataA = d['A_raw']
            self.last_dataB = d['B_raw']
            self.last_interp_A = d['A_interp']
            self.last_interp_B = d['B_interp']
            self.last_interp_x = d['interp_x']

            # 最后一次采样的瞬时频率数组
            self.last_freqs_A_array = d['last_freqs_A']
            self.last_freqs_B_array = d['last_freqs_B']

            # 平均频率/脉宽
            self.last_freqA = float(d['freqA'])
            self.last_freqB = float(d['freqB'])
            self.last_widthA = float(d['widthA'])
            self.last_widthB = float(d['widthB'])

            meta = d['meta'].item() if 'meta' in d else {}

        except Exception as e:
            print("文件内容缺失或格式不对:", e)
            self.log(f"文件内容缺失或格式不对: {e}")
            return

        # 更新 info 标签
        self.infoA.setText(f"A: 重频={self.last_freqA:.1f} Hz, 脉宽={self.last_widthA*1e6:.2f} us")
        self.infoB.setText(f"B: 重频={self.last_freqB:.1f} Hz, 脉宽={self.last_widthB*1e6:.2f} us")

        # 绘制波形
        self.curveA.setData(self.last_interp_x, self.last_interp_A)
        self.curveB.setData(self.last_interp_x, self.last_interp_B)

        # 更新历史频率列表
        self.freq_history_A = self.last_freqs_A_array.tolist()
        self.freq_history_B = self.last_freqs_B_array.tolist()

        # 绘制频率分布
        self.update_freq_hist()

        print(f"已加载 -> {filename}")
        self.log(f"已加载 -> {filename}")



if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    app.setApplicationName("ADC")
    w = ADCWindow()
    w.show()
    sys.exit(app.exec_())
