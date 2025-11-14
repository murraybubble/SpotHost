import cv2
import numpy as np
import sys
import os
import time
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QTime
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QGroupBox, QFormLayout,
                            QDialog, QSlider, QMessageBox, QSpinBox, QDialogButtonBox,
                            QTextEdit, QComboBox, QStackedWidget, QTableWidget, 
                            QTableWidgetItem, QLineEdit, QGridLayout, QButtonGroup,
                            QFileDialog, QSizePolicy, QSpacerItem)

current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在目录（Cam2）的父目录（即外层目录 spot-host）
parent_dir = os.path.dirname(os.path.dirname(current_script_path))
# 将外层目录添加到Python的搜索路径
sys.path.append(parent_dir)

from cam2_3_serialControl import CameraController_1

# 导入MainDlg.py中使用的自定义库
sys.path.append(os.path.dirname(__file__))
from CSMainDialog.spot_detection import preprocess_image_cv, detect_and_draw_spots, energy_distribution
from CSMainDialog.reconstruction3d import generate_3d_image
from CSMainDialog.parameter_calculation import calculate_ideal_divergence, calculate_actual_divergence, calculate_quality_factor
from CSMainDialog.image_cropper import CropDialog
from CSMainDialog.spot_algorithms import detect_spots

class DetailGainDialog(QDialog):
    """细节增益调节对话框"""
    def __init__(self, parent=None, current_value=0):
        super().__init__(parent)
        self.setWindowTitle("细节增益调节 (0-255)")
        self.setFixedSize(300, 150) 
        layout = QVBoxLayout(self)
        
        # 滑块调节
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(current_value)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        
        # 数值显示与输入
        self.value_spin = QSpinBox()
        self.value_spin.setRange(0, 255)
        self.value_spin.setValue(current_value)
        
        # 联动
        self.slider.valueChanged.connect(self.value_spin.setValue)
        self.value_spin.valueChanged.connect(self.slider.setValue)
        
        # 布局
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("增益值:"))
        slider_layout.addWidget(self.value_spin)
        
        layout.addLayout(slider_layout)
        layout.addWidget(self.slider)
        
        # 确认按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_value(self):
        """获取当前设置的增益值"""
        return self.value_spin.value()


class Camera2Thread(QThread):
    """相机线程（支持启动/暂停，复用资源）"""
    frame_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)
    param_signal = pyqtSignal(dict)
    
    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.running = False  # 线程是否处于运行状态（总开关）
        self.paused = False   # 是否暂停
        self.cap = None
        self.thread_tag = id(self)
        self.last_frame = None  # 保存最后一帧用于暂停显示
        print(f"[Camera2Thread] 初始化线程 (RTSP: {self.rtsp_url}, 标识: {self.thread_tag})")

    def run(self):  
        self.running = True
        print(f"[Camera2Thread] 线程开始运行 (标识: {self.thread_tag})")
        self.status_signal.emit(f"正在连接长波相机: {self.rtsp_url}")
        
        try:
            # 初始化相机资源（仅首次启动时初始化）
            if not self.cap:
                self.cap = cv2.VideoCapture(self.rtsp_url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                if hasattr(cv2, 'CAP_PROP_TIMEOUT'):
                    self.cap.set(cv2.CAP_PROP_TIMEOUT, 500)  # 缩短超时，提升响应速度
            
            if not self.cap.isOpened():
                error_msg = "无法连接长波相机（RTSP流打开失败）"
                self.status_signal.emit(error_msg)
                print(f"[Camera2Thread] 错误: {error_msg} (标识: {self.thread_tag})")
                self.running = False
                return
                
            # 首次连接时发送视频参数
            params = {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(self.cap.get(cv2.CAP_PROP_FPS), 1),
                "codec": int(self.cap.get(cv2.CAP_PROP_FOURCC))
            }
            self.param_signal.emit(params)
            print(f"[Camera2Thread] 视频参数: {params} (标识: {self.thread_tag})")
            self.status_signal.emit("长波相机连接成功")
            
            # 核心循环：支持启动/暂停切换
            while self.running:
                # 暂停状态时阻塞，不读取帧
                while self.paused and self.running:
                    self.msleep(100)  # 降低CPU占用
                    continue
                
                # 若线程已被终止，退出循环
                if not self.running:
                    break
                
                # 读取最新帧
                ret, frame = self.cap.read()
                if not ret:
                    error_msg = "长波相机读取帧失败，尝试重连..."
                    self.status_signal.emit(error_msg)
                    print(f"[Camera2Thread] 错误: {error_msg} (标识: {self.thread_tag})")
                    # 重连逻辑
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    if not self.cap.isOpened():
                        self.status_signal.emit("重连失败，视频流停止")
                        self.running = False
                        break
                    continue
                
                # 保存最后一帧并发送给UI
                self.last_frame = frame
                self.frame_signal.emit(frame)
                
        except Exception as e:
            error_msg = f"长波相机错误: {str(e)}"
            self.status_signal.emit(error_msg)
            print(f"[Camera2Thread] 异常: {error_msg} (标识: {self.thread_tag})")
        finally:
            # 线程彻底终止时释放资源
            self.running = False
            self.paused = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                print(f"[Camera2Thread] 已释放视频捕获资源 (标识: {self.thread_tag})")
            print(f"[Camera2Thread] 线程运行结束 (标识: {self.thread_tag})")

    def pause(self):
        """暂停播放（保留资源和最后一帧）"""
        if self.paused:
            print(f"[Camera2Thread] 已处于暂停状态 (标识: {self.thread_tag})")
            return
        self.paused = True
        self.status_signal.emit("视频流已暂停")
        print(f"[Camera2Thread] 线程暂停 (标识: {self.thread_tag})")

    def resume(self):
        """恢复播放（清理旧帧，获取最新画面）"""
        if not self.paused or not self.running:
            print(f"[Camera2Thread] 无法恢复（未暂停或线程未运行） (标识: {self.thread_tag})")
            return
        self.paused = False
        # 清理缓冲区旧帧，确保显示最新画面
        if self.cap:
            for _ in range(2):
                self.cap.read()
        self.status_signal.emit("视频流已恢复")
        print(f"[Camera2Thread] 线程恢复 (标识: {self.thread_tag})")

    def stop_thread(self):
        """彻底停止线程（窗口关闭时调用）"""
        print(f"[Camera2Thread] 开始彻底停止线程 (标识: {self.thread_tag})")
        self.running = False
        self.paused = False
        if self.isRunning():
            self.wait(2000)
        print(f"[Camera2Thread] 线程彻底停止 (标识: {self.thread_tag})")


class ParameterCalculationWindow(QDialog):
    def __init__(self):
        super(ParameterCalculationWindow, self).__init__()

        self.setWindowTitle('激光参数计算器')
        self.setMinimumSize(963, 760)
        self.layout = QVBoxLayout(self)  

        # 设置布局的边距（左、右、上、下），这里给左、右各添加20px的空白
        self.layout.setContentsMargins(20, 5, 20, 20)  # 设置上下左右的边距
        self.layout.setSpacing(15)  # 设置控件之间的间距

        # 添加标题
        self.title_label = QLabel("激光参数计算器")
        self.title_label.setAlignment(Qt.AlignCenter)  # 设置标题居中对齐
        self.title_label.setStyleSheet("font-size: 25px; font-weight: bold; color: #2E3A59;")
        self.layout.addWidget(self.title_label)

        # 创建一个水平布局来放图片
        top_layout = QHBoxLayout()

        # 加载图片
        top_layout = QHBoxLayout()
        self.image_label = QLabel(self)
        pixmap = QPixmap("CSMainDialog\远场光斑发散模型\远场光斑发散模型.png")
        if pixmap.isNull():
            print("图片加载失败！")
        else:
            print("图片加载成功！")
        self.image_label.setPixmap(pixmap.scaled(500, 400, aspectRatioMode=Qt.KeepAspectRatio))
        self.image_label.setStyleSheet("border: 3px solid black;")  # 设置3px粗的黑色边框

        # 强制图片左对齐
        top_layout.addWidget(self.image_label)

        # 创建表格显示区域
        self.table_widget = QTableWidget(self)
        self.table_widget.setRowCount(1)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["远-近夹角", "中-近夹角", "远-中夹角", "测试时间"])
        self.table_widget.setColumnWidth(0, 100)  
        self.table_widget.setColumnWidth(1, 100)  
        self.table_widget.setColumnWidth(2, 100)  
        self.table_widget.setColumnWidth(3, 100)  

        # 将表格添加到布局中
        top_layout.addWidget(self.table_widget)

        # 设置布局的对齐方式，图片和表格左右对齐
        top_layout.setStretch(0, 1)  # 让图片占更多的空间
        top_layout.setStretch(1, 2)  # 让表格占更多空间


        # 设置 `QHBoxLayout` 左对齐
        top_layout.setAlignment(Qt.AlignLeft)

        # 创建一个垂直布局，将标题和图片放在一起
        header_layout = QVBoxLayout()
        header_layout.addLayout(top_layout)  # 添加图片布局
        self.layout.addLayout(header_layout)

        # 创建一个网格布局
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)

        self.label1 = QLabel("请输入 波长(nm)：")
        self.label1.setStyleSheet("font-size: 16px;")  # 增大标签字体
        self.input_wavelength = QLineEdit()
        self.input_wavelength.setStyleSheet("font-size: 14px; height: 30px;")  # 设置输入框字体和高度
        grid_layout.addWidget(self.label1, 0, 0)
        grid_layout.addWidget(self.input_wavelength, 0, 1)

        self.label2 = QLabel("请输入 出射口径(mm)：")
        self.label2.setStyleSheet("font-size: 16px;")  # 增大标签字体
        self.input_aperture = QLineEdit()
        self.input_aperture.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label2, 1, 0)
        grid_layout.addWidget(self.input_aperture, 1, 1)

        self.label3 = QLabel("请输入 远场光斑直径(mm)：")
        self.label3.setStyleSheet("font-size: 16px;")  # 增大标签字体
        self.input_spot_diameter = QLineEdit()
        self.input_spot_diameter.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label3, 2, 0)
        grid_layout.addWidget(self.input_spot_diameter, 2, 1)

        self.label4 = QLabel("请输入 激光功率(W)：")
        self.label4.setStyleSheet("font-size: 16px;")  # 增大标签字体
        self.input_laser_power = QLineEdit()
        self.input_laser_power.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label4, 3, 0)
        grid_layout.addWidget(self.input_laser_power, 3, 1)

        self.label5 = QLabel("请输入 传输距离(m)：")
        self.label5.setStyleSheet("font-size: 16px;")  # 增大标签字体
        self.input_transmission_distance = QLineEdit()
        self.input_transmission_distance.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label5, 4, 0)
        grid_layout.addWidget(self.input_transmission_distance, 4, 1)

        self.label_distance = QLabel("请输入 测距机距离(m)：")
        self.label_distance.setStyleSheet("font-size: 16px;")  # 增大标签字体
        self.input_distance = QLineEdit()
        self.input_distance.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label_distance, 5, 0)
        grid_layout.addWidget(self.input_distance, 5, 1)

        # 标签和输出框 (右侧)
        self.label6 = QLabel("理想半发散角(rad)：")
        self.label6.setStyleSheet("font-size: 16px;")  # 增大标签字体
        self.output_ideal_divergence = QLineEdit()
        self.output_ideal_divergence.setReadOnly(True)  # 只读
        self.output_ideal_divergence.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label6, 0, 2)
        grid_layout.addWidget(self.output_ideal_divergence, 0, 3)

        self.label7 = QLabel("实际半发散角(rad)：")
        self.label7.setStyleSheet("font-size: 16px;")  # 增大标签字体
        self.output_actual_divergence = QLineEdit()
        self.output_actual_divergence.setReadOnly(True)  # 只读
        self.output_actual_divergence.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label7, 1, 2)
        grid_layout.addWidget(self.output_actual_divergence, 1, 3)

        self.label8 = QLabel("质量因子 M²：")
        self.label8.setStyleSheet("font-size: 16px;")  # 增大标签字体
        self.output_quality_factor = QLineEdit()
        self.output_quality_factor.setReadOnly(True)  # 只读
        self.output_quality_factor.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label8, 2, 2)
        grid_layout.addWidget(self.output_quality_factor, 2, 3)

        # 激光相互夹角显示区域
        self.label_angle_A_B = QLabel("A-B激光相互夹角：")
        self.label_angle_A_B.setStyleSheet("font-size: 16px;")
        self.output_angle_A_B = QLineEdit()
        self.output_angle_A_B.setReadOnly(True)
        self.output_angle_A_B.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label_angle_A_B, 3 ,2)
        grid_layout.addWidget(self.output_angle_A_B, 3, 3)

        self.label_angle_B_C = QLabel("B-C激光相互夹角：")
        self.label_angle_B_C.setStyleSheet("font-size: 16px;")
        self.output_angle_B_C = QLineEdit()
        self.output_angle_B_C.setReadOnly(True)
        self.output_angle_B_C.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label_angle_B_C, 4, 2)
        grid_layout.addWidget(self.output_angle_B_C, 4, 3)

        self.label_angle_C_A = QLabel("C-A激光相互夹角：")
        self.label_angle_C_A.setStyleSheet("font-size: 16px;")
        self.output_angle_C_A = QLineEdit()
        self.output_angle_C_A.setReadOnly(True)
        self.output_angle_C_A.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label_angle_C_A, 5, 2)
        grid_layout.addWidget(self.output_angle_C_A, 5, 3)

        # 通过增加一个垂直间距来确保夹角显示区域与底部计算按钮分开
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(spacer)

        # 将网格布局加入到主布局中
        self.layout.addLayout(grid_layout)

        # 提交按钮
        self.submit_button = QPushButton('计算')
        self.submit_button.setStyleSheet("font-size: 14px; height: 30px;")  # 增大按钮字体和高度
        self.submit_button.clicked.connect(self.calculate_parameters)

        self.layout.addWidget(self.submit_button)
        self.setLayout(self.layout)

        # 定时器每1秒更新一次
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_table)
        self.timer.start(1000)  # 1000ms 即 1秒


    def update_table(self):
        # 模拟激光夹角的计算和时间
        angle_A_B = 0.02  # 模拟计算值
        angle_B_C = 0.03
        angle_C_A = 0.04
        current_time = QTime.currentTime().toString('hh:mm:ss')  # 获取当前时间

        # 获取表格当前行数
        row_position = self.table_widget.rowCount()

        # 添加新行
        self.table_widget.insertRow(row_position)

        # 更新新行的数据
        self.table_widget.setItem(row_position, 0, QTableWidgetItem(f"{angle_A_B:.3e} rad"))
        self.table_widget.setItem(row_position, 1, QTableWidgetItem(f"{angle_B_C:.3e} rad"))
        self.table_widget.setItem(row_position, 2, QTableWidgetItem(f"{angle_C_A:.3e} rad"))
        self.table_widget.setItem(row_position, 3, QTableWidgetItem(current_time))

        # 滚动到表格的最后一行
        self.table_widget.scrollToBottom()
    
    #参数输入
    def calculate_parameters(self):
        try:
             # 检查所有输入框是否为空
            if not self.input_wavelength.text() or not self.input_aperture.text() or not self.input_spot_diameter.text() or not self.input_laser_power.text() or not self.input_transmission_distance.text() or not self.input_distance.text():
                QMessageBox.warning(self, "提示", "请输入数据")
                return  # 如果有任何输入框为空，停止执行
            
            wavelength = float(self.input_wavelength.text().strip())
            aperture = float(self.input_aperture.text().strip())
            spot_diameter = float(self.input_spot_diameter.text().strip())
            laser_power = float(self.input_laser_power.text().strip())
            transmission_distance = float(self.input_transmission_distance.text().strip())
            distance = float(self.input_distance.text().strip())  # 测距机距离

            if wavelength <= 0 or wavelength < 10 or wavelength > 1000:
                raise ValueError("波长应大于 0 且在 10 到 1000 纳米之间")
            if laser_power <= 0:
                raise ValueError("激光功率应大于 0")
            if spot_diameter <= 0 or spot_diameter > 100:
                raise ValueError("光斑直径应大于 0 且小于 100 毫米")
            if aperture <= 0 or aperture > 100:
                raise ValueError("出射口径应大于0 且小于100毫米")
            if transmission_distance <= 0:
                raise ValueError("传输距离应大于 0")
            if distance <= 0:
                raise ValueError("测距机距离应大于 0")

            ideal_divergence = calculate_ideal_divergence(wavelength, aperture)
            actual_divergence = calculate_actual_divergence(spot_diameter, aperture, transmission_distance)
            quality_factor = calculate_quality_factor(actual_divergence, ideal_divergence)

            self.output_ideal_divergence.setText(f"{ideal_divergence:.3e} rad")
            self.output_actual_divergence.setText(f"{actual_divergence:.3e} rad")
            self.output_quality_factor.setText(f"{quality_factor:.3e}")

        except ValueError as e:
            QMessageBox.critical(self, "输入错误", str(e))


class Camera2Widget(QWidget):
    """相机界面（包含控制按钮+串口选择+日志窗口+图像处理功能）"""
    image_signal = pyqtSignal(object)
    show3d_finished = pyqtSignal(np.ndarray)
    cropped_image_signal = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.rtsp_url = "rtsp://192.168.0.105/live.sdp"  # RTSP地址统一配置
        self.detail_gain_value = 0  # 细节增益当前值
        self.algo_type = "A"  # 算法类型
        self.last_original_image = None
        self.last_gray = None
        self.last_3d_image = None
        self.cropped_image = None

        # 初始化串口控制器
        self.controller = CameraController_1(baudrate=115200)
        
        # 初始化UI
        self.setWindowTitle("长波红外相机 - 光斑识别系统")
        self.init_ui()
        self.init_serial_connection()

        # 连接信号
        self.image_signal.connect(self._update_display)
        self.show3d_finished.connect(self._on_show3d_finished)
        self.cropped_image_signal.connect(self._process_cropped_image)

    def init_serial_connection(self):
        """初始化串口连接（自动连接，失败则提示）"""
        if self.controller.connect():
            self.update_status(f"串口连接成功", level="info")
        else:
            self.update_status(f"串口连接失败，请检查设备", level="warn")

    def init_ui(self):
        """完整UI初始化（包含所有功能）"""
        main_layout = QHBoxLayout(self)
        
        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(600)
        
        # 标题
        title_label = QLabel("长波红外相机 (RTSP)")
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 16pt;
                font-weight: bold;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin: 5px;
                text-align: center;
            }
        """)
        left_layout.addWidget(title_label)
        
        # 功能控制区域
        control_group = QGroupBox("功能控制")
        control_layout = QHBoxLayout(control_group)
        
        # 视频控制按钮
        video_control_group = QGroupBox("视频控制")
        video_control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("▶ 开始/恢复视频流")
        self.start_btn.setObjectName("func_btn")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_or_resume_camera)
        
        self.stop_btn = QPushButton("⏹ 暂停视频流")
        self.stop_btn.setObjectName("func_btn")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.clicked.connect(self.pause_camera)
        self.stop_btn.setEnabled(False)
        
        video_control_layout.addWidget(self.start_btn)
        video_control_layout.addWidget(self.stop_btn)
        video_control_group.setLayout(video_control_layout)
        left_layout.addWidget(video_control_group)
        
        # 图像处理按钮
        process_group = QGroupBox("图像处理")
        process_layout = QVBoxLayout()
        
        self.crop_btn = QPushButton("✂️ 裁切图像")
        self.crop_btn.setObjectName("control_btn")
        self.crop_btn.setMinimumHeight(30)
        self.crop_btn.clicked.connect(self.crop_image)
        
        self.show3d_btn = QPushButton("📊 显示 3D")
        self.show3d_btn.setObjectName("control_btn")
        self.show3d_btn.setMinimumHeight(30)
        self.show3d_btn.clicked.connect(self.show_3d_image)
        
        self.save_all_btn = QPushButton("💿 保存全部")
        self.save_all_btn.setObjectName("control_btn")
        self.save_all_btn.setMinimumHeight(30)
        self.save_all_btn.clicked.connect(self.save_all)
        
        self.param_calc_btn = QPushButton("📐 参数计算")
        self.param_calc_btn.setObjectName("control_btn")
        self.param_calc_btn.setMinimumHeight(30)
        self.param_calc_btn.clicked.connect(self.open_parameter_calculation_window)
        
        process_layout.addWidget(self.crop_btn)
        process_layout.addWidget(self.show3d_btn)
        process_layout.addWidget(self.save_all_btn)
        process_layout.addWidget(self.param_calc_btn)
        process_group.setLayout(process_layout)
        left_layout.addWidget(process_group)
        
        # 算法选择
        # 在 init_ui 方法中，找到算法选择的部分，替换为以下代码：

        # 算法选择
        algo_group = QGroupBox("检测算法配置")
        algo_layout = QHBoxLayout(algo_group)

        self.btn_grp = QButtonGroup(self)
        algo_buttons = [
            ("标准算法", "A"),
            ("双光斑算法", "B"), 
            ("单光斑去噪", "C"),
            ("框选识别", "D")
        ]

        for idx, (name, key) in enumerate(algo_buttons):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setObjectName("control_btn")
            btn.setMinimumHeight(30)
            self.btn_grp.addButton(btn, idx)
            algo_layout.addWidget(btn)
            if key == "A":
                btn.setChecked(True)

        self.btn_grp.buttonClicked.connect(lambda b: setattr(self, 'algo_type', b.text()[-2]))

        # 将算法组添加到左侧布局中（在图像处理组之后）
        left_layout.addWidget(algo_group)
        
        # 相机参数控制按钮区域
        camera_control_group = QGroupBox("相机控制")
        camera_control_layout = QVBoxLayout()
        
        # 第一行按钮
        hbox1 = QHBoxLayout()
        self.scene_comp_btn = QPushButton("场景补偿")
        self.scene_comp_btn.setObjectName("control_btn")
        self.scene_comp_btn.setMinimumHeight(30)
        self.scene_comp_btn.clicked.connect(self.on_scene_compensation)
        
        self.shutter_comp_btn = QPushButton("快门补偿")
        self.shutter_comp_btn.setObjectName("control_btn")
        self.shutter_comp_btn.setMinimumHeight(30)
        self.shutter_comp_btn.clicked.connect(self.on_shutter_compensation)
        
        hbox1.addWidget(self.scene_comp_btn)
        hbox1.addWidget(self.shutter_comp_btn)
        
        # 第二行按钮
        hbox2 = QHBoxLayout()
        self.tele_btn = QPushButton("远焦+")
        self.tele_btn.setObjectName("control_btn")
        self.tele_btn.setMinimumHeight(30)
        self.tele_btn.clicked.connect(self.on_tele_focus)
        
        self.wide_btn = QPushButton("近焦-")
        self.wide_btn.setObjectName("control_btn")
        self.wide_btn.setMinimumHeight(30)
        self.wide_btn.clicked.connect(self.on_wide_focus)
        
        hbox2.addWidget(self.tele_btn)
        hbox2.addWidget(self.wide_btn)
        
        # 第三行按钮
        hbox3 = QHBoxLayout()
        self.stop_focus_btn = QPushButton("调焦停")
        self.stop_focus_btn.setObjectName("control_btn")
        self.stop_focus_btn.setMinimumHeight(30)
        self.stop_focus_btn.clicked.connect(self.on_stop_focus)
        
        self.detail_gain_btn = QPushButton("细节增益")
        self.detail_gain_btn.setObjectName("control_btn")
        self.detail_gain_btn.setMinimumHeight(30)
        self.detail_gain_btn.clicked.connect(self.on_detail_gain)
        
        hbox3.addWidget(self.stop_focus_btn)
        hbox3.addWidget(self.detail_gain_btn)
        
        camera_control_layout.addLayout(hbox1)
        camera_control_layout.addLayout(hbox2)
        camera_control_layout.addLayout(hbox3)
        camera_control_group.setLayout(camera_control_layout)
        left_layout.addWidget(camera_control_group)
        
        # 串口控制区域
        serial_group = QGroupBox("串口控制")
        serial_layout = QHBoxLayout()
        
        self.serial_combo = QComboBox()
        self.serial_combo.setMinimumWidth(120)
        serial_layout.addWidget(QLabel("端口:"))
        serial_layout.addWidget(self.serial_combo)
        
        self.serial_conn_btn = QPushButton("连接")
        self.serial_conn_btn.setMinimumHeight(30)
        self.serial_conn_btn.clicked.connect(self.toggle_serial_conn)
        serial_layout.addWidget(self.serial_conn_btn)
        
        serial_group.setLayout(serial_layout)
        left_layout.addWidget(serial_group)
        
        # 状态显示
        status_group = QGroupBox("连接状态")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("准备连接长波相机...")
        self.status_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 11pt;")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)
        
        # 视频参数显示
        self.param_group = QGroupBox("视频参数")
        param_layout = QFormLayout()
        self.resolution_label = QLabel("未获取")
        self.fps_label = QLabel("未获取")
        self.codec_label = QLabel("未获取")
        param_layout.addRow("分辨率:", self.resolution_label)
        param_layout.addRow("帧率(FPS):", self.fps_label)
        param_layout.addRow("编码格式:", self.codec_label)
        self.param_group.setLayout(param_layout)
        left_layout.addWidget(self.param_group)
        
        # 日志显示
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setMaximumHeight(150)
        self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        # 填充剩余空间
        left_layout.addStretch()
        
        # 右侧图像显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 四个图像显示区域
        display_group = QGroupBox("图像显示")
        display_layout = QGridLayout(display_group)
        
        self.label1 = QLabel("原始图像")
        self.label2 = QLabel("光斑识别") 
        self.label3 = QLabel("能量分布")
        self.label4 = QLabel("3D重构")
        
        for label in [self.label1, self.label2, self.label3, self.label4]:
            label.setObjectName("image_display")
            label.setFixedSize(320, 240)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("""
                QLabel#image_display {
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    border: 2px solid #34495e;
                    border-radius: 6px;
                    font-weight: bold;
                }
            """)
        
        display_layout.addWidget(self.label1, 0, 0)
        display_layout.addWidget(self.label2, 0, 1)
        display_layout.addWidget(self.label3, 1, 0)
        display_layout.addWidget(self.label4, 1, 1)
        
        right_layout.addWidget(display_group)
        
        # 主布局组装
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # 全局样式
        self.setStyleSheet("""
            QPushButton#func_btn {
                font-size: 12pt;
                font-weight: bold;
                color: white;
                background-color: #3498db;
                border-radius: 5px;
                padding: 5px;
                margin: 5px;
            }
            QPushButton#func_btn:disabled {
                background-color: #bdc3c7;
            }
            QPushButton#control_btn, QPushButton {
                font-size: 11pt;
                font-weight: bold;
                color: white;
                background-color: #2ecc71;
                border-radius: 5px;
                padding: 5px;
                margin: 5px;
            }
            QPushButton:pressed {
                background-color: #27ae60;
            }
            QGroupBox {
                font-size: 11pt;
                font-weight: bold;
                color: #2c3e50;
                margin: 10px;
                padding: 10px;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
            }
            QTextEdit {
                font-size: 10pt;
                color: #333;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox {
                font-size: 11pt;
                padding: 3px;
                margin: 5px;
                border-radius: 3px;
            }
        """)
        
        self.setLayout(main_layout)
        self.setMinimumSize(1350, 700)
        print(f"[Camera2Widget] UI初始化完成")

    def update_status(self, message, level="info"):
        """更新状态标签和日志"""
        self.status_label.setText(message)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text_edit.append(f"[{timestamp}] {message}")
        self.log_text_edit.verticalScrollBar().setValue(
            self.log_text_edit.verticalScrollBar().maximum()
        )
        print(f"[状态更新] {message}")

    def start_or_resume_camera(self):
        """开始或恢复视频流"""
        print(f"[Camera2Widget] 点击开始/恢复按钮")
        
        if not self.camera_thread:
            self.camera_thread = Camera2Thread(self.rtsp_url)
            self.camera_thread.frame_signal.connect(self.process_frame)
            self.camera_thread.status_signal.connect(lambda msg: self.update_status(msg, "info"))
            self.camera_thread.param_signal.connect(self.update_params)
            self.camera_thread.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.update_status(f"首次启动视频流 (线程标识: {self.camera_thread.thread_tag})")
        
        elif self.camera_thread.paused and self.camera_thread.isRunning():
            self.camera_thread.resume()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.update_status(f"恢复视频流 (线程标识: {self.camera_thread.thread_tag})")
        
        else:
            self.update_status("视频流已在运行，忽略操作", level="warn")

    def pause_camera(self):
        """暂停视频流"""
        print(f"[Camera2Widget] 点击暂停按钮")
        if not self.camera_thread or not self.camera_thread.isRunning() or self.camera_thread.paused:
            return
        
        self.camera_thread.pause()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_status(f"暂停视频流 (线程标识: {self.camera_thread.thread_tag})")

    def process_frame(self, frame):
        """处理视频帧并进行图像分析"""
        try:
            # 保存原始图像
            self.last_original_image = frame.copy()
            
            # 图像处理
            gray, blur = preprocess_image_cv(frame)
            spots_output = detect_spots(frame, self.algo_type)
            heatmap = energy_distribution(gray)
            self.last_gray = gray
            
            # 发送图像信号
            self.image_signal.emit((frame, spots_output, heatmap))
            
        except Exception as e:
            error_msg = f"帧处理错误: {str(e)}"
            self.update_status(error_msg, level="error")

    def show_cv_image(self, label, img):
        """在QLabel中显示OpenCV图像"""
        if len(img.shape) == 2:
            qImg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qImg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def _update_display(self, imgs):
        """更新图像显示"""
        try:
            img_color, spots_output, heatmap = imgs
            if img_color is not None:
                self.show_cv_image(self.label1, img_color)
            if spots_output is not None:
                self.show_cv_image(self.label2, spots_output)
            if heatmap is not None:
                self.show_cv_image(self.label3, heatmap)
        except Exception as e:
            self.update_status(f"更新显示异常: {e}", level="error")

    def crop_image(self):
        """图像裁剪功能"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.update_status("请先暂停视频流才能进行图像裁切", level="warn")
            return

        if not hasattr(self, 'last_original_image') or self.last_original_image is None:
            self.update_status("没有可用的图像进行裁切", level="warn")
            return

        dialog = CropDialog(self, self.last_original_image)
        if dialog.exec_() == QDialog.Accepted:
            cropped_img = dialog.get_cropped_image()
            if cropped_img is not None:
                self.update_status("图像裁切完成，正在处理...")
                # 在新线程中处理裁切图像
                from threading import Thread
                Thread(target=self._process_cropped_image_background,
                       args=(cropped_img,), daemon=True).start()

    def _process_cropped_image_background(self, cropped_img):
        """在后台线程中处理裁切图像"""
        try:
            gray, blur = preprocess_image_cv(cropped_img)
            spots_output = detect_and_draw_spots(cropped_img, log_func=self.update_status)
            heatmap = energy_distribution(gray)
            self.cropped_image = cropped_img
            self.last_gray = gray
            self.cropped_image_signal.emit((cropped_img, spots_output, heatmap))
        except Exception as e:
            self.update_status(f"处理裁切图像时出错: {e}", level="error")

    def _process_cropped_image(self, imgs):
        """处理裁切图像结果显示"""
        try:
            cropped_img, spots_output, heatmap = imgs
            self.show_cv_image(self.label1, cropped_img)
            self.show_cv_image(self.label2, spots_output)
            self.show_cv_image(self.label3, heatmap)
            self.update_status("已更新裁切后的图像及处理结果")
        except Exception as e:
            self.update_status(f"更新裁切图像显示时出错: {e}", level="error")

    def show_3d_image(self):
        """显示3D重构图像"""
        if not hasattr(self, 'last_gray') or self.last_gray is None:
            self.update_status("没有可用图像进行3D重构", level="warn")
            return

        self.show3d_btn.setEnabled(False)
        self.update_status("开始3D重构...")

        def worker(gray):
            try:
                img3d = generate_3d_image(gray)
            except Exception as e:
                self.update_status(f"3D重构失败: {e}", level="error")
                img3d = None
            self.show3d_finished.emit(img3d)

        from threading import Thread
        t = Thread(target=worker, args=(self.last_gray.copy(),), daemon=True)
        t.start()

    def _on_show3d_finished(self, proj3d):
        """3D重构完成回调"""
        if proj3d is None:
            self.update_status("3D重构失败", level="error")
        else:
            self.last_3d_image = proj3d
            self.show_cv_image(self.label4, proj3d)
            self.update_status("3D重构完成")
        self.show3d_btn.setEnabled(True)

    def save_all(self):
        """保存所有图像和日志"""
        import cv2
        save_dir = os.path.join(os.getcwd(), "Saved_Images_Camera2")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        def save_label_image(label, name):
            pixmap = label.pixmap()
            if pixmap is None:
                self.update_status(f"⚠️ {name} 窗格为空，跳过保存。", level="warn")
                return False

            qimg = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.frombuffer(ptr, np.uint8)
            try:
                arr = arr.reshape((h, w, 3))
            except Exception as e:
                self.update_status(f"❌ 转换 {name} 图像失败: {e}", level="error")
                return False
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            file_path = os.path.join(save_dir, f"{timestamp}_{name}.jpg")
            success = cv2.imwrite(file_path, img_bgr)
            if success:
                self.update_status(f"✅ 已保存 {file_path}")
            else:
                self.update_status(f"❌ 保存 {name} 失败。", level="error")
            return success

        save_label_image(self.label1, "original")
        save_label_image(self.label2, "spots")
        save_label_image(self.label3, "heatmap")
        save_label_image(self.label4, "3d")

        log_path = os.path.join(save_dir, f"{timestamp}_spots.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(self.log_text_edit.toPlainText())

        self.update_status(f"📝 已保存日志到 {log_path}")
        self.update_status("✅ 所有保存任务完成。")

    def open_parameter_calculation_window(self):
        """打开参数计算器窗口"""
        self.parameter_calculation_window = ParameterCalculationWindow()
        self.parameter_calculation_window.show()
        self.update_status("参数计算器已打开")

    def update_params(self, params):
        """更新视频参数显示"""
        self.resolution_label.setText(f"{params['width']}x{params['height']}")
        self.fps_label.setText(f"{params['fps']}")
        codec = params['codec']
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        self.codec_label.setText(codec_str)
        self.update_status(f"参数更新：分辨率{params['width']}x{params['height']}，FPS{params['fps']}，编码{codec_str}")

    # ----------------------
    # 串口控制相关方法
    # ----------------------
    def toggle_serial_conn(self):
        """切换串口连接状态（连接/断开）"""
        if self.controller.is_connected():
            # 已连接：断开
            self.controller.disconnect()
            self.serial_conn_btn.setText("连接")
            self.update_status("串口已断开", level="info")
        else:
            # 未连接：尝试连接选中端口
            selected_port = self.serial_combo.currentText()
            if selected_port == "无可用串口":
                self.update_status("请先刷新并选择串口", level="error")
                return
            if self.controller.connect(port=selected_port):
                self.serial_conn_btn.setText("断开")
                self.update_status(f"串口[{selected_port}]连接成功", level="info")
            else:
                self.update_status(f"串口[{selected_port}]连接失败", level="error")

    # ----------------------
    # 相机控制接口
    # ----------------------
    def on_scene_compensation(self):
        if self.controller.scene_compensation():
            self.update_status("触发场景补偿（命令发送成功）")
        else:
            self.update_status("场景补偿命令发送失败", level="error")

    def on_shutter_compensation(self):
        if self.controller.shutter_compensation():
            self.update_status("触发快门补偿（命令发送成功）")
        else:
            self.update_status("快门补偿命令发送失败", level="error")

    def on_tele_focus(self):
        if self.controller.tele_focus():
            self.update_status("触发远焦+（命令发送成功）")
        else:
            self.update_status("远焦+命令发送失败", level="error")

    def on_wide_focus(self):
        if self.controller.wide_focus():
            self.update_status("触发近焦-（命令发送成功）")
        else:
            self.update_status("近焦-命令发送失败", level="error")

    def on_stop_focus(self):
        if self.controller.stop_focus():
            self.update_status("触发调焦停（命令发送成功）")
        else:
            self.update_status("调焦停命令发送失败", level="error")

    def on_detail_gain(self):
        dialog = DetailGainDialog(self, self.detail_gain_value)
        if dialog.exec_() == QDialog.Accepted:
            gain_value = dialog.get_value()
            if self.controller.set_detail_gain(gain_value):
                self.detail_gain_value = gain_value
                self.update_status(f"细节增益已设置为 {gain_value}（命令发送成功）")
            else:
                self.update_status(f"细节增益设置失败", level="error")

    def closeEvent(self, event):
        """窗口关闭时释放所有资源"""
        self.update_status("正在关闭窗口，释放资源...")
        if self.camera_thread:
            self.camera_thread.stop_thread()
            try:
                self.camera_thread.frame_signal.disconnect(self.process_frame)
                self.camera_thread.status_signal.disconnect()
                self.camera_thread.param_signal.disconnect(self.update_params)
            except:
                pass
            self.camera_thread = None
        # 断开串口连接
        if self.controller.is_connected():
            self.controller.disconnect()
        self.update_status("资源已释放，窗口关闭")
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Camera2Widget()
    window.show()
    sys.exit(app.exec_())