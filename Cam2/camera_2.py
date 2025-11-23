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
                            QFileDialog, QSizePolicy, QSpacerItem,QFileDialog)

current_script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(parent_dir)

from cam2_3_serialControl import CameraController_1

sys.path.append(os.path.dirname(__file__))
from CSMainDialog.spot_detection import preprocess_image_cv, detect_and_draw_spots, energy_distribution
from CSMainDialog.reconstruction3d import generate_3d_image
from CSMainDialog.parameter_calculation import ParameterCalculationWindow
from CSMainDialog.image_cropper import CropDialog
from CSMainDialog.spot_algorithms import detect_spots

class DetailGainDialog(QDialog):
    """细节增益调节对话框"""
    def __init__(self, parent=None, current_value=0):
        super().__init__(parent)
        self.setWindowTitle("细节增益调节 (0-255)")
        self.setFixedSize(300, 150) 
        layout = QVBoxLayout(self)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(current_value)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        
        self.value_spin = QSpinBox()
        self.value_spin.setRange(0, 255)
        self.value_spin.setValue(current_value)
        
        self.slider.valueChanged.connect(self.value_spin.setValue)
        self.value_spin.valueChanged.connect(self.slider.setValue)
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("增益值:"))
        slider_layout.addWidget(self.value_spin)
        
        layout.addLayout(slider_layout)
        layout.addWidget(self.slider)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_value(self):
        return self.value_spin.value()


class Camera2Thread(QThread):
    """相机线程（支持启动/暂停，复用资源）"""
    frame_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)
    param_signal = pyqtSignal(dict)
    
    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.running = False
        self.paused = False
        self.cap = None
        self.thread_tag = id(self)
        self.last_frame = None
        print(f"[Camera2Thread] 初始化线程 (RTSP: {self.rtsp_url}, 标识: {self.thread_tag})")

    def run(self):  
        self.running = True
        print(f"[Camera2Thread] 线程开始运行 (标识: {self.thread_tag})")
        self.status_signal.emit(f"正在连接长波相机: {self.rtsp_url}")
        
        try:
            if not self.cap:
                self.cap = cv2.VideoCapture(self.rtsp_url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                if hasattr(cv2, 'CAP_PROP_TIMEOUT'):
                    self.cap.set(cv2.CAP_PROP_TIMEOUT, 500)
            
            if not self.cap.isOpened():
                error_msg = "无法连接长波相机（RTSP流打开失败）"
                self.status_signal.emit(error_msg)
                print(f"[Camera2Thread] 错误: {error_msg} (标识: {self.thread_tag})")
                self.running = False
                return
                
            params = {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(self.cap.get(cv2.CAP_PROP_FPS), 1),
                "codec": int(self.cap.get(cv2.CAP_PROP_FOURCC))
            }
            self.param_signal.emit(params)
            print(f"[Camera2Thread] 视频参数: {params} (标识: {self.thread_tag})")
            self.status_signal.emit("长波相机连接成功")
            
            while self.running:
                while self.paused and self.running:
                    self.msleep(100)
                    continue
                
                if not self.running:
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    error_msg = "长波相机读取帧失败，尝试重连..."
                    self.status_signal.emit(error_msg)
                    print(f"[Camera2Thread] 错误: {error_msg} (标识: {self.thread_tag})")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    if not self.cap.isOpened():
                        self.status_signal.emit("重连失败，视频流停止")
                        self.running = False
                        break
                    continue
                
                self.last_frame = frame
                self.frame_signal.emit(frame)
                
        except Exception as e:
            error_msg = f"长波相机错误: {str(e)}"
            self.status_signal.emit(error_msg)
            print(f"[Camera2Thread] 异常: {error_msg} (标识: {self.thread_tag})")
        finally:
            self.running = False
            self.paused = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                print(f"[Camera2Thread] 已释放视频捕获资源 (标识: {self.thread_tag})")
            print(f"[Camera2Thread] 线程运行结束 (标识: {self.thread_tag})")

    def pause(self):
        if self.paused:
            return
        self.paused = True
        self.status_signal.emit("视频流已暂停")
        print(f"[Camera2Thread] 线程暂停 (标识: {self.thread_tag})")

    def resume(self):
        if not self.paused or not self.running:
            return
        self.paused = False
        if self.cap:
            for _ in range(2):
                self.cap.read()
        self.status_signal.emit("视频流已恢复")
        print(f"[Camera2Thread] 线程恢复 (标识: {self.thread_tag})")

    def stop_thread(self):
        print(f"[Camera2Thread] 开始彻底停止线程 (标识: {self.thread_tag})")
        self.running = False
        self.paused = False
        if self.isRunning():
            self.wait(2000)
        print(f"[Camera2Thread] 线程彻底停止 (标识: {self.thread_tag})")

class Camera2Widget(QWidget):
    """相机界面（包含控制按钮+串口选择+日志窗口+图像处理功能）"""
    image_signal = pyqtSignal(object)
    show3d_finished = pyqtSignal(np.ndarray)
    cropped_image_signal = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.rtsp_url = "rtsp://192.168.0.105/live.sdp"
        self.detail_gain_value = 0
        self.algo_type = "A"
        self.last_original_image = None
        self.last_gray = None
        self.last_3d_image = None
        self.cropped_image = None
        
        # 录像相关变量
        self.is_recording = False
        self.video_writer = None
        self.video_filename = ""
        self.video_params = None  # 存储视频参数用于校验

        self.controller = CameraController_1(baudrate=115200)
        
        self.setWindowTitle("长波红外相机 - 光斑识别系统")
        self.init_ui()
        self.init_serial_connection()

        self.image_signal.connect(self._update_display)
        self.show3d_finished.connect(self._on_show3d_finished)
        self.cropped_image_signal.connect(self._process_cropped_image)

    #日志保存
    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text_edit.append(f"[{timestamp}] {message}")
        self.log_text_edit.verticalScrollBar().setValue(
            self.log_text_edit.verticalScrollBar().maximum()
        )
        
    def save_log(self):
        if not self.log_text_edit.toPlainText():
            QMessageBox.information(self, "提示", "日志为空，无需保存")
            return

         # 自动生成文件名
        timestamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
        default_filename = f"日志：相机2 时间：{timestamp}.txt"

    # 打开保存对话框，默认文件名已填好
        file_path, _ = QFileDialog.getSaveFileName(
        self, 
        "保存日志", 
        default_filename,        # ← 默认填写文件名
        "文本文件 (*.txt);;所有文件 (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text_edit.toPlainText())
                self.add_log(f"日志已保存至: {file_path}")
                QMessageBox.information(self, "成功", f"日志已保存至:\n{file_path}")
            except Exception as e:
                self.add_log(f"日志保存失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")

    def init_serial_connection(self):
        if self.controller.connect():
            self.update_status(f"串口连接成功", level="info")
        else:
            self.update_status(f"串口连接失败，请检查设备", level="warn")

    def init_ui(self):
        # 主布局改为垂直布局，顶部添加工具栏
        main_layout = QVBoxLayout(self)
        
        # 顶部工具栏 - 放置常用控制按钮
        top_toolbar = QWidget()
        top_toolbar.setFixedHeight(60)
        top_layout = QHBoxLayout(top_toolbar)
        
        # 视频控制按钮 - 放置在顶部
        self.start_btn = QPushButton("▶ 开始/恢复视频流")
        self.start_btn.setObjectName("func_btn")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_or_resume_camera)
        
        self.stop_btn = QPushButton("⏹ 暂停视频流")
        self.stop_btn.setObjectName("func_btn")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.clicked.connect(self.pause_camera)
        self.stop_btn.setEnabled(False)
        
        self.record_start_btn = QPushButton("⏺ 开始录像")
        self.record_start_btn.setObjectName("func_btn")
        self.record_start_btn.setMinimumHeight(40)
        self.record_start_btn.clicked.connect(self.start_recording)
        self.record_start_btn.setEnabled(False)
        
        self.record_stop_btn = QPushButton("■ 停止录像")
        self.record_stop_btn.setObjectName("func_btn")
        self.record_stop_btn.setMinimumHeight(40)
        self.record_stop_btn.clicked.connect(self.stop_recording)
        self.record_stop_btn.setEnabled(False)
        
        top_layout.addWidget(self.start_btn)
        top_layout.addWidget(self.stop_btn)
        top_layout.addWidget(self.record_start_btn)
        top_layout.addWidget(self.record_stop_btn)
        
        # 添加分隔线
        top_layout.addSpacing(20)
        
        # 图像处理按钮 - 放置在顶部
        self.crop_btn = QPushButton("✂️ 裁切图像")
        self.crop_btn.setObjectName("control_btn")
        self.crop_btn.setMinimumHeight(40)
        self.crop_btn.clicked.connect(self.crop_image)
        
        self.show3d_btn = QPushButton("📊 显示 3D")
        self.show3d_btn.setObjectName("control_btn")
        self.show3d_btn.setMinimumHeight(40)
        self.show3d_btn.clicked.connect(self.show_3d_image)
        
        self.save_all_btn = QPushButton("💿 保存全部")
        self.save_all_btn.setObjectName("control_btn")
        self.save_all_btn.setMinimumHeight(40)
        self.save_all_btn.clicked.connect(self.save_all)
        
        self.param_calc_btn = QPushButton("📐 参数计算")
        self.param_calc_btn.setObjectName("control_btn")
        self.param_calc_btn.setMinimumHeight(40)
        self.param_calc_btn.clicked.connect(self.open_parameter_calculation_window)

        self.save_log_btn = QPushButton("💾 保存日志")
        self.save_log_btn.setObjectName("control_btn")
        self.save_log_btn.setMinimumHeight(40)
        self.save_log_btn.clicked.connect(self.save_log)
        
        top_layout.addWidget(self.crop_btn)
        top_layout.addWidget(self.show3d_btn)
        top_layout.addWidget(self.save_all_btn)
        top_layout.addWidget(self.param_calc_btn)
        top_layout.addWidget(self.save_log_btn)
        
        top_layout.addStretch()
        main_layout.addWidget(top_toolbar)
        
        # 主内容区域 - 分为左侧和右侧
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        
        # 左侧面板 - 缩小宽度，使其在1080p下更合适
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)  # 从600调整为400
        left_layout = QVBoxLayout(left_panel)
        
        title_label = QLabel("长波红外相机 (RTSP)")
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 14pt;
                font-weight: bold;
                padding: 8px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin: 5px;
                text-align: center;
            }
        """)
        left_layout.addWidget(title_label)
        
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

        for idx, (text, key) in enumerate(algo_buttons):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setObjectName("func_btn")
            btn.setFixedHeight(35)
            btn.setProperty("algo_key", key)       
            self.btn_grp.addButton(btn, idx)
            algo_layout.addWidget(btn)
            if key == "A":                         
                btn.setChecked(True)

        self.btn_grp.buttonClicked.connect(
            lambda b: setattr(self, 'algo_type', b.property("algo_key"))
        )

        left_layout.addWidget(algo_group)
        
        # 相机控制
        camera_control_group = QGroupBox("相机控制")
        camera_control_layout = QVBoxLayout()
        
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
        
        # 串口控制
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
        
        # 连接状态
        status_group = QGroupBox("连接状态")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("准备连接长波相机...")
        self.status_label.setStyleSheet("color: #7f8c8d; padding: 5px; font-size: 10pt;")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)
        
        # 视频参数
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
        
        # 系统日志 - 调整高度，使其不占用过多空间
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setMaximumHeight(120)  # 从150调整为120
        self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        left_layout.addStretch()
        
        # 右侧面板 - 图像显示区域，尽可能大
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        display_group = QGroupBox("图像显示 (640x512)")
        display_layout = QGridLayout(display_group)
        display_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 图像标签 - 设置合理的最小尺寸，保持640x512比例
        self.label1 = QLabel("原始图像")
        self.label2 = QLabel("光斑识别") 
        self.label3 = QLabel("能量分布")
        self.label4 = QLabel("3D重构")
        
        # 计算640x512的宽高比 (1.25)，设置合适的最小尺寸
        min_width = 320
        min_height = 256  # 保持640x512的比例
        
        for label in [self.label1, self.label2, self.label3, self.label4]:
            label.setObjectName("image_display")
            label.setMinimumSize(min_width, min_height)
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
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        display_layout.addWidget(self.label1, 0, 0)
        display_layout.addWidget(self.label2, 0, 1)
        display_layout.addWidget(self.label3, 1, 0)
        display_layout.addWidget(self.label4, 1, 1)
        
        # 设置网格布局的拉伸因子，使图像区域尽可能大
        display_layout.setRowStretch(0, 1)
        display_layout.setRowStretch(1, 1)
        display_layout.setColumnStretch(0, 1)
        display_layout.setColumnStretch(1, 1)
        
        right_layout.addWidget(display_group)
        right_layout.setStretch(0, 1)  # 让显示区域拉伸填充空间
        
        content_layout.addWidget(left_panel)
        content_layout.addWidget(right_panel, 1)  # 右侧权重更高，获得更多空间
        
        main_layout.addWidget(content_widget, 1)  # 内容区域权重更高
        
        # 调整样式表，使按钮在较小空间内仍清晰可见
        self.setStyleSheet("""
            QPushButton#func_btn {
                font-size: 10pt;
                font-weight: bold;
                color: white;
                background-color: #3498db;
                border-radius: 5px;
                padding: 5px;
                margin: 3px;
            }
            QPushButton#func_btn:disabled {
                background-color: #bdc3c7;
            }
            QPushButton#func_btn:checked {
                background-color: #e74c3c;
            }
            QPushButton#control_btn, QPushButton {
                font-size: 10pt;
                font-weight: bold;
                color: white;
                background-color: #2ecc71;
                border-radius: 5px;
                padding: 3px;
                margin: 3px;
            }
            QPushButton:pressed {
                background-color: #27ae60;
            }
            QGroupBox {
                font-size: 10pt;
                font-weight: bold;
                color: #2c3e50;
                margin: 8px;
                padding: 8px;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
            }
            QTextEdit {
                font-size: 9pt;
                color: #333;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 3px;
            }
            QComboBox {
                font-size: 10pt;
                padding: 2px;
                margin: 3px;
                border-radius: 3px;
            }
        """)
        
        self.setLayout(main_layout)
        self.setMinimumSize(1200, 700)  # 调整最小尺寸，适合1080p
        print(f"[Camera2Widget] UI初始化完成")

    def update_status(self, message, level="info"):
        self.status_label.setText(message)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text_edit.append(f"[{timestamp}] {message}")
        self.log_text_edit.verticalScrollBar().setValue(
        self.log_text_edit.verticalScrollBar().maximum()
        )
        print(f"[状态更新] {message}")

    def start_or_resume_camera(self):
        print(f"[Camera2Widget] 点击开始/恢复按钮")
        
        if not self.camera_thread:
            self.camera_thread = Camera2Thread(self.rtsp_url)
            self.camera_thread.frame_signal.connect(self.process_frame)
            self.camera_thread.status_signal.connect(lambda msg: self.update_status(msg, "info"))
            self.camera_thread.param_signal.connect(self.update_params)
            self.camera_thread.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.record_start_btn.setEnabled(True)
            self.update_status(f"首次启动视频流 (线程标识: {self.camera_thread.thread_tag})")
        
        elif self.camera_thread.paused and self.camera_thread.isRunning():
            self.camera_thread.resume()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.record_start_btn.setEnabled(True)
            self.update_status(f"恢复视频流 (线程标识: {self.camera_thread.thread_tag})")
        
        else:
            self.update_status("视频流已在运行，忽略操作", level="warn")

    def pause_camera(self):
        print(f"[Camera2Widget] 点击暂停按钮")
        if not self.camera_thread or not self.camera_thread.isRunning() or self.camera_thread.paused:
            return
        
        if self.is_recording:
            self.stop_recording()
            
        self.camera_thread.pause()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.record_start_btn.setEnabled(False)
        self.update_status(f"暂停视频流 (线程标识: {self.camera_thread.thread_tag})")

    def start_recording(self):
        if not self.camera_thread or not self.camera_thread.isRunning() or self.camera_thread.paused:
            QMessageBox.warning(self, "警告", "请先启动视频流再开始录像")
            return
            
        if self.is_recording:
            QMessageBox.information(self, "提示", "已经在录像中")
            return
            
        if not self.video_params:
            QMessageBox.warning(self, "警告", "未获取到视频参数，无法录像")
            return
            
        try:
            current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.video_filename = f"./Saved_Files/Cam2/Cam2_recording_{current_time}.mp4"
            
            # 使用预存的视频参数（已校验）
            width = self.video_params["width"]
            height = self.video_params["height"]
            fps = self.video_params["fps"]
            
            # 再次校验参数范围
            if width <= 0 or width > 4096 or height <=0 or height > 2160:
                raise ValueError(f"无效的视频尺寸: {width}x{height}")
            if fps <= 0 or fps > 60:
                raise ValueError(f"无效的帧率: {fps}")
            
            # 强制转换为C语言兼容的整数类型
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_filename, 
                fourcc, 
                float(fps),  # 确保帧率为浮点数
                (int(width), int(height))  # 确保宽高为整数
            )
            
            if not self.video_writer.isOpened():
                raise Exception("无法创建视频文件写入器")
                
            self.is_recording = True
            self.record_start_btn.setEnabled(False)
            self.record_stop_btn.setEnabled(True)
            self.update_status(f"开始录像，文件将保存为: {self.video_filename}")
            
        except Exception as e:
            self.update_status(f"录像启动失败: {str(e)}", level="error")
            QMessageBox.critical(self, "错误", f"录像启动失败: {str(e)}")

    def stop_recording(self):
        if not self.is_recording or not self.video_writer:
            return
            
        try:
            self.is_recording = False
            self.video_writer.release()
            self.video_writer = None
            self.record_start_btn.setEnabled(True)
            self.record_stop_btn.setEnabled(False)
            self.update_status(f"录像已停止，文件已保存: {self.video_filename}")
            
        except Exception as e:
            self.update_status(f"录像停止失败: {str(e)}", level="error")
            QMessageBox.critical(self, "错误", f"录像停止失败: {str(e)}")

    def process_frame(self, frame):
        try:
            # 校验帧尺寸是否合法
            if frame is None or frame.size == 0:
                raise ValueError("空帧，无法处理")
                
            height, width = frame.shape[:2]
            # 录像时写入帧
            if self.is_recording and self.video_writer:
                # 确保帧尺寸与录像参数一致
                if (frame.shape[1], frame.shape[0]) != (self.video_params["width"], self.video_params["height"]):
                    frame = cv2.resize(frame, (self.video_params["width"], self.video_params["height"]))
                self.video_writer.write(frame)
                
            self.last_original_image = frame.copy()
            
            # 图像处理（增加异常处理）
            gray, blur = preprocess_image_cv(frame)
            spots_output = detect_spots(frame, self.algo_type)
            heatmap = energy_distribution(gray)
            self.last_gray = gray
            
            self.image_signal.emit((frame, spots_output, heatmap))
            
        except Exception as e:
            error_msg = f"帧处理错误: {str(e)}"
            self.update_status(error_msg, level="error")

    def show_cv_image(self, label, img):
        try:  
            # 确保图像尺寸合法，保持640x512的比例
            height, width = img.shape[:2]
            # 计算图像的宽高比
            img_ratio = width / height
            # 计算标签的宽高比
            label_ratio = label.width() / label.height()
            
            # 根据比例决定缩放方式，保持图像比例
            if img_ratio > label_ratio:
                # 图像更宽，按宽度缩放
                new_width = label.width()
                new_height = int(new_width / img_ratio)
            else:
                # 图像更高，按高度缩放
                new_height = label.height()
                new_width = int(new_height * img_ratio)
                
            img = cv2.resize(img, (new_width, new_height))
            height, width = img.shape[:2]
            
            if len(img.shape) == 2:
                bytes_per_line = width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb_img.shape
                bytes_per_line = channels * width
                q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                label.width(), label.height(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.update_status(f"图像显示错误: {str(e)}", level="error")

    def _update_display(self, images):
        frame, spots_output, heatmap = images
        self.show_cv_image(self.label1, frame)
        self.show_cv_image(self.label2, spots_output)
        self.show_cv_image(self.label3, heatmap)
        
        if self.last_3d_image is not None:
            self.show_cv_image(self.label4, self.last_3d_image)

    def _on_show3d_finished(self, image_3d):
        self.last_3d_image = image_3d
        self.show_cv_image(self.label4, image_3d)

    def _process_cropped_image(self, cropped_img):
        self.cropped_image = cropped_img
        if cropped_img is not None:
            gray, blur = preprocess_image_cv(cropped_img)
            spots_output = detect_spots(cropped_img, self.algo_type)
            heatmap = energy_distribution(gray)
            self.show_cv_image(self.label1, cropped_img)
            self.show_cv_image(self.label2, spots_output)
            self.show_cv_image(self.label3, heatmap)

    def crop_image(self):
        if self.last_original_image is None:
            QMessageBox.warning(self, "警告", "没有可裁切的图像，请先获取视频帧")
            return
            
        dialog = CropDialog(self, self.last_original_image)
        if dialog.exec_():
            cropped_img = dialog.get_cropped_image()
            self.cropped_image_signal.emit(cropped_img)
            self.update_status("图像裁切完成")

    def show_3d_image(self):
        if self.last_gray is None:
            QMessageBox.warning(self, "警告", "没有可处理的图像，请先获取视频帧")
            return
            
        self.update_status("正在生成3D图像...")
        class Generate3DThread(QThread):
            finished = pyqtSignal(np.ndarray)
            
            def __init__(self, gray_img):
                super().__init__()
                self.gray_img = gray_img
                
            def run(self):
                try:
                    image_3d = generate_3d_image(self.gray_img)
                    self.finished.emit(image_3d)
                except Exception as e:
                    print(f"生成3D图像错误: {str(e)}")
                    self.finished.emit(None)
        
        self.gen_3d_thread = Generate3DThread(self.last_gray)
        self.gen_3d_thread.finished.connect(self.show3d_finished)
        self.gen_3d_thread.start()

    def save_all(self):
        if self.last_original_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像，请先获取视频帧")
            return

        try:
            # 创建保存目录
            save_dir = "./Saved_Files/Cam2"
            os.makedirs(save_dir, exist_ok=True)

            # 生成时间戳
            current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

            # === 1. 保存原图 ===
            orig_filename = f"{save_dir}/original_{current_time}.png"
            if not cv2.imwrite(orig_filename, self.last_original_image):
                raise IOError("原始图像写入失败")

            # === 2. 保存斑点检测图 ===
            gray, blur = preprocess_image_cv(self.last_original_image)
            spots_output = detect_spots(self.last_original_image, self.algo_type)
            spots_filename = f"{save_dir}/spots_{current_time}.png"
            if not cv2.imwrite(spots_filename, spots_output):
                raise IOError("斑点检测图保存失败")

            # === 3. 保存热力图 ===
            heatmap = energy_distribution(gray)
            heat_filename = f"{save_dir}/heatmap_{current_time}.png"
            if not cv2.imwrite(heat_filename, heatmap):
                raise IOError("热力图保存失败")

            # === 4. 可选：保存 3D 图 ===
            if self.last_3d_image is not None:
                d3_filename = f"{save_dir}/3d_{current_time}.png"
                if not cv2.imwrite(d3_filename, self.last_3d_image):
                    raise IOError("3D 图保存失败")
            else:
                d3_filename = "（无 3D 图）"

            # 状态更新
            self.update_status(
                f"保存完成:\n原图: {orig_filename}\n斑点图: {spots_filename}\n热力图: {heat_filename}\n3D 图: {d3_filename}"
            )
            QMessageBox.information(self, "成功", "所有图像保存完成")

        except Exception as e:
            error_msg = f"图像保存失败: {str(e)}"
            self.update_status(error_msg, level="error")
            QMessageBox.critical(self, "错误", error_msg)


    def open_parameter_calculation_window(self):
        self.param_window = ParameterCalculationWindow()
        self.param_window.show()

    def on_scene_compensation(self):
        try:
            self.controller.scene_compensation()
            self.update_status("已发送场景补偿命令")
        except Exception as e:
            self.update_status(f"发送场景补偿命令失败: {str(e)}", level="error")

    def on_shutter_compensation(self):
        try:
            self.controller.shutter_compensation()
            self.update_status("已发送快门补偿命令")
        except Exception as e:
            self.update_status(f"发送快门补偿命令失败: {str(e)}", level="error")

    def on_tele_focus(self):
        try:
            self.controller.tele_focus()
            self.update_status("已发送远焦调节命令")
        except Exception as e:
            self.update_status(f"发送远焦调节命令失败: {str(e)}", level="error")

    def on_wide_focus(self):
        try:
            self.controller.wide_focus()
            self.update_status("已发送近焦调节命令")
        except Exception as e:
            self.update_status(f"发送近焦调节命令失败: {str(e)}", level="error")

    def on_stop_focus(self):
        try:
            self.controller.stop_focus()
            self.update_status("已发送停止调焦命令")
        except Exception as e:
            self.update_status(f"发送停止调焦命令失败: {str(e)}", level="error")

    def on_detail_gain(self):
        dialog = DetailGainDialog(self, self.detail_gain_value)
        if dialog.exec_() == QDialog.Accepted:
            gain_value = dialog.get_value()
            if self.controller.set_detail_gain(gain_value):
                self.detail_gain_value = gain_value
                self.update_status(f"细节增益已设置为 {gain_value}（命令发送成功）")
            else:
                self.update_status(f"细节增益设置失败", level="error")

    def toggle_serial_conn(self):
        if self.controller.is_connected():
            try:
                self.controller.disconnect()
                self.serial_conn_btn.setText("连接")
                self.update_status("串口已断开")
            except Exception as e:
                self.update_status(f"断开串口失败: {str(e)}", level="error")
        else:
            try:
                port = self.serial_combo.currentText()
                if port:
                    self.controller.port = port
                    if self.controller.connect():
                        self.serial_conn_btn.setText("断开")
                        self.update_status(f"串口 {port} 连接成功")
                    else:
                        self.update_status(f"串口 {port} 连接失败", level="error")
                else:
                    self.update_status("请先选择串口端口", level="warn")
            except Exception as e:
                self.update_status(f"连接串口失败: {str(e)}", level="error")

    def update_params(self, params):
        """保存并显示视频参数，增加校验"""
        self.video_params = params  # 保存参数用于录像
        self.resolution_label.setText(f"{params['width']}x{params['height']}")
        self.fps_label.setText(f"{params['fps']}")
        codec = params['codec']
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        self.codec_label.setText(codec_str)

    def closeEvent(self, event):
        if self.is_recording:
            self.stop_recording()
            
        if self.camera_thread:
            self.camera_thread.stop_thread()
            
        if self.controller.is_connected():
            self.controller.disconnect()
            
        event.accept()