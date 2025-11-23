import cv2
import os
import numpy as np
import sys
import time
import serial
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QGroupBox, QFormLayout,
                            QDialog, QSlider, QMessageBox, QSpinBox, QDialogButtonBox,
                            QTextEdit, QComboBox, QStackedWidget, QTableWidget, 
                            QTableWidgetItem, QLineEdit, QGridLayout, QButtonGroup,
                         QSpacerItem, QRadioButton, QScrollArea,QFileDialog)


# 添加maindlg的系统路径
current_script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(parent_dir)

sys.path.append(os.path.dirname(__file__))

#导入自己写的包
from cam2_3_serialControl import CameraController_2  # 导入相机控制类
from CSMainDialog.spot_detection import preprocess_image_cv, detect_and_draw_spots, energy_distribution
from CSMainDialog.reconstruction3d import generate_3d_image
from CSMainDialog.parameter_calculation import ParameterCalculationWindow
from CSMainDialog.image_cropper import CropDialog
from CSMainDialog.spot_algorithms import detect_spots


class Camera3Thread(QThread):
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
        print(f"[Camera3Thread] 初始化线程 (RTSP: {self.rtsp_url}, 标识: {self.thread_tag})")

    def run(self):  
        self.running = True
        print(f"[Camera3Thread] 线程开始运行 (标识: {self.thread_tag})")
        self.status_signal.emit(f"正在连接中波相机: {self.rtsp_url}")
        
        try:
            # 初始化相机资源（仅首次启动时初始化）
            if not self.cap:
                self.cap = cv2.VideoCapture(self.rtsp_url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                if hasattr(cv2, 'CAP_PROP_TIMEOUT'):
                    self.cap.set(cv2.CAP_PROP_TIMEOUT, 500)  # 缩短超时，提升响应速度
            
            if not self.cap.isOpened():
                error_msg = "无法连接中波相机（RTSP流打开失败）"
                self.status_signal.emit(error_msg)
                print(f"[Camera3Thread] 错误: {error_msg} (标识: {self.thread_tag})")
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
            print(f"[Camera3Thread] 视频参数: {params} (标识: {self.thread_tag})")
            self.status_signal.emit("中波相机连接成功")
            
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
                    error_msg = "中波相机读取帧失败，尝试重连..."
                    self.status_signal.emit(error_msg)
                    print(f"[Camera3Thread] 错误: {error_msg} (标识: {self.thread_tag})")
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
            error_msg = f"中波相机错误: {str(e)}"
            self.status_signal.emit(error_msg)
            print(f"[Camera3Thread] 异常: {error_msg} (标识: {self.thread_tag})")
        finally:
            # 线程彻底终止时释放资源
            self.running = False
            self.paused = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                print(f"[Camera3Thread] 已释放视频捕获资源 (标识: {self.thread_tag})")
            print(f"[Camera3Thread] 线程运行结束 (标识: {self.thread_tag})")

    def pause(self):
        """暂停播放（保留资源和最后一帧）"""
        if self.paused:
            print(f"[Camera3Thread] 已处于暂停状态 (标识: {self.thread_tag})")
            return
        self.paused = True
        self.status_signal.emit("视频流已暂停")
        print(f"[Camera3Thread] 线程暂停 (标识: {self.thread_tag})")

    def resume(self):
        """恢复播放（清理旧帧，获取最新画面）"""
        if not self.paused or not self.running:
            print(f"[Camera3Thread] 无法恢复（未暂停或线程未运行） (标识: {self.thread_tag})")
            return
        self.paused = False
        # 清理缓冲区旧帧，确保显示最新画面
        if self.cap:
            for _ in range(2):
                self.cap.read()
        self.status_signal.emit("视频流已恢复")
        print(f"[Camera3Thread] 线程恢复 (标识: {self.thread_tag})")

    def stop_thread(self):
        """彻底停止线程（窗口关闭时调用）"""
        print(f"[Camera3Thread] 开始彻底停止线程 (标识: {self.thread_tag})")
        self.running = False
        self.paused = False
        if self.isRunning():
            self.wait(2000)
        print(f"[Camera3Thread] 线程彻底停止 (标识: {self.thread_tag})")


class ImageProcessingThread(QThread):
    """图像处理线程，独立于UI线程"""
    processed_signal = pyqtSignal(tuple)  # (原始帧, 光斑识别结果, 能量分布)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.current_frame = None
        self.algo_type = "A"
        self.lock = False  # 用于帧丢弃机制的锁
        
    def set_frame(self, frame):
        """设置当前要处理的帧，如果正在处理则丢弃旧帧"""
        if self.lock:
            return  # 正在处理，丢弃当前帧
        self.current_frame = frame
        
    def set_algo_type(self, algo_type):
        """设置算法类型"""
        self.algo_type = algo_type
        
    def run(self):
        while self.running:
            if self.current_frame is not None and not self.lock:
                self.lock = True  # 标记正在处理
                try:
                    frame = self.current_frame
                    self.current_frame = None  # 处理后清空，准备接收新帧
                    
                    # 处理帧
                    gray, blur = preprocess_image_cv(frame)
                    spots_output = detect_spots(frame, self.algo_type)
                    heatmap = energy_distribution(gray)
                    
                    # 发送处理结果
                    self.processed_signal.emit((frame, spots_output, heatmap))
                except Exception as e:
                    print(f"图像处理错误: {str(e)}")
                finally:
                    self.lock = False  # 处理完成，解锁
            else:
                self.msleep(10)  # 短暂休眠，降低CPU占用
                
    def stop(self):
        self.running = False
        self.wait()


class Camera3Widget(QWidget):
    # 添加相机界面
    image_signal = pyqtSignal(object)
    show3d_finished = pyqtSignal(np.ndarray)
    cropped_image_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.rtsp_url = "rtsp://192.168.0.106/live.sdp"  # RTSP地址统一配置
        self.controller = CameraController_2()  # 创建相机控制器实例
        self.setWindowTitle("RTSP视频流监控与相机控制")
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

        # 创建图像处理线程
        self.processing_thread = ImageProcessingThread()
        self.processing_thread.processed_signal.connect(self._on_processed)
        self.processing_thread.start()

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
        default_filename = f"日志：相机3 时间：{timestamp}.txt"

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
            self.update_status(f"串口连接成功")
        else:
            self.update_status(f"串口连接失败，请检查设备")

    def init_ui(self):
        """完整UI初始化（优化1080p显示效果）"""
        main_layout = QVBoxLayout(self)
        
        # 顶部工具栏 - 放置核心控制按钮
        top_toolbar = QWidget()
        top_layout = QHBoxLayout(top_toolbar)
        top_toolbar.setFixedHeight(70)
        
        # 视频控制按钮（顶部）
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
        
        # 图像处理按钮（顶部）
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

        # 算法选择（顶部）
        algo_label = QLabel("检测算法:")
        algo_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        top_layout.addWidget(algo_label)
        
        self.btn_grp = QButtonGroup(self)
        algo_buttons = [("标准", "A"), ("双光斑", "B"), ("单光斑去噪", "C"), ("框选识别", "D")]
        for text, key in algo_buttons:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setObjectName("func_btn")
            btn.setMinimumHeight(40)
            btn.setMinimumWidth(80)
            btn.setProperty("algo_key", key)
            self.btn_grp.addButton(btn)
            top_layout.addWidget(btn)
            if key == "A":
                btn.setChecked(True)
        
        self.btn_grp.buttonClicked.connect(
            lambda b: self._on_algo_changed(b.property("algo_key"))
        )
        
        top_layout.addStretch()
        main_layout.addWidget(top_toolbar)
        
        # 主内容区域
        content_layout = QHBoxLayout()
        
        # 左侧控制面板（使用滚动区域，宽度减小以适应1080p）
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(350)  # 1080p下更窄的控制面板
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 标题
        title_label = QLabel("中波红外相机 (RTSP)")
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
        
        # 串口连接控制
        serial_group = QGroupBox("串口连接")
        serial_layout = QHBoxLayout()
        
        self.connect_serial_btn = QPushButton("🔌 连接串口")
        self.connect_serial_btn.setObjectName("func_btn")
        self.connect_serial_btn.setMinimumHeight(30)
        self.connect_serial_btn.clicked.connect(self.connect_serial)
        
        self.disconnect_serial_btn = QPushButton("🔌 断开串口")
        self.disconnect_serial_btn.setObjectName("func_btn")
        self.disconnect_serial_btn.setMinimumHeight(30)
        self.disconnect_serial_btn.clicked.connect(self.disconnect_serial)
        self.disconnect_serial_btn.setEnabled(False)
        
        serial_layout.addWidget(self.connect_serial_btn)
        serial_layout.addWidget(self.disconnect_serial_btn)
        serial_group.setLayout(serial_layout)
        left_layout.addWidget(serial_group)
        
        # 调焦控制
        focus_group = QGroupBox("调焦控制")
        focus_layout = QHBoxLayout()
        
        self.tele_focus_btn = QPushButton("调焦+")
        self.tele_focus_btn.setObjectName("func_btn")
        self.tele_focus_btn.setMinimumHeight(30)
        self.tele_focus_btn.clicked.connect(self.tele_focus)
        self.tele_focus_btn.setEnabled(False)
        
        self.wide_focus_btn = QPushButton("调焦-")
        self.wide_focus_btn.setObjectName("func_btn")
        self.wide_focus_btn.setMinimumHeight(30)
        self.wide_focus_btn.clicked.connect(self.wide_focus)
        self.wide_focus_btn.setEnabled(False)
        
        self.stop_focus_btn = QPushButton("调焦停")
        self.stop_focus_btn.setObjectName("func_btn")
        self.stop_focus_btn.setMinimumHeight(30)
        self.stop_focus_btn.clicked.connect(self.stop_focus)
        self.stop_focus_btn.setEnabled(False)
        
        focus_layout.addWidget(self.tele_focus_btn)
        focus_layout.addWidget(self.wide_focus_btn)
        focus_layout.addWidget(self.stop_focus_btn)
        focus_group.setLayout(focus_layout)
        left_layout.addWidget(focus_group)
        
        # 场景补偿
        scene_group = QGroupBox("场景控制")
        scene_layout = QVBoxLayout()
        
        self.scene_compensation_btn = QPushButton("场景补偿")
        self.scene_compensation_btn.setObjectName("func_btn")
        self.scene_compensation_btn.setMinimumHeight(30)
        self.scene_compensation_btn.clicked.connect(self.scene_compensation)
        self.scene_compensation_btn.setEnabled(False)
        
        scene_layout.addWidget(self.scene_compensation_btn)
        scene_group.setLayout(scene_layout)
        left_layout.addWidget(scene_group)
        
        # 电子放大控制
        zoom_group = QGroupBox("电子放大")
        zoom_layout = QHBoxLayout()
        
        self.zoom_group = QButtonGroup(self)
        self.zoom_1x_btn = QRadioButton("1倍")
        self.zoom_2x_btn = QRadioButton("2倍")
        self.zoom_4x_btn = QRadioButton("4倍")
        self.zoom_1x_btn.setChecked(True)
        
        self.zoom_group.addButton(self.zoom_1x_btn, 0)
        self.zoom_group.addButton(self.zoom_2x_btn, 1)
        self.zoom_group.addButton(self.zoom_4x_btn, 2)
        self.zoom_group.buttonClicked.connect(self.set_zoom)
        
        zoom_layout.addWidget(self.zoom_1x_btn)
        zoom_layout.addWidget(self.zoom_2x_btn)
        zoom_layout.addWidget(self.zoom_4x_btn)
        zoom_group.setLayout(zoom_layout)
        left_layout.addWidget(zoom_group)
        
        # 积分时间控制
        integration_group = QGroupBox("积分时间 (ms)")
        integration_layout = QHBoxLayout()
        
        self.integration_input = QLineEdit()
        self.integration_input.setPlaceholderText("输入积分时间")
        self.set_integration_btn = QPushButton("设置")
        self.set_integration_btn.setObjectName("func_btn")
        self.set_integration_btn.clicked.connect(self.set_integration_time)
        self.set_integration_btn.setEnabled(False)
        
        integration_layout.addWidget(self.integration_input)
        integration_layout.addWidget(self.set_integration_btn)
        integration_group.setLayout(integration_layout)
        left_layout.addWidget(integration_group)
        
        # 帧频控制
        fps_group = QGroupBox("帧频 (Hz)")
        fps_layout = QHBoxLayout()
        
        self.fps_input = QLineEdit()
        self.fps_input.setPlaceholderText("输入帧频")
        self.set_fps_btn = QPushButton("设置")
        self.set_fps_btn.setObjectName("func_btn")
        self.set_fps_btn.clicked.connect(self.set_frame_rate)
        self.set_fps_btn.setEnabled(False)
        
        fps_layout.addWidget(self.fps_input)
        fps_layout.addWidget(self.set_fps_btn)
        fps_group.setLayout(fps_layout)
        left_layout.addWidget(fps_group)
        
        # 状态显示
        status_group = QGroupBox("连接状态")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("准备连接中波相机...")
        self.status_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
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
    
    
      # 系统日志 - 调整高度，使其不占用过多空间
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setMaximumHeight(120)  # 从150调整为120
        self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        # 填充剩余空间
        left_layout.addStretch()
        
        left_scroll.setWidget(left_panel)
        content_layout.addWidget(left_scroll)
        
        # 右侧视频显示区域（扩大以适应1080p）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        display_group = QGroupBox("图像显示")
        display_layout = QGridLayout(display_group)
        
        self.label1 = QLabel("原始图像")
        self.label2 = QLabel("光斑识别") 
        self.label3 = QLabel("能量分布")
        self.label4 = QLabel("3D重构")
        
        for label in [self.label1, self.label2, self.label3, self.label4]:
            label.setObjectName("image_display")
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
        
        # 设置网格布局比例，使图像区域尽可能大
        display_layout.setRowStretch(0, 1)
        display_layout.setRowStretch(1, 1)
        display_layout.setColumnStretch(0, 1)
        display_layout.setColumnStretch(1, 1)
        
        right_layout.addWidget(display_group)
        content_layout.addWidget(right_panel, 1)  # 权重1，让显示区域尽可能大
        
        main_layout.addLayout(content_layout, 1)  # 权重1，让内容区域占据主要空间
        
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
                padding: 5px;
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
                font-size: 10pt;
                color: #333;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox {
                font-size: 10pt;
                padding: 3px;
                margin: 3px;
                border-radius: 3px;
            }
            QLineEdit {
                padding: 5px;
                margin: 3px;
                font-size: 10pt;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
            QRadioButton {
                margin: 3px;
                padding: 3px;
                font-size: 10pt;
            }
        """)
        
        self.setMinimumSize(1280, 720)  # 适合1080p显示器的最小尺寸
        print(f"[Camera3Widget] UI初始化完成")

    # def update_status(self, message):
    #     """更新状态信息"""
    #     self.status_label.setText(message)
    #     print(f"[状态更新] {message}")
    #日志系统控件
    def update_status(self, message, level="info"):
        self.status_label.setText(message)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text_edit.append(f"[{timestamp}] {message}")
        self.log_text_edit.verticalScrollBar().setValue(
        self.log_text_edit.verticalScrollBar().maximum()
        )
        print(f"[状态更新] {message}")


    def start_or_resume_camera(self):
        """开始或恢复视频流（统一处理）"""
        print(f"[Camera3Widget] 点击开始/恢复按钮")


        # 情况1：线程未创建（首次启动）
        if not self.camera_thread:
            self.camera_thread = Camera3Thread(self.rtsp_url)
            self.camera_thread.frame_signal.connect(self.update_frame)
            self.camera_thread.status_signal.connect(self.update_status)
            self.camera_thread.param_signal.connect(self.update_params)
            self.camera_thread.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.record_start_btn.setEnabled(True)  # 启动后允许录像
            #日志1
            self.update_status(f"首次启动视频流 (线程标识: {self.camera_thread.thread_tag})")
            print(f"[Camera3Widget] 首次启动视频流 (线程标识: {self.camera_thread.thread_tag})")
        
        # 情况2：线程已创建且处于暂停状态
        elif self.camera_thread.paused and self.camera_thread.isRunning():
            self.camera_thread.resume()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.record_start_btn.setEnabled(True)
            #日志2
            self.update_status(f"恢复视频流 (线程标识: {self.camera_thread.thread_tag})")
            print(f"[Camera3Widget] 恢复视频流 (线程标识: {self.camera_thread.thread_tag})")
        
        # 情况3：线程已在运行（忽略重复点击）
        else:
            #日志3
            self.update_status("视频流已在运行，忽略操作", level="warn")
            print(f"[Camera3Widget] 视频流已在运行，忽略操作")

    def pause_camera(self):
        """暂停视频流（保留画面和资源）"""
        print(f"[Camera3Widget] 点击暂停按钮")
        if not self.camera_thread or not self.camera_thread.isRunning() or self.camera_thread.paused:
            return
        
        self.camera_thread.pause()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.record_start_btn.setEnabled(False)  # 暂停时不允许录像
        if self.is_recording:
            self.stop_recording()  # 暂停时自动停止录像
            #日志4
            self.update_status(f"暂停视频流 (线程标识: {self.camera_thread.thread_tag})")
        print(f"[Camera3Widget] 暂停视频流 (线程标识: {self.camera_thread.thread_tag})")

    def update_frame(self, frame):
        """接收新帧并交给处理线程"""
        try:
            # 录像处理在主线程简单处理，只写原始帧
            self.handle_recording(frame)
            
            # 保存原始帧引用
            self.last_original_image = frame.copy()
            
            # 将帧交给处理线程
            self.processing_thread.set_frame(frame)
            
            # 快速显示原始帧，不等待处理结果
            self._fast_show_original(frame)
            
        except Exception as e:
            error_msg = f"帧接收错误: {str(e)}"
            self.update_status(error_msg)
            print(f"[Camera3Widget] {error_msg}")

    def _fast_show_original(self, frame):
        """快速显示原始帧，减少延迟"""
        try:
            # 颜色空间转换（BGR->RGB）
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为QImage
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # 缩放适配显示区域
            pixmap = QPixmap.fromImage(qt_image).scaled(
                self.label1.width(), 
                self.label1.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.label1.setPixmap(pixmap)
        except Exception as e:
            print(f"快速显示错误: {str(e)}")

    def handle_recording(self, frame):
        """处理录像逻辑，只在主线程做简单操作"""
        if self.is_recording and self.video_writer and self.video_params:
            try:
                # 确保帧尺寸与录像参数一致
                if (frame.shape[1], frame.shape[0]) != (self.video_params["width"], self.video_params["height"]):
                    frame = cv2.resize(frame, (self.video_params["width"], self.video_params["height"]))
                self.video_writer.write(frame)
            except Exception as e:
                print(f"录像写入错误: {str(e)}")

    # 录像相关函数
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
            # 创建保存目录
            save_dir = "./Saved_Files/Cam3"
            os.makedirs(save_dir, exist_ok=True)
            self.video_filename = f"{save_dir}/Cam3_recording_{current_time}.mp4"
            
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
            self.update_status(f"录像启动失败: {str(e)}")
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
            self.update_status(f"录像停止失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"录像停止失败: {str(e)}")

    def update_params(self, params):
        """更新视频参数显示"""
        self.video_params = params  # 保存参数用于录像
        self.resolution_label.setText(f"{params['width']}x{params['height']}")
        self.fps_label.setText(f"{params['fps']}")
        # 编码格式转换为可读字符串
        codec = params['codec']
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        self.codec_label.setText(codec_str)
        print(f"[参数更新] 分辨率: {params['width']}x{params['height']}, FPS: {params['fps']}, 编码: {codec_str}")

    def _on_processed(self, results):
        """处理图像处理线程返回的结果"""
        try:
            frame, spots_output, heatmap = results
            self.last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 显示处理后的图像
            self.show_cv_image(self.label2, spots_output)
            self.show_cv_image(self.label3, heatmap)
            
        except Exception as e:
            error_msg = f"处理结果显示错误: {str(e)}"
            self.update_status(error_msg)

    def show_cv_image(self, label, img):
        """优化的图像显示函数"""
        try:  
            if img is None or img.size == 0:
                return
                
            # 获取标签尺寸
            label_width = label.width()
            label_height = label.height()
            
            # 图像尺寸
            height, width = img.shape[:2]
            
            # 计算缩放比例
            scale = min(label_width / width, label_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 缩放图像以提高显示效率
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # 根据图像类型转换为QImage
            if len(img.shape) == 2:  # 灰度图
                q_img = QImage(img.data, new_width, new_height, new_width, QImage.Format_Grayscale8)
            else:  # 彩色图
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_img.data, new_width, new_height, new_width * 3, QImage.Format_RGB888)
                
            # 显示图像
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap)
            
        except Exception as e:
            self.update_status(f"图像显示错误: {str(e)}")

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

    def _on_algo_changed(self, algo_type):
        """算法类型改变时更新"""
        self.algo_type = algo_type
        self.processing_thread.set_algo_type(algo_type)
        print(f"算法类型已切换为: {algo_type}")

    # 串口控制函数
    def connect_serial(self):
        """连接串口"""
        if self.controller.connect():
            self.update_status("串口连接成功")
            self.connect_serial_btn.setEnabled(False)
            self.disconnect_serial_btn.setEnabled(True)
            # 启用其他控制按钮
            self.tele_focus_btn.setEnabled(True)
            self.wide_focus_btn.setEnabled(True)
            self.stop_focus_btn.setEnabled(True)
            self.scene_compensation_btn.setEnabled(True)
            self.set_integration_btn.setEnabled(True)
            self.set_fps_btn.setEnabled(True)
        else:
            self.update_status("串口连接失败")
            QMessageBox.warning(self, "连接失败", "无法连接到串口设备，请检查设备是否正确连接")

    def disconnect_serial(self):
        """断开串口连接"""
        self.controller.disconnect()
        self.update_status("串口已断开连接")
        self.connect_serial_btn.setEnabled(True)
        self.disconnect_serial_btn.setEnabled(False)
        # 禁用控制按钮
        self.tele_focus_btn.setEnabled(False)
        self.wide_focus_btn.setEnabled(False)
        self.stop_focus_btn.setEnabled(False)
        self.scene_compensation_btn.setEnabled(False)
        self.set_integration_btn.setEnabled(False)
        self.set_fps_btn.setEnabled(False)

    def tele_focus(self):
        """调焦+"""
        if self.controller:
            self.controller.tele_focus()  # 假设控制器有此方法
            self.update_status("正在调焦+")

    def wide_focus(self):
        """调焦-"""
        if self.controller:
            self.controller.wide_focus()  # 假设控制器有此方法
            self.update_status("正在调焦-")

    def stop_focus(self):
        """调焦停"""
        if self.controller:
            self.controller.stop_focus()  # 假设控制器有此方法
            self.update_status("调焦已停止")

    def scene_compensation(self):
        """场景补偿"""
        if self.controller:
            self.controller.scene_compensation()  # 假设控制器有此方法
            self.update_status("已执行场景补偿")

    def set_zoom(self, button):
        """设置电子放大倍数"""
        zoom_level = 1
        if button == self.zoom_2x_btn:
            zoom_level = 2
        elif button == self.zoom_4x_btn:
            zoom_level = 4
            
        if self.controller:
            self.controller.set_zoom(zoom_level)  # 假设控制器有此方法
            self.update_status(f"电子放大已设置为 {zoom_level}倍")

    def set_integration_time(self):
        """设置积分时间"""
        try:
            time = int(self.integration_input.text())
            if self.controller:
                self.controller.set_integration_time(time)  # 假设控制器有此方法
                self.update_status(f"积分时间已设置为 {time}ms")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的整数")

    def set_frame_rate(self):
        """设置帧频"""
        try:
            fps = int(self.fps_input.text())
            if self.controller:
                self.controller.set_frame_rate(fps)  # 假设控制器有此方法
                self.update_status(f"帧频已设置为 {fps}Hz")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的整数")

    def save_all(self):
        """保存全部数据"""
        # 创建保存目录
        save_dir = "./Saved_Files/Cam3"
        os.makedirs(save_dir, exist_ok=True)
        
        current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        try:
            if self.last_original_image is not None:
                orig_path = f"{save_dir}/Cam3_original_{current_time}.png"
                cv2.imwrite(orig_path, self.last_original_image)
                
            if self.last_gray is not None:
                gray_path = f"{save_dir}/Cam3_gray_{current_time}.png"
                cv2.imwrite(gray_path, self.last_gray)
                
            if self.last_3d_image is not None:
                img3d_path = f"{save_dir}/Cam3_3d_{current_time}.png"
                cv2.imwrite(img3d_path, self.last_3d_image)
                
            self.update_status(f"数据保存完成，路径: {save_dir}")
        except Exception as e:
            self.update_status(f"数据保存失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def open_parameter_calculation_window(self):
        """打开参数计算窗口"""
        self.update_status("已打开激光参数计算器")
        self.param_window = ParameterCalculationWindow()
        self.param_window.show()

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        # 停止相机线程
        if self.camera_thread:
            self.camera_thread.stop_thread()
        
        # 停止图像处理线程
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
        
        # 确保录像已停止
        if self.is_recording:
            self.stop_recording()
            
        # 断开串口连接
        self.disconnect_serial()
        
        event.accept()


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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Camera3Widget()
    window.show()
    sys.exit(app.exec_())