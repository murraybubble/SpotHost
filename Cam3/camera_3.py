import cv2
import numpy as np
import sys
import time
import serial
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QGroupBox, QFormLayout,
                            QRadioButton, QButtonGroup, QLineEdit, QMessageBox)
from cam2_3_serialControl import CameraController_2  # 导入相机控制类

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


class Camera3Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.rtsp_url = "rtsp://192.168.0.106/live.sdp"  # RTSP地址统一配置
        self.camera_controller = CameraController_2()  # 创建相机控制器实例
        self.setWindowTitle("RTSP视频流监控与相机控制")
        print(f"[Camera3Widget] 初始化界面")
        self.init_ui()

    def init_ui(self):
        """完整UI初始化（包含新增控制功能）"""
        main_layout = QHBoxLayout(self)
        
        # 左侧控制面板（宽度调整为600以容纳更多控件）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(600)
        
        # 标题
        title_label = QLabel("中波红外相机 (RTSP)")
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
        
        # 视频控制按钮区域
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
        
        # 填充剩余空间
        left_layout.addStretch()
        
        # 右侧视频显示区域（暂停时保留最后一帧）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 6px;
                color: white;
                font-weight: bold;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("视频显示区域\n等待启动...")
        right_layout.addWidget(self.video_label)
        right_layout.setAlignment(Qt.AlignCenter)
        
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
            QGroupBox {
                font-size: 11pt;
                font-weight: bold;
                color: #2c3e50;
                margin: 10px;
                padding: 10px;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
            }
            QLineEdit {
                padding: 5px;
                margin: 5px;
                font-size: 11pt;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
            QRadioButton {
                margin: 5px;
                padding: 5px;
                font-size: 11pt;
            }
        """)
        
        self.setLayout(main_layout)
        self.setMinimumSize(1250, 650)
        print(f"[Camera3Widget] UI初始化完成")

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
            print(f"[Camera3Widget] 首次启动视频流 (线程标识: {self.camera_thread.thread_tag})")
        
        # 情况2：线程已创建且处于暂停状态
        elif self.camera_thread.paused and self.camera_thread.isRunning():
            self.camera_thread.resume()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            print(f"[Camera3Widget] 恢复视频流 (线程标识: {self.camera_thread.thread_tag})")
        
        # 情况3：线程已在运行（忽略重复点击）
        else:
            print(f"[Camera3Widget] 视频流已在运行，忽略操作")

    def pause_camera(self):
        """暂停视频流（保留画面和资源）"""
        print(f"[Camera3Widget] 点击暂停按钮")
        if not self.camera_thread or not self.camera_thread.isRunning() or self.camera_thread.paused:
            return
        
        self.camera_thread.pause()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        print(f"[Camera3Widget] 暂停视频流 (线程标识: {self.camera_thread.thread_tag})")

    def update_frame(self, frame):
        """更新视频帧（暂停时保留最后一帧）"""
        try:
            # 颜色空间转换（BGR->RGB）
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为QImage
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # 缩放适配显示区域
            pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(), 
                self.video_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(pixmap)
        except Exception as e:
            error_msg = f"帧处理错误: {str(e)}"
            self.update_status(error_msg)
            print(f"[Camera3Widget] {error_msg}")

    def update_status(self, message):
        """更新状态信息"""
        self.status_label.setText(message)
        print(f"[状态更新] {message}")

    def update_params(self, params):
        """更新视频参数显示"""
        self.resolution_label.setText(f"{params['width']}x{params['height']}")
        self.fps_label.setText(f"{params['fps']}")
        # 编码格式转换为可读字符串
        codec = params['codec']
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        self.codec_label.setText(codec_str)
        print(f"[参数更新] 分辨率: {params['width']}x{params['height']}, FPS: {params['fps']}, 编码: {codec_str}")

    # 新增的串口控制函数
    def connect_serial(self):
        """连接串口"""
        if self.camera_controller.connect():
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
        self.camera_controller.disconnect()
        self.update_status("串口已断开连接")
        self.connect_serial_btn.setEnabled(True)
        self.disconnect_serial_btn.setEnabled(False)
        # 禁用其他控制按钮
        self.tele_focus_btn.setEnabled(False)
        self.wide_focus_btn.setEnabled(False)
        self.stop_focus_btn.setEnabled(False)
        self.scene_compensation_btn.setEnabled(False)
        self.set_integration_btn.setEnabled(False)
        self.set_fps_btn.setEnabled(False)

    # 新增的相机控制函数
    def tele_focus(self):
        """调焦+"""
        if self.camera_controller.tele_focus():
            self.update_status("发送调焦+命令")
        else:
            self.update_status("调焦+命令发送失败")

    def wide_focus(self):
        """调焦-"""
        if self.camera_controller.wide_focus():
            self.update_status("发送调焦-命令")
        else:
            self.update_status("调焦-命令发送失败")

    def stop_focus(self):
        """调焦停"""
        if self.camera_controller.stop_focus():
            self.update_status("发送调焦停命令")
        else:
            self.update_status("调焦停命令发送失败")

    def set_zoom(self, button):
        """设置电子放大倍数"""
        zoom_level = self.zoom_group.id(button)
        if self.camera_controller.set_zoom(zoom_level):
            self.update_status(f"设置电子放大为{[1, 2, 4][zoom_level]}倍")
        else:
            self.update_status("电子放大设置失败")

    def set_integration_time(self):
        """设置积分时间"""
        try:
            ms = float(self.integration_input.text())
            if self.camera_controller.set_integration_time(ms):
                self.update_status(f"设置积分时间为{ms}ms")
            else:
                self.update_status("积分时间设置失败")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字")

    def set_frame_rate(self):
        """设置帧频"""
        try:
            hz = float(self.fps_input.text())
            if self.camera_controller.set_frame_rate(hz):
                self.update_status(f"设置帧频为{hz}Hz")
            else:
                self.update_status("帧频设置失败")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字")

    def scene_compensation(self):
        """场景补偿"""
        if self.camera_controller.scene_compensation():
            self.update_status("发送场景补偿命令")
        else:
            self.update_status("场景补偿命令发送失败")

    def closeEvent(self, event):
        """窗口关闭时彻底停止线程并释放资源"""
        print(f"[Camera3Widget] 窗口关闭，彻底停止线程和串口连接")
        # 停止相机线程
        if self.camera_thread:
            self.camera_thread.stop_thread()
            # 安全断开信号
            try:
                self.camera_thread.frame_signal.disconnect(self.update_frame)
                self.camera_thread.status_signal.disconnect(self.update_status)
                self.camera_thread.param_signal.disconnect(self.update_params)
            except:
                pass
            self.camera_thread = None
        # 断开串口连接
        self.camera_controller.disconnect()
        super().closeEvent(event)

