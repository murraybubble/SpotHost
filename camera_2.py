import cv2
import numpy as np
import sys
import datetime  # 新增：日志时间戳需要
from PyQt5.QtCore import QThread, pyqtSignal,Qt
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QGroupBox, QFormLayout,
                            QDialog, QSlider, QMessageBox, QSpinBox, QDialogButtonBox,
                            QTextEdit, QComboBox)  # 新增：QTextEdit（日志）、QComboBox（串口选择）
from cam2_3_serialControl import CameraController_1

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


class Camera2Widget(QWidget):
    """相机界面（包含控制按钮+串口选择+日志窗口）"""
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.rtsp_url = "rtsp://192.168.0.105/live.sdp"  # RTSP地址统一配置
        self.detail_gain_value = 0  # 细节增益当前值

        # 初始化串口控制器
        self.controller = CameraController_1(baudrate=115200)
        
        # 关键修改：先初始化UI（创建所有控件），再连接串口
        self.setWindowTitle("RTSP视频流监控")
        self.init_ui()  # 先创建控件（包括status_label和日志窗口）
        self.init_serial_connection()  # 后连接串口（此时控件已存在）

    def init_serial_connection(self):
        """初始化串口连接（自动连接，失败则提示）"""
        if self.controller.connect():
            self.update_status(f"串口连接成功", level="info")
        else:
            self.update_status(f"串口连接失败，请检查设备", level="warn")

    def init_ui(self):
        """完整UI初始化（含串口选择+日志窗口）"""
        main_layout = QHBoxLayout(self)
        
        # 左侧控制面板（宽度500）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(500)
        
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
        
        # 视频控制按钮区域（开始和停止按钮）
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
        
        # 串口控制区域（新增：解决串口选择问题）
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
        
        # 状态显示（status_label在这里创建，不会重复）
        status_group = QGroupBox("连接状态")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("准备连接长波相机...")  # 唯一的status_label定义
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
        
        # 日志窗口（新增：按需求添加）
        log_group = QGroupBox("操作日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        log_layout.addWidget(self.log_text)
        
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.setMinimumHeight(30)
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(self.clear_log_btn)
        
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        # 填充剩余空间
        left_layout.addStretch()
        
        # 右侧视频显示区域
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
        self.setMinimumSize(1350, 700)  # 适配日志窗口，加宽窗口
        print(f"[Camera2Widget] UI初始化完成")
        
        # 初始化时刷新串口列表

    # ----------------------
    # 新增：串口控制相关方法
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
    # 新增：日志相关方法
    # ----------------------
    def add_log(self, message, level="info"):
        """添加带时间戳和颜色的日志"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 日志颜色：info(黑)、warn(橙)、error(红)
        color_map = {"info": "#000000", "warn": "#FFA500", "error": "#FF0000"}
        color = color_map.get(level, "#000000")
        log_line = f'<span style="color:{color}">[{timestamp}] [{level.upper()}] {message}</span><br>'
        self.log_text.insertHtml(log_line)
        self.log_text.moveCursor(QTextCursor.End)  # 自动滚动到底部

    def clear_log(self):
        """清空日志窗口"""
        self.log_text.clear()
        self.add_log("日志已清空", level="info")

    # ----------------------
    # 核心功能方法（修改update_status）
    # ----------------------
    def update_status(self, message, level="info"):
        """同时更新状态标签和日志（解决原报错核心）"""
        self.status_label.setText(message)  # 状态标签显示最新信息
        self.add_log(message, level)  # 日志记录历史信息
        print(f"[状态更新] {message}")

    def start_or_resume_camera(self):
        """开始或恢复视频流（统一处理）"""
        print(f"[Camera2Widget] 点击开始/恢复按钮")
        
        if not self.camera_thread:
            self.camera_thread = Camera2Thread(self.rtsp_url)
            self.camera_thread.frame_signal.connect(self.update_frame)
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
        """暂停视频流（保留画面和资源）"""
        print(f"[Camera2Widget] 点击暂停按钮")
        if not self.camera_thread or not self.camera_thread.isRunning() or self.camera_thread.paused:
            return
        
        self.camera_thread.pause()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_status(f"暂停视频流 (线程标识: {self.camera_thread.thread_tag})")

    def update_frame(self, frame):
        """更新视频帧（暂停时保留最后一帧）"""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(), 
                self.video_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(pixmap)
        except Exception as e:
            error_msg = f"帧处理错误: {str(e)}"
            self.update_status(error_msg, level="error")

    def update_params(self, params):
        """更新视频参数显示"""
        self.resolution_label.setText(f"{params['width']}x{params['height']}")
        self.fps_label.setText(f"{params['fps']}")
        codec = params['codec']
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        self.codec_label.setText(codec_str)
        self.update_status(f"参数更新：分辨率{params['width']}x{params['height']}，FPS{params['fps']}，编码{codec_str}")

    # 相机控制接口（保持不变，已对接串口）
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
                self.camera_thread.frame_signal.disconnect(self.update_frame)
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

