import cv2
import numpy as np
import sys
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QGroupBox, QFormLayout)

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
    """相机界面（仅开始/停止两个按钮）"""
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.rtsp_url = "rtsp://192.168.0.105/live.sdp"  # RTSP地址统一配置
        self.setWindowTitle("RTSP视频流监控")
        print(f"[Camera2Widget] 初始化界面")
        self.init_ui()

    def init_ui(self):
        """完整UI初始化（仅两个控制按钮）"""
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
        
        # 控制按钮区域（仅开始和停止两个按钮）
        control_group = QGroupBox("视频控制")
        control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("▶ 开始/恢复视频流")
        self.start_btn.setObjectName("func_btn")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_or_resume_camera)
        
        self.stop_btn = QPushButton("⏹ 暂停视频流")
        self.stop_btn.setObjectName("func_btn")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.clicked.connect(self.pause_camera)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        # 状态显示
        status_group = QGroupBox("连接状态")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("准备连接长波相机...")
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
        """)
        
        self.setLayout(main_layout)
        self.setMinimumSize(1150, 650)
        print(f"[Camera2Widget] UI初始化完成")

    def start_or_resume_camera(self):
        """开始或恢复视频流（统一处理）"""
        print(f"[Camera2Widget] 点击开始/恢复按钮")
        
        # 情况1：线程未创建（首次启动）
        if not self.camera_thread:
            self.camera_thread = Camera2Thread(self.rtsp_url)
            self.camera_thread.frame_signal.connect(self.update_frame)
            self.camera_thread.status_signal.connect(self.update_status)
            self.camera_thread.param_signal.connect(self.update_params)
            self.camera_thread.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            print(f"[Camera2Widget] 首次启动视频流 (线程标识: {self.camera_thread.thread_tag})")
        
        # 情况2：线程已创建且处于暂停状态
        elif self.camera_thread.paused and self.camera_thread.isRunning():
            self.camera_thread.resume()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            print(f"[Camera2Widget] 恢复视频流 (线程标识: {self.camera_thread.thread_tag})")
        
        # 情况3：线程已在运行（忽略重复点击）
        else:
            print(f"[Camera2Widget] 视频流已在运行，忽略操作")

    def pause_camera(self):
        """暂停视频流（保留画面和资源）"""
        print(f"[Camera2Widget] 点击暂停按钮")
        if not self.camera_thread or not self.camera_thread.isRunning() or self.camera_thread.paused:
            return
        
        self.camera_thread.pause()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        print(f"[Camera2Widget] 暂停视频流 (线程标识: {self.camera_thread.thread_tag})")

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
            print(f"[Camera2Widget] {error_msg}")

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

    def closeEvent(self, event):
        """窗口关闭时彻底停止线程并释放资源"""
        print(f"[Camera2Widget] 窗口关闭，彻底停止线程")
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
        super().closeEvent(event)
