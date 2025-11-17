import sys
import os
import platform
import time
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import cv2 as cv
from threading import Thread
import CSMainDialog.spot_detection
sys.path.append(os.path.dirname(__file__))  # 添加当前文件夹到模块搜索路径
from spot_detection import preprocess_image_cv, detect_and_draw_spots, energy_distribution
from reconstruction3d import generate_3d_image
from parameter_calculation import ParameterCalculationWindow
from RangeFinder_driverForGUI import DistanceMeterManager, ContinuousMeasureThread, ProtocolConst, MeasureResult
from camera_control import (
    AutoAdjustExposureGain, SetupExposure, SetupGain,
    g_autoAdjust, SaveExposureAndGain, LoadExposureAndGain
)
from image_cropper import CropDialog
from spot_algorithms import detect_spots
from Cam2.camera_2 import Camera2Widget
from Cam3.camera_3 import Camera3Widget

if platform.system() == 'Windows':
    sys.path.append(os.environ['IPX_CAMSDK_ROOT'] + '/bin/win64_x64/')
    sys.path.append(os.environ['IPX_CAMSDK_ROOT'] + '/bin/win32_i86/')
    import IpxCameraGuiApiPy as IpxCameraGuiApiPy
else:
    import libIpxCameraGuiApiPy as IpxCameraGuiApiPy


class main_Dialog(QWidget):
    log_signal = pyqtSignal(str)
    show3d_finished = pyqtSignal(np.ndarray)
    image_signal = pyqtSignal(object)
    cropped_image_signal = pyqtSignal(object)
    range_result_signal = pyqtSignal(MeasureResult)

    def __init__(self):
        super(main_Dialog, self).__init__()
        self.range_meter = DistanceMeterManager()
        self.continuous_thread = None
        self.range_data = None
        self.cropped_image = None
        self.last_original_image = None
        self.last_gray = None
        self.last_3d_image = None
        self.counter = 0
        self.stop = False
        self.parView = None
        self.algo_type = "A" 

        # 初始化相机系统
        self.PyIpxSystem1 = IpxCameraGuiApiPy.PyIpxSystem()

        self.init_ui()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.log_signal.connect(self.add_log)
        self.show3d_finished.connect(self._on_show3d_finished)
        self.image_signal.connect(self._update_display)
        self.cropped_image_signal.connect(self._process_cropped_image)
        self.range_result_signal.connect(self.update_range_display)

    def closeEvent(self, event):
        """关闭事件，确保所有相机线程都停止"""
        self.camDisconnect()

        for i in range(self.camera_stack.count()):
            widget = self.camera_stack.widget(i)
            if hasattr(widget, 'stop_camera'):
                widget.stop_camera()

        if self.range_meter.connected:
            self.range_meter.disconnect()

        super(main_Dialog, self).closeEvent(event)

    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text_edit.append(f"[{timestamp}] {message}")
        self.log_text_edit.verticalScrollBar().setValue(
            self.log_text_edit.verticalScrollBar().maximum()
        )

    def log(self, message):
        self.log_signal.emit(message)

    def save_log(self):
        if not self.log_text_edit.toPlainText():
            QMessageBox.information(self, "提示", "日志为空，无需保存")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存日志", "", "文本文件 (*.txt);;所有文件 (*)", options=options
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text_edit.toPlainText())
                self.log(f"日志已保存至: {file_path}")
                QMessageBox.information(self, "成功", f"日志已保存至:\n{file_path}")
            except Exception as e:
                self.log(f"日志保存失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")

    def refresh_ports(self):
        self.port_combo.clear()
        ports = self.range_meter.get_available_ports()
        if ports:
            self.port_combo.addItems(ports)
            self.log(f"发现{len(ports)}个可用串口")
        else:
            self.log("未发现可用串口")

    def connect_range_finder(self):
        if not self.port_combo.currentText():
            self.log("请选择串口")
            return

        port = self.port_combo.currentText()
        success, msg = self.range_meter.connect(port)
        self.log(f"测距机连接: {msg}")

        if success:
            self.connect_range_btn.setEnabled(False)
            self.disconnect_range_btn.setEnabled(True)
            self.single_measure_btn.setEnabled(True)
            self.continuous_measure_btn.setEnabled(True)
            self.log(f"波特率: {ProtocolConst.BAUDRATE}")
            self.log(f"数据位: {ProtocolConst.BYTESIZE}")
            self.log(f"校验位: {ProtocolConst.PARITY}")
            self.log(f"停止位: {ProtocolConst.STOPBITS}")

    def disconnect_range_finder(self):
        if self.range_meter.in_continuous_mode:
            self.toggle_continuous_measure()

        msg = self.range_meter.disconnect()
        self.log(f"测距机: {msg}")
        self.connect_range_btn.setEnabled(True)
        self.disconnect_range_btn.setEnabled(False)
        self.single_measure_btn.setEnabled(False)
        self.continuous_measure_btn.setEnabled(False)

    def single_measure(self):
        if not self.range_meter.connected:
            self.log("未连接测距机")
            return

        self.log("开始单次测距...")
        result, msg = self.range_meter.single_measure()
        self.log(f"单次测距: {msg}")

        if result:
            self.range_result_signal.emit(result)

    def toggle_continuous_measure(self):
        if not self.range_meter.connected:
            self.log("未连接测距机")
            return

        if self.range_meter.in_continuous_mode:
            if self.continuous_thread and self.continuous_thread.isRunning():
                self.continuous_thread.stop()
                self.continuous_thread = None
            self.continuous_measure_btn.setText("开始连续测距")
            self.log("已停止连续测距")
        else:
            freq = self.freq_combo.currentData()
            self.continuous_thread = ContinuousMeasureThread(self.range_meter, freq)
            self.continuous_thread.measure_signal.connect(self.range_result_signal)
            self.continuous_thread.status_signal.connect(self.log)
            self.continuous_thread.error_signal.connect(self.log)
            self.continuous_thread.start()
            self.continuous_measure_btn.setText("停止连续测距")
            self.log(f"开始{self.freq_combo.currentText()}连续测距")

    def update_range_display(self, result: MeasureResult):
        self.range_result_table.setItem(0, 1, QTableWidgetItem(str(result.valid)))
        self.range_result_table.setItem(1, 1, QTableWidgetItem(f"{result.distance_first:.1f}"))
        self.range_result_table.setItem(2, 1, QTableWidgetItem(f"{result.distance_last:.1f}"))
        self.range_result_table.setItem(3, 1, QTableWidgetItem(str(result.has_target)))
        self.range_result_table.setItem(4, 1, QTableWidgetItem(str(result.apd_temperature)))
        self.range_result_table.resizeColumnsToContents()

    def crop_image(self):
        if hasattr(self, 'thread') and self.thread.is_alive():
            QMessageBox.warning(self, "警告", "请先停止相机才能进行图像裁切")
            return

        if not hasattr(self, 'last_original_image') or self.last_original_image is None:
            QMessageBox.warning(self, "警告", "没有可用的图像进行裁切")
            return

        dialog = CropDialog(self, self.last_original_image)
        if dialog.exec_() == QDialog.Accepted:
            cropped_img = dialog.get_cropped_image()
            if cropped_img is not None:
                self.log("图像裁切完成，正在处理...")
                Thread(target=self._process_cropped_image_background,
                       args=(cropped_img,), daemon=True).start()

    def _process_cropped_image_background(self, cropped_img):
        try:
            gray, blur = preprocess_image_cv(cropped_img)
            spots_output = detect_and_draw_spots(cropped_img, log_func=self.log)
            heatmap = energy_distribution(gray)
            self.cropped_image = cropped_img
            self.last_gray = gray
            self.cropped_image_signal.emit((cropped_img, spots_output, heatmap))
        except Exception as e:
            self.log(f"处理裁切图像时出错: {e}")

    def _process_cropped_image(self, imgs):
        try:
            cropped_img, spots_output, heatmap = imgs
            self.show_cv_image(self.label1, cropped_img)
            self.show_cv_image(self.label2, spots_output)
            self.show_cv_image(self.label3, heatmap)
            self.log("已更新裁切后的图像及处理结果")
        except Exception as e:
            self.log(f"更新裁切图像显示时出错: {e}")
    

    def show_3d_image(self):
        if not hasattr(self, 'last_gray') or self.last_gray is None:
            self.log("没有可用图像进行3D重构")
            return

        self.pbShow3D.setEnabled(False)
        self.log("开始3D重构...")

        def worker(gray):
            try:
                img3d = generate_3d_image(gray)
            except Exception as e:
                self.log(f"3D重构失败: {e}")
                img3d = None
            self.show3d_finished.emit(img3d)

        t = Thread(target=worker, args=(self.last_gray.copy(),), daemon=True)
        t.start()

    def save_all(self):
        import cv2
        save_dir = os.path.join(os.getcwd(), "Saved_Images")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        def save_label_image(label, name):
            pixmap = label.pixmap()
            if pixmap is None:
                self.log(f"⚠️ {name} 窗格为空，跳过保存。")
                return False

            qimg = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.frombuffer(ptr, np.uint8)
            try:
                arr = arr.reshape((h, w, 3))
            except Exception as e:
                self.log(f"❌ 转换 {name} 图像失败: {e}")
                return False
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            file_path = os.path.join(save_dir, f"{timestamp}_{name}.jpg")
            success = cv2.imwrite(file_path, img_bgr)
            if success:
                self.log(f"✅ 已保存 {file_path}")
            else:
                self.log(f"❌ 保存 {name} 失败。")
            return success

        save_label_image(self.label1, "original")
        save_label_image(self.label2, "spots")
        save_label_image(self.label3, "heatmap")
        save_label_image(self.label4, "3d")

        log_path = os.path.join(save_dir, f"{timestamp}_spots.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(self.log_text_edit.toPlainText())

        self.log(f"📝 已保存日志到 {log_path}")
        self.log("✅ 所有保存任务完成。")

    def CreateDataStreamBuffers(self):
        if hasattr(self, 'data_stream'):
            self.data_stream.FlushBuffers(self.data_stream.Flush_AllDiscard)
        if hasattr(self, 'list1'):
            for x in self.list1:
                self.data_stream.RevokeBuffer(x)
            self.data_stream.ReleaseBufferQueue()

        bufSize = self.data_stream.GetBufferSize()
        minNumBuffers = self.data_stream.GetMinNumBuffers()
        self.list1 = []
        for x in range(minNumBuffers + 1):
            self.list1.append(self.data_stream.CreateBuffer(bufSize))
        self.log(f"已创建 {len(self.list1)} 个数据流缓冲区")
        return self.list1

    def show_cv_image(self, label, img):
        if len(img.shape) == 2:
            qImg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        else:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            qImg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def GrabNewBuffer(self):
        buffer = self.data_stream.GetBuffer(1000)
        if buffer is None:
            self.log("获取数据流缓冲区失败")
            return 0

        if buffer.IsIncomplete():
            self.log("接收到不完整的缓冲区")
            self.data_stream.QueueBuffer(buffer)
            return 0

        img = np.array(buffer.GetBufferPtr()).reshape((buffer.GetHeight(), buffer.GetWidth()))
        img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        self.last_original_image = img_color.copy()

        gray, blur = preprocess_image_cv(img_color)
        spots_output = detect_spots(img_color, self.algo_type)
        heatmap = energy_distribution(gray)
        self.last_gray = gray

        self.image_signal.emit((img_color, spots_output, heatmap))

        self.data_stream.QueueBuffer(buffer)
        self.counter += 1
        if self.counter % 10 == 0:
            self.log(f"已处理 {self.counter} 帧")

        IpxCameraGuiApiPy.PyShowImageOnDisplay(buffer.GetImage())
        return 0

    def threaded_function(self):
        self.stop = False
        self.log("开始图像采集线程")
        while not self.stop:
            self.GrabNewBuffer()
        self.log("图像采集线程已停止")
    
    def auto_adjust(self):
        global g_autoAdjust
        if not hasattr(self, 'device') or not self.device.IsValid():
            self.log("相机未连接")
            QMessageBox.critical(self, "错误", "相机未连接")
            return
        try:
            if hasattr(self, 'thread') and self.thread.is_alive():
                self.log("暂停图像采集以进行自动调节")
                self.camStop()

            g_autoAdjust = True
            success = AutoAdjustExposureGain(self.device, target=140.0, tol=8.0, max_iter=10)
            if success:
                self.log("自动调节快门时间和增益成功")
                pars = self.device.GetCameraParameters()
                parExp = pars.GetFloat("ExposureTimeRaw") or pars.GetInt("ExposureTimeRaw")
                parG = pars.GetFloat("GainRaw") or pars.GetInt("GainRaw")
                if parExp and parG:
                    self.shutter_input.setText(f"{parExp.GetValue()[1]:.2f}")
                    self.gain_input.setText(f"{parG.GetValue()[1]:.2f}")
            else:
                self.log("自动调节失败")
                QMessageBox.critical(self, "错误", "自动调节失败")

            if self.pbPlay.isEnabled() == False and self.pbStop.isEnabled() == True:
                self.log("恢复图像采集")
                self.camPlay()

        except Exception as e:
            self.log(f"自动调节失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"自动调节失败:\n{str(e)}")
        finally:
            g_autoAdjust = False

    def confirm_settings(self):
        if not hasattr(self, 'device') or not self.device.IsValid():
            self.log("相机未连接")
            QMessageBox.critical(self, "错误", "相机未连接")
            return

        pars = self.device.GetCameraParameters()
        if pars is None:
            self.log("无法获取相机参数")
            QMessageBox.critical(self, "错误", "无法获取相机参数")
            return

        shutter_text = self.shutter_input.text().strip()
        gain_text = self.gain_input.text().strip()

        if not shutter_text and not gain_text:
            QMessageBox.warning(self, "输入错误", "请输入正确数值")
            self.log("用户未输入任何值")
            return

        success = True

        if shutter_text:
            try:
                exp_value = float(shutter_text)
                parExp = pars.GetFloat("ExposureTimeRaw") or pars.GetInt("ExposureTimeRaw")
                if parExp is None:
                    raise ValueError("不支持 ExposureTimeRaw 参数")
                exp_min, exp_max = parExp.GetMin()[1], parExp.GetMax()[1]
                if not (exp_min <= exp_value <= exp_max):
                    QMessageBox.warning(self, "输入错误", f"请输入正确数值\n快门时间范围: [{exp_min}, {exp_max}]")
                    self.log(f"快门时间 {exp_value} 超出范围 [{exp_min}, {exp_max}]")
                    success = False
                elif not SetupExposure(self.device, exp_value):
                    success = False
                else:
                    self.log(f"快门时间设置为 {exp_value} us")
            except ValueError:
                QMessageBox.warning(self, "输入错误", "请输入正确数值\n快门时间必须是数字")
                self.log(f"快门时间输入无效: {shutter_text}")
                success = False

        if gain_text:
            try:
                gain_value = float(gain_text)
                parG = pars.GetInt("GainRaw") or pars.GetFloat("GainRaw")
                if parG is None:
                    raise ValueError("不支持 GainRaw 参数")
                gain_min, gain_max = parG.GetMin()[1], parG.GetMax()[1]
                if not (gain_min <= gain_value <= gain_max):
                    QMessageBox.warning(self, "输入错误", f"请输入正确数值\n增益范围: [{gain_min}, {gain_max}]")
                    self.log(f"增益 {gain_value} 超出范围 [{gain_min}, {gain_max}]")
                    success = False
                elif not SetupGain(self.device, gain_value):
                    success = False
                else:
                    self.log(f"增益设置为 {gain_value}")
            except ValueError:
                QMessageBox.warning(self, "输入错误", "请输入正确数值\n增益必须是数字")
                self.log(f"增益输入无效: {gain_text}")
                success = False

        if not success:
            return

        QMessageBox.information(self, "成功", "参数设置成功！")
        self.log("手动参数设置完成")

    def camConnect(self):
        self.log("正在尝试连接相机...")
        self.deviceInfo = self.PyIpxSystem1.SelectCamera(self.winId())
        if self.deviceInfo is None:
            self.log("相机选择已取消或失败")
            return

        self.pbConnect.setEnabled(0)
        self.pbDisconnect.setEnabled(1)
        self.pbPlay.setEnabled(1)
        self.pbStop.setEnabled(0)
        self.pbTree.setEnabled(1)
        self.pbAutoAdjust.setEnabled(1)
        self.pbConfirmSettings.setEnabled(1)
        self.pbCropImage.setEnabled(1)
        self.pbSaveSettings.setEnabled(1)
        self.pbLoadSettings.setEnabled(1)

        self.infoTable.setItem(0, 1, QTableWidgetItem(self.deviceInfo.GetVendor()))
        self.infoTable.setItem(1, 1, QTableWidgetItem(self.deviceInfo.GetModel()))
        self.infoTable.setItem(2, 1, QTableWidgetItem(self.deviceInfo.GetUserDefinedName()))
        self.infoTable.setItem(3, 1, QTableWidgetItem(self.deviceInfo.GetVersion()))
        self.infoTable.setItem(4, 1, QTableWidgetItem(self.deviceInfo.GetSerialNumber()))

        self.device = IpxCameraGuiApiPy.PyIpxCreateDevice(self.deviceInfo)
        self.data_stream = self.device.GetStreamByIndex(0)
        self.gPars = self.device.GetCameraParameters()

        self.log(f"已连接相机：{self.deviceInfo.GetModel()} ({self.deviceInfo.GetSerialNumber()})")

    def camAction(self):
        self.log("正在执行相机操作...")
        IpxCameraGuiApiPy.PyActionCamera(self.winId())

    def camDisconnect(self):
        self.log("正在断开相机连接...")
        self.pbDisconnect.setEnabled(0)
        if hasattr(self, 'device') and self.device.IsValid():
            self.camStop()
            if hasattr(self, 'data_stream'):
                self.data_stream.FlushBuffers(self.data_stream.Flush_AllDiscard)
            if hasattr(self, 'list1'):
                for x in self.list1:
                    self.data_stream.RevokeBuffer(x)
            if hasattr(self, 'data_stream'):
                self.data_stream.Release()
            if self.parView:
                IpxCameraGuiApiPy.PyDestroyGenParamTreeView(self.parView)
                self.parView = None
            if hasattr(self, 'device'):
                self.device.Release()

        self.pbPlay.setEnabled(0)
        self.pbStop.setEnabled(0)
        self.pbConnect.setEnabled(1)
        self.pbTree.setEnabled(0)
        self.pbAutoAdjust.setEnabled(0)
        self.pbConfirmSettings.setEnabled(0)
        self.pbCropImage.setEnabled(0)
        self.pbSaveSettings.setEnabled(0)
        self.pbLoadSettings.setEnabled(0)
        self.log("相机已断开连接")
 
    def camPlay(self):
        self.log("开始相机回放")
        self.CreateDataStreamBuffers()
        IpxCameraGuiApiPy.PyResetDisplay()
        self.pbPlay.setEnabled(0)
        self.gPars.SetIntegerValue("TLParamsLocked", 1)
        self.data_stream.StartAcquisition()
        self.gPars.ExecuteCommand("AcquisitionStart")
        self.thread = Thread(target=self.threaded_function)
        self.thread.start()
        self.pbStop.setEnabled(1)
        self.pbCropImage.setEnabled(0)
        self.log("相机回放已开始")

    def camStop(self):
        self.log("停止相机回放")
        self.pbStop.setEnabled(0)
        self.stop = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        if hasattr(self, 'gPars'):
            self.gPars.ExecuteCommand("停止采集")
        if hasattr(self, 'data_stream'):
            self.data_stream.StopAcquisition(1)
        if hasattr(self, 'gPars'):
            self.gPars.SetIntegerValue("TLParamsLocked", 0)
        self.pbPlay.setEnabled(1)
        self.pbCropImage.setEnabled(1)
        self.log("相机回放已停止")

    def camTree(self):
        self.log("正在打开相机参数树")
        if self.parView:
            IpxCameraGuiApiPy.PyDestroyGenParamTreeView(self.parView)
        self.parView = IpxCameraGuiApiPy.PyCreateGenParamTreeViewForArray(self.gPars, self.winId())

    def _on_show3d_finished(self, proj3d):
        if proj3d is None:
            self.log("3D重构失败")
        else:
            self.last_3d_image = proj3d
            self.show_cv_image(self.label4, proj3d)
            self.log("3D重构完成")
        self.pbShow3D.setEnabled(True)

    def _update_display(self, imgs):
        try:
            img_color, spots_output, heatmap = imgs
            if img_color is not None:
                self.show_cv_image(self.label1, img_color)
            if spots_output is not None:
                self.show_cv_image(self.label2, spots_output)
            if heatmap is not None:
                self.show_cv_image(self.label3, heatmap)
        except Exception as e:
            self.log(f"_update_display 异常: {e}")

    def open_parameter_calculation_window(self):
        self.parameter_calculation_window = ParameterCalculationWindow()
        self.parameter_calculation_window.show()
        self.log("参数计算器已打开")

    
    def switch_camera(self, index):
        current_widget = self.camera_stack.currentWidget()
        if hasattr(current_widget, 'stop_camera'):
            current_widget.stop_camera()

        self.camera_stack.setCurrentIndex(index)
        self.btn_camera1.setChecked(index == 0)
        self.btn_camera2.setChecked(index == 1)
        self.btn_camera3.setChecked(index == 2)

        camera_names = ["相机1", "长波红外相机", "中波红外相机"]
        self.log(f"切换至{camera_names[index]}界面")

    def save_camera_settings(self):
        if not hasattr(self, 'device') or not self.device.IsValid():
            self.log("相机未连接，无法保存参数")
            QMessageBox.critical(self, "错误", "相机未连接")
            return
        if SaveExposureAndGain(self.device):
            self.log("相机参数（快门时间与增益）已成功保存到 camera_settings.txt")
            QMessageBox.information(self, "成功", "参数保存成功")
        else:
            self.log("保存相机参数失败")
            QMessageBox.critical(self, "错误", "保存失败，请查看日志")

    def load_camera_settings(self):
        if not hasattr(self, 'device') or not self.device.IsValid():
            self.log("相机未连接，无法加载参数")
            QMessageBox.critical(self, "错误", "相机未连接")
            return
        if LoadExposureAndGain(self.device):
            self.log("相机参数已从 camera_settings.txt 成功加载并应用")
            QMessageBox.information(self, "成功", "参数加载成功")
            pars = self.device.GetCameraParameters()
            parExp = pars.GetFloat("ExposureTimeRaw") or pars.GetInt("ExposureTimeRaw")
            parG = pars.GetFloat("GainRaw") or pars.GetInt("GainRaw")
            if parExp and parG:
                self.shutter_input.setText(f"{parExp.GetValue()[1]:.2f}")
                self.gain_input.setText(f"{parG.GetValue()[1]:.2f}")
        else:
            self.log("加载相机参数失败")
            QMessageBox.critical(self, "错误", "加载失败，请查看日志")

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 9pt;
            }
            QWidget#top_menu {
                background-color: #2d3e50;
                border-bottom: 2px solid #1a2530;
            }
            QPushButton#menu_btn {
                background-color: #34495e;
                color: #ecf0f1;
                border: none;
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton#menu_btn:hover {
                background-color: #4a6a8b;
            }
            QPushButton#menu_btn:checked {
                background-color: #3498db;
                color: white;
            }
            QPushButton#menu_btn:disabled {
                background-color: #465669;
                color: #7f8c8d;
            }
            QWidget#function_area {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                margin: 4px;
            }
            QPushButton#func_btn {
                background-color: #ffffff;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: normal;
            }
            QPushButton#func_btn:hover {
                background-color: #3498db;
                color: white;
                border: 1px solid #2980b9;
            }
            QPushButton#func_btn:disabled {
                background-color: #f5f5f5;
                color: #95a5a6;
                border: 1px solid #ddd;
            }
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #bdc3c7;
                gridline-color: #ecf0f1;
                selection-background-color: #3498db;
            }
            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid #ecf0f1;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 4px 8px;
                selection-background-color: #3498db;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
            QLabel {
                color: #2c3e50;
            }
            QLabel#title_label {
                font-weight: bold;
                font-size: 10pt;
                color: #2c3e50;
                padding: 4px;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 4px;
            }
            QLabel#image_display {
                background-color: #2c3e50;
                color: white;
                border: 2px solid #34495e;
                border-radius: 4px;
            }
        """)

        top_menu_widget = QWidget()
        top_menu_widget.setObjectName("top_menu")
        top_menu_widget.setFixedHeight(50)
        top_menu_layout = QHBoxLayout(top_menu_widget)
        top_menu_layout.setContentsMargins(10, 5, 10, 5)
        top_menu_layout.setSpacing(8)

        self.btn_camera1 = QPushButton("📷 相机1")
        self.btn_camera2 = QPushButton("📷 相机2")
        self.btn_camera3 = QPushButton("📷 相机3")

        for btn in [self.btn_camera1, self.btn_camera2, self.btn_camera3]:
            btn.setObjectName("menu_btn")
            btn.setCheckable(True)
            btn.setFixedHeight(36)
            top_menu_layout.addWidget(btn)

        top_menu_layout.addStretch()

        title_label = QLabel("光斑识别系统 v1.0")
        title_label.setStyleSheet("color: #ecf0f1; font-size: 14pt; font-weight: bold; padding: 8px;")
        top_menu_layout.addWidget(title_label)

        self.btn_camera1.setChecked(True)

        self.camera_stack = QStackedWidget()

        camera1_widget = QWidget()
        camera1_layout = QVBoxLayout(camera1_widget)
        camera1_layout.setSpacing(8)
        camera1_layout.setContentsMargins(10, 10, 10, 10)

        control_group = QWidget()
        control_group.setObjectName("function_area")
        control_layout = QHBoxLayout(control_group)
        control_layout.setSpacing(6)

        def create_function_btn(name, func, enabled=True):
            btn = QPushButton(name)
            btn.setObjectName("func_btn")
            btn.clicked.connect(func)
            btn.setEnabled(enabled)
            btn.setFixedHeight(40)
            return btn

        self.pbConnect = create_function_btn('🔗 连接', self.camConnect, True)
        self.pbDisconnect = create_function_btn('🔌 断开连接', self.camDisconnect, False)
        self.pbPlay = create_function_btn('▶ 开始', self.camPlay, False)
        self.pbStop = create_function_btn('⏹ 停止', self.camStop, False)
        self.pbTree = create_function_btn('🌳 GenICam 树', self.camTree, False)
        self.pbAction = create_function_btn('⚡ 执行动作', self.camAction, True)
        self.pbSaveLog = create_function_btn('💾 保存日志', self.save_log, True)
        self.pbCropImage = create_function_btn('✂️ 裁切图像', self.crop_image, False)
        self.pbShow3D = create_function_btn('📊 显示 3D', self.show_3d_image, True)
        self.pbSaveAll = create_function_btn('💿 保存全部', self.save_all, True)
        self.pbParameterCalculation = create_function_btn('📐 参数计算',
                                                          self.open_parameter_calculation_window, True)
        #self.pbImport = create_function_btn('🖼 导入图片', self.toggle_import_mode, True)



        control_layout.addWidget(self.pbConnect)
        control_layout.addWidget(self.pbDisconnect)
        control_layout.addWidget(self.pbPlay)
        control_layout.addWidget(self.pbStop)
        control_layout.addWidget(self.pbTree)
        control_layout.addWidget(self.pbAction)
        control_layout.addWidget(self.pbSaveLog)
        control_layout.addWidget(self.pbCropImage)
        control_layout.addWidget(self.pbShow3D)
        control_layout.addWidget(self.pbSaveAll)
        control_layout.addWidget(self.pbParameterCalculation)
        #control_layout.addWidget(self.pbImport)
        control_layout.addWidget(QLabel(" | "))
        self.btn_grp = QButtonGroup(self)
        for idx, (name, key) in enumerate([("标准算法","A"),
                                           ("双光斑算法","B"),
                                           ("单光斑去噪","C"),
                                           ("框选识别","D")]):
            btn = QPushButton(name)
            btn.setCheckable(True); btn.setObjectName("func_btn")
            btn.setFixedHeight(40)
            self.btn_grp.addButton(btn, idx)
            control_layout.addWidget(btn)
            if key == "A": btn.setChecked(True)
        self.btn_grp.buttonClicked.connect(lambda b: setattr(self, 'algo_type', b.text()[-2]))

        control_layout.addStretch()
        control_layout.addStretch()

        camera1_layout.addWidget(control_group)

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(10)

        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        device_group = QGroupBox("设备信息")
        device_layout = QVBoxLayout(device_group)
        self.infoTable = QTableWidget()
        self.initInfoTable()
        device_layout.addWidget(self.infoTable)
        left_layout.addWidget(device_group)

        left_layout.addWidget(self.init_range_control())

        settings_group = QGroupBox("相机设置")
        settings_layout = QGridLayout(settings_group)

        self.pbAutoAdjust = create_function_btn('🔄 Auto Adjust', self.auto_adjust)
        self.pbAutoAdjust.setEnabled(False)
        settings_layout.addWidget(self.pbAutoAdjust, 0, 0, 1, 2)

        settings_layout.addWidget(QLabel('快门时间 (μs):'), 1, 0)
        self.shutter_input = QLineEdit()
        self.shutter_input.setPlaceholderText('输入快门时间')
        settings_layout.addWidget(self.shutter_input, 1, 1)

        settings_layout.addWidget(QLabel('增益:'), 2, 0)
        self.gain_input = QLineEdit()
        self.gain_input.setPlaceholderText('输入增益')
        settings_layout.addWidget(self.gain_input, 2, 1)

        self.pbConfirmSettings = create_function_btn('✅ 确认设置', self.confirm_settings)
        self.pbConfirmSettings.setEnabled(False)
        settings_layout.addWidget(self.pbConfirmSettings, 3, 0, 1, 2)

        self.pbSaveSettings = create_function_btn('💾 保存参数', self.save_camera_settings)
        self.pbSaveSettings.setEnabled(False)
        settings_layout.addWidget(self.pbSaveSettings, 4, 0, 1, 2)

        self.pbLoadSettings = create_function_btn('📂 加载参数', self.load_camera_settings)
        self.pbLoadSettings.setEnabled(False)
        settings_layout.addWidget(self.pbLoadSettings, 5, 0, 1, 2)

        left_layout.addWidget(settings_group)
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        display_group = QGroupBox("图像显示")
        display_layout = QGridLayout(display_group)

        self.label1 = QLabel("原始图像")
        self.label2 = QLabel("光斑识别")
        self.label3 = QLabel("能量分布")
        self.label4 = QLabel("3D重构")

        for i, label in enumerate([self.label1, self.label2, self.label3, self.label4]):
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

        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setMaximumHeight(200)
        self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit)
        right_layout.addWidget(log_group)

        content_layout.addWidget(left_panel)
        content_layout.addWidget(right_panel)

        camera1_layout.addWidget(content_widget)
        self.camera_stack.addWidget(camera1_widget)

        camera2_widget = Camera2Widget()
        self.camera_stack.addWidget(camera2_widget)

        camera3_widget = Camera3Widget()
        self.camera_stack.addWidget(camera3_widget)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(top_menu_widget)
        main_layout.addWidget(self.camera_stack)

        self.btn_camera1.clicked.connect(lambda: self.switch_camera(0))
        self.btn_camera2.clicked.connect(lambda: self.switch_camera(1))
        self.btn_camera3.clicked.connect(lambda: self.switch_camera(2))

        self.setWindowTitle("光斑识别系统")
        self.setMinimumSize(1400, 900)
        self.refresh_ports()

    
    def initInfoTable(self):
        self.infoTable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.infoTable.setRowCount(5)
        self.infoTable.setColumnCount(2)
        self.infoTable.setItem(0, 0, QTableWidgetItem('Manufacturer'))
        self.infoTable.setItem(1, 0, QTableWidgetItem('Model'))
        self.infoTable.setItem(2, 0, QTableWidgetItem('Name'))
        self.infoTable.setItem(3, 0, QTableWidgetItem('Version'))
        self.infoTable.setItem(4, 0, QTableWidgetItem('Serial Number'))
        h1 = self.infoTable.horizontalHeader()
        h1.setStretchLastSection(True)
        h1.hide()
        v1 = self.infoTable.verticalHeader()
        v1.hide()

    def init_range_control(self):
        range_panel = QGroupBox("测距机控制")
        range_layout = QVBoxLayout(range_panel)

        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("串口:"))
        self.port_combo = QComboBox()
        port_layout.addWidget(self.port_combo)
        self.refresh_port_btn = QPushButton("🔄 刷新")
        self.refresh_port_btn.setObjectName("func_btn")
        self.refresh_port_btn.clicked.connect(self.refresh_ports)
        port_layout.addWidget(self.refresh_port_btn)
        range_layout.addLayout(port_layout)

        connect_layout = QHBoxLayout()
        self.connect_range_btn = QPushButton("🔗 连接测距机")
        self.connect_range_btn.setObjectName("func_btn")
        self.connect_range_btn.clicked.connect(self.connect_range_finder)
        connect_layout.addWidget(self.connect_range_btn)

        self.disconnect_range_btn = QPushButton("🔌 断开连接")
        self.disconnect_range_btn.setObjectName("func_btn")
        self.disconnect_range_btn.clicked.connect(self.disconnect_range_finder)
        self.disconnect_range_btn.setEnabled(False)
        connect_layout.addWidget(self.disconnect_range_btn)
        range_layout.addLayout(connect_layout)

        measure_layout = QHBoxLayout()
        self.single_measure_btn = QPushButton("📏 单次测距")
        self.single_measure_btn.setObjectName("func_btn")
        self.single_measure_btn.clicked.connect(self.single_measure)
        self.single_measure_btn.setEnabled(False)
        measure_layout.addWidget(self.single_measure_btn)

        self.continuous_measure_btn = QPushButton("🔄 开始连续测距")
        self.continuous_measure_btn.setObjectName("func_btn")
        self.continuous_measure_btn.clicked.connect(self.toggle_continuous_measure)
        self.continuous_measure_btn.setEnabled(False)
        measure_layout.addWidget(self.continuous_measure_btn)
        range_layout.addLayout(measure_layout)

        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("连续测距频率:"))
        self.freq_combo = QComboBox()
        self.freq_combo.addItem("4Hz", ProtocolConst.FREQ_4HZ)
        self.freq_combo.addItem("1Hz", ProtocolConst.FREQ_1HZ)
        freq_layout.addWidget(self.freq_combo)
        range_layout.addLayout(freq_layout)

        range_layout.addWidget(QLabel("测距结果:"))
        self.range_result_table = QTableWidget()
        self.range_result_table.setRowCount(5)
        self.range_result_table.setColumnCount(2)
        self.range_result_table.setItem(0, 0, QTableWidgetItem("数据有效性"))
        self.range_result_table.setItem(1, 0, QTableWidgetItem("首目标距离(m)"))
        self.range_result_table.setItem(2, 0, QTableWidgetItem("末目标距离(m)"))
        self.range_result_table.setItem(3, 0, QTableWidgetItem("是否有目标"))
        self.range_result_table.setItem(4, 0, QTableWidgetItem("APD温度(℃)"))
        h = self.range_result_table.horizontalHeader()
        h.setStretchLastSection(True)
        h.hide()
        v = self.range_result_table.verticalHeader()
        v.hide()
        range_layout.addWidget(self.range_result_table)

        return range_panel


