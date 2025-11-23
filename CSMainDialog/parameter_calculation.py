import math

import sys
import os
from PyQt5.QtCore import Qt, QTimer, QTime
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                            QGridLayout, QPushButton, QTableWidget, QTableWidgetItem,
                            QSpacerItem, QSizePolicy, QMessageBox)


def calculate_ideal_divergence(wavelength, aperture):
    """计算理想半发散角"""
    # 将波长从纳米转换为米
    wavelength_m = wavelength * 10**-9
    # 将出射口径从毫米转换为米
    aperture_m = aperture * 10**-3
    return (2*wavelength_m) / (math.pi * aperture_m)

def calculate_actual_divergence(spot_diameter, aperture, transmission_distance):
    """计算实际半发散角"""
    # 将半径从毫米转换为米
    spot_diameter_m = spot_diameter * 10**-3
    aperture_m = aperture * 10**-3
    # 将传输距离保持为米
    return (spot_diameter_m - aperture_m) / (2 * transmission_distance)

def calculate_quality_factor(actual_divergence, ideal_divergence):
    """计算质量因子M²"""
    if ideal_divergence == 0:
        return 0
    return actual_divergence / ideal_divergence

class ParameterCalculationWindow(QDialog):
    def __init__(self):
        super(ParameterCalculationWindow, self).__init__()
        self.setWindowTitle('激光参数计算器')
        self.setMinimumSize(963, 760)
        self.center_A = None
        self.center_B = None
        self.center_C = None
        self.layout = QVBoxLayout(self)  
        self.layout.setContentsMargins(20, 5, 20, 20)
        self.layout.setSpacing(15)
        

        self.title_label = QLabel("激光参数计算器")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 25px; font-weight: bold; color: #2E3A59;")
        self.layout.addWidget(self.title_label)


        top_layout = QHBoxLayout()
        self.image_label = QLabel(self)
        # 注意：图片路径可能需要调整，根据新文件位置修改
        pixmap = QPixmap("CSMainDialog\远场光斑发散模型\远场光斑发散模型.png")
        if pixmap.isNull():
            print("图片加载失败！")
        else:
            print("图片加载成功！")
        self.image_label.setPixmap(pixmap.scaled(500, 400, aspectRatioMode=Qt.KeepAspectRatio))
        self.image_label.setStyleSheet("border: 3px solid black;")

        top_layout.addWidget(self.image_label)

        self.table_widget = QTableWidget(self)
        self.table_widget.setRowCount(1)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["远-近夹角", "中-近夹角", "远-中夹角", "测试时间"])
        self.table_widget.setColumnWidth(0, 100)  
        self.table_widget.setColumnWidth(1, 100)  
        self.table_widget.setColumnWidth(2, 100)  
        self.table_widget.setColumnWidth(3, 100)  

        top_layout.addWidget(self.table_widget)
        top_layout.setStretch(0, 1)
        top_layout.setStretch(1, 2)
        top_layout.setAlignment(Qt.AlignLeft)

        header_layout = QVBoxLayout()
        header_layout.addLayout(top_layout)
        self.layout.addLayout(header_layout)

        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)

        self.label1 = QLabel("请输入 波长(nm)：")
        self.label1.setStyleSheet("font-size: 16px;")
        self.input_wavelength = QLineEdit()
        self.input_wavelength.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label1, 0, 0)
        grid_layout.addWidget(self.input_wavelength, 0, 1)

        self.label2 = QLabel("请输入 出射口径(mm)：")
        self.label2.setStyleSheet("font-size: 16px;")
        self.input_aperture = QLineEdit()
        self.input_aperture.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label2, 1, 0)
        grid_layout.addWidget(self.input_aperture, 1, 1)

        self.label3 = QLabel("请输入 远场光斑直径(mm)：")
        self.label3.setStyleSheet("font-size: 16px;")
        self.input_spot_diameter = QLineEdit()
        self.input_spot_diameter.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label3, 2, 0)
        grid_layout.addWidget(self.input_spot_diameter, 2, 1)

        self.label4 = QLabel("请输入 激光功率(W)：")
        self.label4.setStyleSheet("font-size: 16px;")
        self.input_laser_power = QLineEdit()
        self.input_laser_power.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label4, 3, 0)
        grid_layout.addWidget(self.input_laser_power, 3, 1)

        self.label5 = QLabel("请输入 传输距离(m)：")
        self.label5.setStyleSheet("font-size: 16px;")
        self.input_transmission_distance = QLineEdit()
        self.input_transmission_distance.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label5, 4, 0)
        grid_layout.addWidget(self.input_transmission_distance, 4, 1)

        self.label_distance = QLabel("请输入 测距机距离(m)：")
        self.label_distance.setStyleSheet("font-size: 16px;")
        self.input_distance = QLineEdit()
        self.input_distance.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label_distance, 5, 0)
        grid_layout.addWidget(self.input_distance, 5, 1)

        self.label6 = QLabel("理想半发散角(rad)：")
        self.label6.setStyleSheet("font-size: 16px;")
        self.output_ideal_divergence = QLineEdit()
        self.output_ideal_divergence.setReadOnly(True)
        self.output_ideal_divergence.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label6, 0, 2)
        grid_layout.addWidget(self.output_ideal_divergence, 0, 3)

        self.label7 = QLabel("实际半发散角(rad)：")
        self.label7.setStyleSheet("font-size: 16px;")
        self.output_actual_divergence = QLineEdit()
        self.output_actual_divergence.setReadOnly(True)
        self.output_actual_divergence.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label7, 1, 2)
        grid_layout.addWidget(self.output_actual_divergence, 1, 3)

        self.label8 = QLabel("质量因子 M²：")
        self.label8.setStyleSheet("font-size: 16px;")
        self.output_quality_factor = QLineEdit()
        self.output_quality_factor.setReadOnly(True)
        self.output_quality_factor.setStyleSheet("font-size: 14px; height: 30px;")
        grid_layout.addWidget(self.label8, 2, 2)
        grid_layout.addWidget(self.output_quality_factor, 2, 3)

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

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(spacer)

        self.layout.addLayout(grid_layout)

        self.submit_button = QPushButton('计算')
        self.submit_button.setStyleSheet("font-size: 14px; height: 30px;")
        self.submit_button.clicked.connect(self.calculate_parameters)

        self.layout.addWidget(self.submit_button)
        self.setLayout(self.layout)

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_table)
        # self.timer.start(1000)


    # def update_table(self):
    #     angle_A_B = 0.02
    #     angle_B_C = 0.03
    #     angle_C_A = 0.04
    #     current_time = QTime.currentTime().toString('hh:mm:ss')

    #     row_position = self.table_widget.rowCount()
    #     self.table_widget.insertRow(row_position)

    #     self.table_widget.setItem(row_position, 0, QTableWidgetItem(f"{angle_A_B:.3e} rad"))
    #     self.table_widget.setItem(row_position, 1, QTableWidgetItem(f"{angle_B_C:.3e} rad"))
    #     self.table_widget.setItem(row_position, 2, QTableWidgetItem(f"{angle_C_A:.3e} rad"))
    #     self.table_widget.setItem(row_position, 3, QTableWidgetItem(current_time))

    #     self.table_widget.scrollToBottom()

    def calculate_parameters(self):
        try:
            if not self.input_wavelength.text() or not self.input_aperture.text() or not self.input_spot_diameter.text() or not self.input_laser_power.text() or not self.input_transmission_distance.text() or not self.input_distance.text():
                QMessageBox.warning(self, "提示", "请输入数据")
                return
            
            wavelength = float(self.input_wavelength.text().strip())
            aperture = float(self.input_aperture.text().strip())
            spot_diameter = float(self.input_spot_diameter.text().strip())
            laser_power = float(self.input_laser_power.text().strip())
            transmission_distance = float(self.input_transmission_distance.text().strip())
            distance = float(self.input_distance.text().strip())

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

      # ------------------ 定时器槽函数 ------------------
    def update_angles_periodically(self):
        """每秒刷新光斑夹角"""
        if self.angles_active and all([self.center_A, self.center_B, self.center_C]):
            self.update_angles()  # 调用更新夹角和表格
            
    def update_laser_centers(self, centers, transmission_distance):
        if len(centers) < 3:
            return

        self.center_A, self.center_B, self.center_C = centers[:3]

        def calc_angle(p1, p2):
            b = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
            return b / self.transmission_distance

        angle_A_B = calc_angle(self.center_A, self.center_B)
        angle_B_C = calc_angle(self.center_B, self.center_C)
        angle_C_A = calc_angle(self.center_C, self.center_A)

        # 更新输出框
        self.output_angle_A_B.setText(f"{angle_A_B:.3e} rad")
        self.output_angle_B_C.setText(f"{angle_B_C:.3e} rad")
        self.output_angle_C_A.setText(f"{angle_C_A:.3e} rad")

        # 更新表格
        current_time = QTime.currentTime().toString('hh:mm:ss')
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)
        self.table_widget.setItem(row_position, 0, QTableWidgetItem(f"{angle_A_B:.3e} rad"))
        self.table_widget.setItem(row_position, 1, QTableWidgetItem(f"{angle_B_C:.3e} rad"))
        self.table_widget.setItem(row_position, 2, QTableWidgetItem(f"{angle_C_A:.3e} rad"))
        self.table_widget.setItem(row_position, 3, QTableWidgetItem(current_time))
        self.table_widget.scrollToBottom()