import math
import re  
import sys
import os
from PyQt5.QtCore import Qt, QTimer, QTime
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                            QGridLayout, QPushButton, QTableWidget, QTableWidgetItem,
                            QSpacerItem, QSizePolicy, QMessageBox,QFileDialog,QHeaderView)


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

def calculate_distance(coord1, coord2):
    """计算两个坐标之间的像素距离"""
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class ParameterCalculationWindow(QDialog):
    def __init__(self):
        super(ParameterCalculationWindow, self).__init__()
        self.setWindowTitle('激光参数计算器')
        self.setMinimumSize(963, 760)
        self.coordinates = []  # 用于存储读取的坐标数据

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
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["A-C", "B-C", "A-B"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


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

        self.label_ac = QLabel("A-C：远红外-近红外激光相互夹角")
        self.label_ac.setStyleSheet("font-size: 16px;")
        grid_layout.addWidget(self.label_ac, 3, 2)

        self.label_bc = QLabel("B-C：中红外-近红外激光相互夹角")
        self.label_bc.setStyleSheet("font-size: 16px;")
        grid_layout.addWidget(self.label_bc, 4, 2)

        self.label_ab = QLabel("A-B：远红外-中红外激光相互夹角")
        self.label_ab.setStyleSheet("font-size: 16px;")
        grid_layout.addWidget(self.label_ab, 5, 2)



        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(spacer)

        self.layout.addLayout(grid_layout)

        self.submit_button = QPushButton('计算')
        self.submit_button.setStyleSheet("font-size: 14px; height: 30px;")
        self.submit_button.clicked.connect(self.calculate_parameters)

        self.read_log_button = QPushButton('读取日志')
        self.read_log_button.setStyleSheet("font-size: 14px; height: 30px;")
        self.read_log_button.clicked.connect(self.read_log_file)

        self.layout.addWidget(self.read_log_button)
        self.layout.addWidget(self.submit_button)
        self.setLayout(self.layout)

    def read_log_file(self):
        """读取日志文件并提取坐标信息"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择日志文件", "", "文本文件 (*.txt);;所有文件 (*)")
        if not file_path:
            return  # 如果没有选择文件则返回

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            coordinates = []
            for line in lines:
                # 使用正则表达式提取坐标，格式：(x, y)
                match = re.findall(r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)', line)
                if len(match) == 3:  # 每行应该有三个坐标
                    coordinates.append([(float(x), float(y)) for x, y in match])

            self.coordinates = coordinates  # 存储读取的坐标数据
            QMessageBox.information(self, "读取日志", "日志已成功读取！")
        except ValueError as e:
            QMessageBox.critical(self, "输入错误", str(e))

    def calculate_parameters(self):
        try:
            if not self.input_wavelength.text() or not self.input_aperture.text() or not self.input_spot_diameter.text() or not self.input_laser_power.text() or not self.input_transmission_distance.text() or not self.input_distance.text():
                QMessageBox.warning(self, "提示", "请输入数据")
                return
            if not self.coordinates:
                QMessageBox.warning(self, "提示", "请先读取日志文件")
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

            # 获取每像素实际距离 (直接使用事先设定的值)
            pixel_scale = 0.0005  # 每像素实际距离，单位：米/像素 (0.5mm/px)
            # 获取传输距离
            # transmission_distance = float(self.input_transmission_distance.text().strip())
            distances = []
            for coord_set in self.coordinates:
              if len(coord_set) == 3:
                dist1 = calculate_distance(coord_set[0], coord_set[1])
                dist2 = calculate_distance(coord_set[0], coord_set[2])
                dist3 = calculate_distance(coord_set[1], coord_set[2])
            # 将距离除以传输距离，得到夹角
                angle1 = dist1 * pixel_scale / transmission_distance
                angle2 = dist2 * pixel_scale / transmission_distance
                angle3 = dist3 * pixel_scale / transmission_distance
                distances.append((angle1, angle2, angle3))

                # 更新表格，将夹角数据填入表格
                self.table_widget.setRowCount(len(distances))  # 设置表格行数为计算出的行数
                for row, dist in enumerate(distances):
                  self.table_widget.setItem(row, 0, QTableWidgetItem(f"{dist[0]:.3f} rad"))  # 远-近夹角
                  self.table_widget.setItem(row, 1, QTableWidgetItem(f"{dist[1]:.3f} rad"))  # 中-近夹角
                  self.table_widget.setItem(row, 2, QTableWidgetItem(f"{dist[2]:.3f} rad"))  # 远-中夹角
            
            QMessageBox.information(self, "计算完成", "激光相互夹角计算完成！")

        except ValueError as e:
            QMessageBox.critical(self, "输入错误", str(e))
        except Exception as e:
            QMessageBox.critical(self, "读取日志文件错误", f"错误：{e}")
