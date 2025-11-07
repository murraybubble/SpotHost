import cv2 as cv
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint

class CropDialog(QDialog):
    def __init__(self, parent=None, image=None):
        super(CropDialog, self).__init__(parent)
        self.setWindowTitle("裁切图像")
        self.image = image.copy() if image is not None else None
        self.original_image = image.copy() if image is not None else None
        self.roi = None
        self.is_selecting = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 图像显示区域
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(800, 600)
        self.update_display()
        layout.addWidget(self.label)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("重置选择")
        self.reset_btn.clicked.connect(self.reset_selection)
        btn_layout.addWidget(self.reset_btn)
        
        self.confirm_btn = QPushButton("确认裁切")
        self.confirm_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.confirm_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
        
    def update_display(self):
        if self.image is None:
            return
            
        # 创建显示用图像
        display_img = self.image.copy()
        
        # 绘制选择框
        if self.is_selecting and not self.start_point.isNull() and not self.end_point.isNull():
            cv.rectangle(display_img, 
                        (self.start_point.x(), self.start_point.y()),
                        (self.end_point.x(), self.end_point.y()),
                        (0, 255, 0), 2)
        
        # 转换为QImage并显示
        if len(display_img.shape) == 2:
            qimg = QImage(display_img.data, display_img.shape[1], display_img.shape[0], 
                         display_img.strides[0], QImage.Format_Grayscale8)
        else:
            img_rgb = cv.cvtColor(display_img, cv.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], 
                         img_rgb.strides[0], QImage.Format_RGB888)
                         
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.label.width(), self.label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(pixmap)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.label.underMouse():
            # 转换坐标到图像坐标系
            label_rect = self.label.rect()
            pixmap = self.label.pixmap()
            
            if pixmap.isNull():
                return
                
            # 计算缩放比例
            scale_x = pixmap.width() / label_rect.width()
            scale_y = pixmap.height() / label_rect.height()
            
            # 计算图像上的坐标
            pos = event.pos()
            img_x = int(pos.x() * scale_x)
            img_y = int(pos.y() * scale_y)
            
            # 确保坐标在图像范围内
            img_x = max(0, min(img_x, self.image.shape[1] - 1))
            img_y = max(0, min(img_y, self.image.shape[0] - 1))
            
            self.start_point = QPoint(img_x, img_y)
            self.end_point = QPoint(img_x, img_y)
            self.is_selecting = True
            
    def mouseMoveEvent(self, event):
        if self.is_selecting and self.label.underMouse():
            # 转换坐标到图像坐标系
            label_rect = self.label.rect()
            pixmap = self.label.pixmap()
            
            if pixmap.isNull():
                return
                
            scale_x = pixmap.width() / label_rect.width()
            scale_y = pixmap.height() / label_rect.height()
            
            pos = event.pos()
            img_x = int(pos.x() * scale_x)
            img_y = int(pos.y() * scale_y)
            
            img_x = max(0, min(img_x, self.image.shape[1] - 1))
            img_y = max(0, min(img_y, self.image.shape[0] - 1))
            
            self.end_point = QPoint(img_x, img_y)
            self.update_display()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            # 确保起点小于终点，便于后续处理
            if self.start_point.x() > self.end_point.x():
                self.start_point, self.end_point = self.end_point, self.start_point
            if self.start_point.y() > self.end_point.y():
                temp = self.start_point.y()
                self.start_point.setY(self.end_point.y())
                self.end_point.setY(temp)
            
            # 确保选择区域有一定大小
            if (self.end_point.x() - self.start_point.x() < 10 or 
                self.end_point.y() - self.start_point.y() < 10):
                self.start_point = QPoint()
                self.end_point = QPoint()
                self.update_display()
                QMessageBox.warning(self, "警告", "选择区域太小，请重新选择")
            else:
                self.roi = (self.start_point.y(), self.end_point.y(), 
                           self.start_point.x(), self.end_point.x())
                
    def reset_selection(self):
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.roi = None
        if self.original_image is not None:
            self.image = self.original_image.copy()
        self.update_display()
        
    def get_cropped_image(self):
        if self.roi is None or self.original_image is None:
            return None
        y1, y2, x1, x2 = self.roi
        return self.original_image[y1:y2, x1:x2]