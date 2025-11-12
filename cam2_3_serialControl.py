import serial
import serial.tools.list_ports
import time

class CameraController_1:
    """相机控制器，负责通过串口发送控制命令"""   
    def __init__(self, baudrate=115200, timeout=0.5):
        self.ser = None
        self.baudrate = baudrate
        self.timeout = timeout
        self.connected = False

    def connect(self, port=None):
        """连接到串口设备
        
        Args:
            port: 串口名称，如"COM3"或"/dev/ttyUSB0"，若为None则尝试自动连接
            
        Returns:
            bool: 连接成功返回True，否则返回False
        """
        try:
            # 自动查找串口
            if not port:
                ports = list(serial.tools.list_ports.comports())
                if not ports:
                    print("未发现可用串口")
                    return False
                # 优先选择第一个串口
                port = ports[0].device
                print(f"自动选择串口: {port}")
            
            # 关闭已存在的连接
            if self.ser and self.ser.is_open:
                self.ser.close()
                
            # 建立新连接
            self.ser = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # 检查连接状态
            if self.ser.is_open:
                self.connected = True
                print(f"成功连接到串口: {port}")
                return True
            else:
                self.connected = False
                print(f"无法打开串口: {port}")
                return False
                
        except Exception as e:
            self.connected = False
            print(f"串口连接错误: {str(e)}")
            return False

    def disconnect(self):
        """断开串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("已断开串口连接")
        self.connected = False

    def send_command(self, command):
        """发送命令到相机
        
        Args:
            command: 字节数组命令
            
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        if not self.connected or not self.ser or not self.ser.is_open:
            print("未建立串口连接，无法发送命令")
            return False
            
        try:
            # 发送命令
            self.ser.write(command)
            # 短暂延时确保命令发送完成
            time.sleep(0.05)
            print(f"发送命令成功: {[hex(b) for b in command]}")
            return True
        except Exception as e:
            print(f"命令发送失败: {str(e)}")
            return False

    def stop_focus(self):
        """发送调焦停命令"""
        # 命令: 55 AA 07 03 00 06 00 00 00 00 02 F0
        command = bytes([0x55, 0xAA, 0x07, 0x03, 0x00, 0x06, 
                        0x00, 0x00, 0x00, 0x00, 0x02, 0xF0])
        return self.send_command(command)

    def tele_focus(self):
        """发送远焦+命令"""
        # 命令: 55 AA 07 03 00 06 00 00 00 01 03 F0
        command = bytes([0x55, 0xAA, 0x07, 0x03, 0x00, 0x06, 
                        0x00, 0x00, 0x00, 0x01, 0x03, 0xF0])
        return self.send_command(command)

    def wide_focus(self):
        """发送近焦-命令"""
        # 命令: 55 AA 07 03 00 06 00 00 00 02 00 F0
        command = bytes([0x55, 0xAA, 0x07, 0x03, 0x00, 0x06, 
                        0x00, 0x00, 0x00, 0x02, 0x00, 0xF0])
        return self.send_command(command)

    def scene_compensation(self):
        """发送场景补偿命令"""
        # 命令: 55 AA 07 02 01 07 00 00 00 01 02 F0
        command = bytes([0x55, 0xAA, 0x07, 0x02, 0x01, 0x07, 
                        0x00, 0x00, 0x00, 0x01, 0x02, 0xF0])
        return self.send_command(command)

    def shutter_compensation(self):
        """发送快门补偿命令"""
        # 命令: 55 AA 07 02 01 08 00 00 00 01 0D F0
        command = bytes([0x55, 0xAA, 0x07, 0x02, 0x01, 0x08, 
                        0x00, 0x00, 0x00, 0x01, 0x0D, 0xF0])
        return self.send_command(command)

    def set_detail_gain(self, gain_value):
        """设置增强细节增益
        
        Args:
            gain_value: 增益值(0-255)
            
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        # 校验增益值范围
        if not (0 <= gain_value <= 255):
            print(f"增益值必须在0-255之间，当前值: {gain_value}")
            return False
            
        # 计算校验位(XOR)
        # 命令格式: 55 AA 07 02 02 12 00 00 00 xx XOR F0
        prefix = [0x55, 0xAA, 0x07, 0x02, 0x02, 0x12, 0x00, 0x00, 0x00]
        xor_value = 0xF0
        
        # 计算XOR校验
        checksum = xor_value
        for b in prefix:
            checksum ^= b
        checksum ^= gain_value
        
        # 构建完整命令
        command = bytes(prefix + [gain_value, checksum])
        return self.send_command(command)

    def is_connected(self):
        """检查是否已连接"""
        return self.connected and self.ser and self.ser.is_open
    

class CameraController_2:
    """相机控制器2，负责通过串口发送控制命令（遵循新通信协议）"""   
    def __init__(self, baudrate=115200, timeout=0.5):
        self.ser = None
        self.baudrate = baudrate
        self.timeout = timeout
        self.connected = False

    def connect(self, port=None):
        """连接到串口设备
        
        Args:
            port: 串口名称，如"COM3"或"/dev/ttyUSB0"，若为None则尝试自动连接
            
        Returns:
            bool: 连接成功返回True，否则返回False
        """
        try:
            # 自动查找串口
            if not port:
                ports = list(serial.tools.list_ports.comports())
                if not ports:
                    print("未发现可用串口")
                    return False
                # 优先选择第一个串口
                port = ports[0].device
                print(f"自动选择串口: {port}")
            
            # 关闭已存在的连接
            if self.ser and self.ser.is_open:
                self.ser.close()
                
            # 建立新连接
            self.ser = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # 检查连接状态
            if self.ser.is_open:
                self.connected = True
                print(f"成功连接到串口: {port}")
                return True
            else:
                self.connected = False
                print(f"无法打开串口: {port}")
                return False
                
        except Exception as e:
            self.connected = False
            print(f"串口连接错误: {str(e)}")
            return False

    def disconnect(self):
        """断开串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("已断开串口连接")
        self.connected = False

    def send_command(self, command):
        """发送命令到相机
        
        Args:
            command: 字节数组命令
            
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        if not self.connected or not self.ser or not self.ser.is_open:
            print("未建立串口连接，无法发送命令")
            return False
            
        try:
            # 发送命令
            self.ser.write(command)
            # 短暂延时确保命令发送完成
            time.sleep(0.05)
            print(f"发送命令成功: {[hex(b) for b in command]}")
            return True
        except Exception as e:
            print(f"命令发送失败: {str(e)}")
            return False

    def tele_focus(self):
        """发送调焦+命令"""
        # 功能位: 01 01 00 00，校验位: 01^01^00^00=00
        command = bytes([0x55, 0xAA, 0x01, 0x01, 0x00, 0x00, 0x00, 0xF0])
        return self.send_command(command)

    def wide_focus(self):
        """发送调焦-命令"""
        # 功能位: 01 02 00 00，校验位: 01^02^00^00=03
        command = bytes([0x55, 0xAA, 0x01, 0x02, 0x00, 0x00, 0x03, 0xF0])
        return self.send_command(command)

    def stop_focus(self):
        """发送调焦停命令"""
        # 功能位: 01 03 00 00，校验位: 01^03^00^00=02
        command = bytes([0x55, 0xAA, 0x01, 0x03, 0x00, 0x00, 0x02, 0xF0])
        return self.send_command(command)

    def set_zoom(self, zoom_level):
        """设置电子放大倍数
        
        Args:
            zoom_level: 放大倍数等级（0:1倍, 1:2倍, 2:4倍）
            
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        if zoom_level not in [0, 1, 2]:
            print(f"放大倍数等级必须为0、1或2，当前值: {zoom_level}")
            return False
            
        # 功能位: 02 05 XX 00，XX为放大等级
        data1 = 0x02
        data2 = 0x05
        data3 = zoom_level
        data4 = 0x00
        # 计算校验位
        checksum = data1 ^ data2 ^ data3 ^ data4
        command = bytes([0x55, 0xAA, data1, data2, data3, data4, checksum, 0xF0])
        return self.send_command(command)

    def set_integration_time(self, ms):
        """设置积分时间（ms）
        
        Args:
            ms: 积分时间（毫秒），实际发送值为ms×10
            
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        try:
            # 计算传输值（ms×10）
            value = int(ms * 10)
            if value < 0 or value > 0xFFFF:  # 确保在两个字节范围内
                print(f"积分时间超出范围，有效范围: 0-{0xFFFF//10}ms")
                return False
                
            # 拆分高低字节
            high_byte = (value >> 8) & 0xFF
            low_byte = value & 0xFF
            
            # 功能位: 02 0F 高字节 低字节
            data1 = 0x02
            data2 = 0x0F
            data3 = high_byte
            data4 = low_byte
            # 计算校验位
            checksum = data1 ^ data2 ^ data3 ^ data4
            command = bytes([0x55, 0xAA, data1, data2, data3, data4, checksum, 0xF0])
            return self.send_command(command)
            
        except Exception as e:
            print(f"积分时间设置错误: {str(e)}")
            return False

    def set_frame_rate(self, hz):
        """设置帧频（Hz）
        
        Args:
            hz: 帧频值，实际发送值为hz×100
            
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        try:
            # 计算传输值（hz×100）
            value = int(hz * 100)
            if value < 0 or value > 0xFFFF:  # 确保在两个字节范围内
                print(f"帧频超出范围，有效范围: 0-{0xFFFF//100}Hz")
                return False
                
            # 拆分高低字节
            high_byte = (value >> 8) & 0xFF
            low_byte = value & 0xFF
            
            # 功能位: 02 11 高字节 低字节
            data1 = 0x02
            data2 = 0x11
            data3 = high_byte
            data4 = low_byte
            # 计算校验位
            checksum = data1 ^ data2 ^ data3 ^ data4
            command = bytes([0x55, 0xAA, data1, data2, data3, data4, checksum, 0xF0])
            return self.send_command(command)
            
        except Exception as e:
            print(f"帧频设置错误: {str(e)}")
            return False

    def scene_compensation(self):
        """发送场景补偿命令"""
        # 功能位: 04 04 00 00，校验位: 04^04^00^00=00
        command = bytes([0x55, 0xAA, 0x04, 0x04, 0x00, 0x00, 0x00, 0xF0])
        return self.send_command(command)

    def is_connected(self):
        """检查是否已连接"""
        return self.connected and self.ser and self.ser.is_open
    
    