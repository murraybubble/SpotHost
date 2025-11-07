import serial
from dataclasses import dataclass
from typing import Optional, List, Tuple  # 关键：导入Tuple类用于类型标注
from PyQt5.QtCore import QThread, pyqtSignal
from serial.tools.list_ports import comports

# 协议常量（严格遵循《905通讯协议二.pdf》定义）
@dataclass(frozen=True)
class ProtocolConst:
    # 基础通讯参数（、）
    BAUDRATE = 115200       # 固定波特率115200bps
    BYTESIZE = serial.EIGHTBITS  # 8位数据位
    PARITY = serial.PARITY_NONE  # 无奇偶校验
    STOPBITS = serial.STOPBITS_ONE  # 1位停止位
    # 命令字（表3）
    CMD_STANDBY = 0x00           # 待机命令（停止连续测距）
    CMD_SINGLE_MEASURE = 0x01    # 单次测距命令
    CMD_CONTINUOUS_MEASURE = 0x02# 连续测距命令
    # 响应帧参数（表4、表5）
    RESP_CMD_CONTINUOUS = 0x02   # 连续测距响应命令字
    LEN_MEASURE = 14             # 测距响应帧总长度（14字节）
    LEN_STANDBY = 6              # 待机响应帧总长度（6字节）
    # 连续测距频率（表3备注）
    FREQ_4HZ = 0x00              # 默认4Hz（最快频率）
    FREQ_1HZ = 0x01              # 1Hz（低速频率）
    # 测距数据字节索引（表5 D9-D0定义）
    IDX_D9 = 3       # D9：标志字节（判断是否有目标）
    IDX_D8 = 4       # D8：首目标距离高位
    IDX_D7 = 5       # D7：首目标距离中位
    IDX_D6 = 6       # D6：首目标距离低位
    IDX_D5 = 7       # D5：末目标距离高位
    IDX_D4 = 8       # D4：末目标距离中位
    IDX_D3 = 9       # D3：末目标距离低位
    IDX_APD_VOLT = 11  # D1：APD高压值
    IDX_APD_TEMP = 12  # D0：APD温度

# 测量数据模型（封装表5所有有效字段）
@dataclass
class MeasureResult:
    valid: bool = False          # 数据是否有效
    distance_first: float = 0.0  # 首目标距离（米，按除以10换算）
    distance_last: float = 0.0   # 末目标距离（米）
    has_target: bool = False     # 是否有目标（D9=0x28/0x38，）
    apd_voltage: int = 0         # APD高压值（V，）
    apd_temperature: int = 0     # APD温度（℃，）

class DistanceMeterManager:
    """905测距机核心功能抽象（非UI类，仅依赖协议逻辑）"""
    def __init__(self):
        self.ser: Optional[serial.Serial] = None
        self.connected: bool = False  # 连接状态
        self.in_continuous_mode: bool = False  # 是否连续测距模式

    @staticmethod
    def get_available_ports() -> List[str]:
        """获取系统所有可用串口（静态方法，方便调用）"""
        return [port.device for port in comports() if port.device]

    def connect(self, port: str, timeout: float = 0.5) -> Tuple[bool, str]:
        """
        连接测距机（按、初始化串口）
        :return: (连接成功标识, 提示信息)
        """
        if self.connected:
            self.disconnect()
        
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=ProtocolConst.BAUDRATE,
                bytesize=ProtocolConst.BYTESIZE,
                parity=ProtocolConst.PARITY,
                stopbits=ProtocolConst.STOPBITS,
                timeout=timeout,
                xonxoff=False,
                rtscts=False
            )
            self.connected = self.ser.is_open
            return self.connected, f"成功连接测距机（端口：{port}）"
        except serial.SerialException as e:
            return False, f"连接失败：{str(e)}"
        except Exception as e:
            return False, f"未知错误：{str(e)}"

    def _calc_checksum(self, data: bytes) -> int:
        """计算异或校验码（遵循、：除校验字节外所有字节异或）"""
        chk = 0
        for byte in data:
            chk ^= byte
        return chk

    def _build_cmd_frame(self, cmd: int, params: List[int] = [0x00, 0x00]) -> bytes:
        """构建命令帧（遵循、发送格式：STX0+CMD+LEN+DATA+CHK）"""
        len_param = len(params)
        frame = bytes([0x55, cmd, len_param]) + bytes(params)  # 0x55为帧头（）
        frame += bytes([self._calc_checksum(frame)])  # 追加校验码
        return frame

    def _parse_measure_frame(self, frame: bytes) -> MeasureResult:
        """解析测距响应帧（严格按表5 D8-D6/D5-D3取数+单位换算）"""
        result = MeasureResult()
        # 基础校验：帧长度、帧头、校验码（、）
        if len(frame) != ProtocolConst.LEN_MEASURE:
            return result
        if frame[0] != 0x55:  # 帧头必须为0x55（）
            return result
        if self._calc_checksum(frame[:-1]) != frame[-1]:  # 校验码不匹配（）
            return result

        # 解析目标状态（D9标志字节，）
        d9 = frame[ProtocolConst.IDX_D9]
        result.has_target = d9 in [0x28, 0x38]  # 0x28/0x38表示有目标

        # 解析首目标距离（D8-D6，3字节大端，单位0.1m→米，）
        d8 = frame[ProtocolConst.IDX_D8]
        d7 = frame[ProtocolConst.IDX_D7]
        d6 = frame[ProtocolConst.IDX_D6]
        dist_first_raw = (d8 << 16) | (d7 << 8) | d6
        result.distance_first = dist_first_raw / 10.0  # 按协议除以10转换为米

        # 解析末目标距离（D5-D3，3字节大端，）
        d5 = frame[ProtocolConst.IDX_D5]
        d4 = frame[ProtocolConst.IDX_D4]
        d3 = frame[ProtocolConst.IDX_D3]
        dist_last_raw = (d5 << 16) | (d4 << 8) | d3
        result.distance_last = dist_last_raw / 10.0

        # 解析APD参数（D1=高压，D0=温度，）
        result.apd_voltage = frame[ProtocolConst.IDX_APD_VOLT]
        result.apd_temperature = frame[ProtocolConst.IDX_APD_TEMP]

        result.valid = True
        return result

    def send_standby_cmd(self) -> Tuple[bool, str]:
        """发送待机命令（停止连续测距，表3命令字0x00）"""
        if not self.connected:
            return False, "未连接测距机"
        
        try:
            cmd_frame = self._build_cmd_frame(ProtocolConst.CMD_STANDBY)
            self.ser.write(cmd_frame)
            # 验证待机响应（6字节帧，命令字一致，、）
            resp_frame = self.ser.read(ProtocolConst.LEN_STANDBY)
            if len(resp_frame) == ProtocolConst.LEN_STANDBY and resp_frame[1] == ProtocolConst.CMD_STANDBY:
                self.in_continuous_mode = False
                return True, "已进入待机模式"
            return False, "待机命令无响应"
        except Exception as e:
            return False, f"待机命令失败：{str(e)}"

    def single_measure(self) -> Tuple[Optional[MeasureResult], str]:
        """单次测距（表3命令字0x01，即时返回结果）"""
        if not self.connected:
            return None, "未连接测距机"
        if self.in_continuous_mode:
            # 连续模式下需先切换为待机（主从通讯逻辑）
            success, msg = self.send_standby_cmd()
            if not success:
                return None, f"切换待机失败：{msg}"
        
        try:
            # 命令参数：20 00（表3单次测距示例代码）
            cmd_frame = self._build_cmd_frame(ProtocolConst.CMD_SINGLE_MEASURE, [0x20, 0x00])
            self.ser.write(cmd_frame)
            # 读取14字节测距帧（表5总长度）
            resp_frame = self.ser.read(ProtocolConst.LEN_MEASURE)
            result = self._parse_measure_frame(resp_frame)
            if result.valid:
                return result, "单次测距成功"
            return None, "未获取有效测距数据"
        except Exception as e:
            return None, f"单次测距失败：{str(e)}"

    def start_continuous_measure(self, freq: int = ProtocolConst.FREQ_4HZ) -> Tuple[bool, str]:
        """
        启动连续测距（表3命令字0x02）
        :param freq: 频率（ProtocolConst.FREQ_4HZ=4Hz，FREQ_1HZ=1Hz）
        :return: (启动成功标识, 提示信息)
        """
        if not self.connected:
            return False, "未连接测距机"
        if freq not in [ProtocolConst.FREQ_4HZ, ProtocolConst.FREQ_1HZ]:
            freq = ProtocolConst.FREQ_4HZ
        
        try:
            # 命令参数：20 + 频率（表3连续测距示例代码：55 02 02 20 01 74）
            cmd_frame = self._build_cmd_frame(ProtocolConst.CMD_CONTINUOUS_MEASURE, [0x20, freq])
            self.ser.write(cmd_frame)
            # 验证启动：读取首次测距帧（测距状态回传逻辑）
            resp_frame = self.ser.read(ProtocolConst.LEN_MEASURE)
            if len(resp_frame) == ProtocolConst.LEN_MEASURE and resp_frame[1] == ProtocolConst.RESP_CMD_CONTINUOUS:
                self.in_continuous_mode = True
                freq_text = "4Hz" if freq == ProtocolConst.FREQ_4HZ else "1Hz"
                return True, f"已启动{freq_text}连续测距"
            return False, "连续测距命令无响应"
        except Exception as e:
            return False, f"启动连续测距失败：{str(e)}"

    def read_continuous_data(self) -> Optional[MeasureResult]:
        """读取连续测距数据（仅连续模式下有效，非阻塞，实时回传逻辑）"""
        if not self.connected or not self.in_continuous_mode:
            return None
        
        try:
            # 读取14字节测距帧（表5总长度）
            resp_frame = self.ser.read(ProtocolConst.LEN_MEASURE)
            return self._parse_measure_frame(resp_frame)
        except Exception as e:
            print(f"连续测距读取失败：{str(e)}")
            return None

    def disconnect(self) -> str:
        """断开测距机连接（先发送待机命令，确保设备安全，主从通讯逻辑）"""
        if self.connected and self.ser:
            try:
                if self.in_continuous_mode:
                    self.send_standby_cmd()
                self.ser.close()
            except Exception as e:
                return f"断开连接时出错：{str(e)}"
            finally:
                self.ser = None
                self.connected = False
                self.in_continuous_mode = False
        return "已断开测距机连接"

class ContinuousMeasureThread(QThread):
    """
    PyQt连续测距线程（避免阻塞GUI主线程）
    信号：
    - measure_signal: 发送有效测距结果（MeasureResult）
    - status_signal: 发送状态信息（str）
    - error_signal: 发送错误信息（str）
    """
    measure_signal = pyqtSignal(MeasureResult)
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, meter_manager: DistanceMeterManager, freq: int = ProtocolConst.FREQ_4HZ):
        super().__init__()
        self.meter_manager = meter_manager  # 注入测距机管理器实例
        self.freq = freq
        self.running = False  # 线程运行标识

    def run(self):
        """线程主逻辑：启动连续测距→循环读取数据→发送信号（遵循实时回传）"""
        success, msg = self.meter_manager.start_continuous_measure(self.freq)
        if not success:
            self.error_signal.emit(f"连续测距启动失败：{msg}")
            return
        
        self.running = True
        self.status_signal.emit(f"连续测距线程启动（{self._get_freq_text()}）")
        
        # 循环读取数据（4Hz对应250ms/次，10ms轮询避免遗漏数据）
        while self.running:
            result = self.meter_manager.read_continuous_data()
            if result and result.valid:
                self.measure_signal.emit(result)  # 发送有效结果
            self.msleep(10)  # 降低CPU占用

    def stop(self):
        """停止线程（安全退出，先发送待机命令）"""
        self.running = False
        self.wait()
        if self.meter_manager.connected:
            self.meter_manager.send_standby_cmd()
        self.status_signal.emit("连续测距线程已停止")

    def _get_freq_text(self) -> str:
        """将频率常量转换为文本描述（表3频率定义）"""
        return "4Hz" if self.freq == ProtocolConst.FREQ_4HZ else "1Hz"