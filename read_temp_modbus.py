import csv
import sys
import time
from datetime import datetime
import minimalmodbus
import serial


def read_register(instr, addr, functioncode=3, signed=False):
    # Read 1 register as 16-bit int
    if functioncode == 3:
        val = instr.read_register(addr, number_of_decimals=0, functioncode=3, signed=signed)
    elif functioncode == 4:
        val = instr.read_register(addr, number_of_decimals=0, functioncode=4, signed=signed)
    else:
        raise ValueError("Unsupported function")
    return val


def main():
    # 配置参数 - 支持多个温度控制器
    CONFIG = {
        'port': "COM3",              # 串口端口
        'baud': 38400,              # 波特率
        'bytesize': 8,              # 数据位
        'parity': 'N',              # 校验位 (N=无, E=偶, O=奇)
        'stopbits': 1,              # 停止位
        'timeout': 1.0,             # 串口超时时间(秒)
        'interval': 1.0,            # 读取间隔(秒)
        'scale': 0.1,               # 缩放因子 (254 -> 25.4°C)
        'signed': False,            # 是否为有符号整数
        'csv': "temperature_log_multi.csv",  # 输出CSV文件名
        'once': False               # 是否只读取一次就退出
    }
    
    # 温度控制器配置 - 修改这里来配置您的设备
    DEVICES = [
        {'slave_id': 1, 'register': 0, 'function': 3, 'name': '温控器1'},
        {'slave_id': 1, 'register': 1, 'function': 3, 'name': '温控器2'},
        {'slave_id': 1, 'register': 2, 'function': 3, 'name': '温控器3'},
        # {'slave_id': 1, 'register': 3, 'function': 3, 'name': '温控器4'},
        # {'slave_id': 1, 'register': 4, 'function': 3, 'name': '温控器5'},
        {'slave_id': 1, 'register': 5, 'function': 3, 'name': '温控器6'},
        {'slave_id': 1, 'register': 6, 'function': 3, 'name': '温控器7'},
        # {'slave_id': 1, 'register': 7, 'function': 3, 'name': '温控器8'},
    ]
    
    # 显示当前配置信息
    print("=== Modbus配置信息 ===")
    print(f"串口端口: {CONFIG['port']}")
    print(f"波特率: {CONFIG['baud']}")
    print(f"缩放因子: {CONFIG['scale']}")
    print(f"设备数量: {len(DEVICES)}")
    for i, device in enumerate(DEVICES, 1):
        print(f"  设备{i}: ID={device['slave_id']}, 寄存器={device['register']}, 名称={device['name']}")
    print("=====================")
    
    # 准备CSV文件
    csv_path = CONFIG['csv']
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    # 写入表头（如果文件为空）
    if csv_file.tell() == 0:
        writer.writerow(["timestamp", "device_name", "slave_id", "port", "register", "function", "raw_value", "temperature_celsius"])

    try:
        while True:
            ts = datetime.now().isoformat(timespec="seconds")
            print(f"\n[{ts}] 开始读取所有设备...")
            
            # 循环读取每个设备
            for device in DEVICES:
                try:
                    # 为每个设备创建单独的仪器实例
                    instr = minimalmodbus.Instrument(CONFIG['port'], device['slave_id'], mode=minimalmodbus.MODE_RTU)
                    instr.serial.baudrate = CONFIG['baud']
                    instr.serial.bytesize = CONFIG['bytesize']
                    instr.serial.parity = {"N": serial.PARITY_NONE, "E": serial.PARITY_EVEN, "O": serial.PARITY_ODD}[CONFIG['parity']]
                    instr.serial.stopbits = CONFIG['stopbits']
                    instr.serial.timeout = CONFIG['timeout']
                    instr.clear_buffers_before_each_transaction = True
                    
                    # 读取寄存器
                    raw = read_register(instr, device['register'], functioncode=device['function'], signed=CONFIG['signed'])
                    temperature = raw * CONFIG['scale']
                    
                    # 写入CSV和控制台输出
                    writer.writerow([ts, device['name'], device['slave_id'], CONFIG['port'], device['register'], device['function'], raw, temperature])
                    print(f"  {device['name']} (ID:{device['slave_id']}): {raw} -> {temperature:.1f}°C")
                    
                except Exception as e:
                    print(f"  {device['name']} (ID:{device['slave_id']}): ERROR - {e}")
                    writer.writerow([ts, device['name'], device['slave_id'], CONFIG['port'], device['register'], device['function'], "ERROR", str(e)])
            
            csv_file.flush()
            
            if CONFIG['once']:
                break
            
            # print(f"等待 {CONFIG['interval']} 秒")
            time.sleep(CONFIG['interval'])
            
    except KeyboardInterrupt:
        print("\n停止读取")
    finally:
        csv_file.close()
        print("CSV文件已保存")


if __name__ == "__main__":
    main()