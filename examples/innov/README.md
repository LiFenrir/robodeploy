# Innov Arm 脚本

## 数据采集

```bash
bash record_innov.sh    # 双臂本体示教采集
```

## 工具

```bash
python set_motor_zero.py --left_port /dev/ttyACM0 --right_port /dev/ttyACM1   # 双臂归零
python set_motor_zero.py --port /dev/ttyACM0                                   # 单臂归零
```
