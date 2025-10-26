#!/usr/bin/env python
# coding=utf-8
"""
简化版GPU显存监控 - 无需邮箱配置
当空闲显存达到阈值时，发出系统通知并打印提醒
"""
import os
import time
import subprocess
from datetime import datetime

def get_gpu_info():
    """获取GPU显存信息"""
    try:
        result = os.popen(
            'nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu '
            '--format=csv,noheader,nounits'
        ).read().strip()
        
        if not result:
            return None
        
        total, used, free, util, temp = map(int, result.split(','))
        return {
            'total': total,
            'used': used,
            'free': free,
            'util': util,
            'temp': temp
        }
    except Exception as e:
        print(f"❌ 获取GPU信息失败: {e}")
        return None


def send_notification(title, message):
    """发送系统通知（Linux桌面）"""
    try:
        subprocess.run(['notify-send', title, message], check=False)
    except:
        pass  # 如果notify-send不可用，忽略


if __name__ == "__main__":
    # ========== 配置参数 ==========
    FREE_THRESHOLD_GB = 18  # 空闲阈值（GB），当空闲显存≥18GB时提醒
    CHECK_INTERVAL = 60     # 检查间隔（秒）
    # ==============================
    
    print("=" * 70)
    print("🔍 GPU 显存监控器")
    print("=" * 70)
    print(f"📊 监控目标: NVIDIA GeForce RTX 3090")
    print(f"📏 空闲阈值: {FREE_THRESHOLD_GB} GB")
    print(f"⏱️  检查间隔: {CHECK_INTERVAL} 秒")
    print("=" * 70)
    print()
    
    try:
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info = get_gpu_info()
            
            if info is None:
                time.sleep(CHECK_INTERVAL)
                continue
            
            free_gb = info['free'] / 1024  # 转换为 GB
            used_gb = info['used'] / 1024
            total_gb = info['total'] / 1024
            
            # 打印当前状态
            status = "✅ 可用" if free_gb >= FREE_THRESHOLD_GB else "⏳ 等待"
            print(f"[{now}] {status} | "
                  f"空闲: {free_gb:.1f}GB | "
                  f"已用: {used_gb:.1f}GB / {total_gb:.1f}GB | "
                  f"GPU利用率: {info['util']}% | "
                  f"温度: {info['temp']}°C", 
                  end='\r')
            
            # 检查是否达到阈值
            if free_gb >= FREE_THRESHOLD_GB:
                print()  # 换行
                print("=" * 70)
                print("🎉 GPU 显存已空闲！")
                print("=" * 70)
                print(f"⏰ 时间: {now}")
                print(f"💾 空闲显存: {free_gb:.2f} GB (≥ {FREE_THRESHOLD_GB} GB)")
                print(f"🌡️  温度: {info['temp']}°C")
                print(f"📊 GPU利用率: {info['util']}%")
                print("=" * 70)
                print()
                print("✅ 可以开始训练了！运行以下命令：")
                print()
                print("   cd /home/zhuwei/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion")
                print("   bash train_full.sh")
                print()
                print("=" * 70)
                
                # 发送系统通知
                send_notification(
                    "GPU显存已空闲",
                    f"空闲显存: {free_gb:.1f}GB\n可以开始训练了！"
                )
                
                # 发出蜂鸣声（如果终端支持）
                print("\a" * 3)
                
                break  # 提醒后退出
            
            time.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n监控已停止")



