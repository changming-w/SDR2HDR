import os
import time
import re
from datetime import datetime
import numpy as np

# ===================== 配置参数（根据实际情况修改）=====================
LOG_PATH = "./experiments/sr_resnet_2x_split_plus/log/train.log"  # 训练日志路径
TOTAL_ITERS = 50000  # 总迭代次数（与配置文件一致）
REFRESH_INTERVAL = 3  # 刷新间隔（秒）
HISTORY_SIZE = 50  # 展示最近多少个损失值的趋势
# ======================================================================

class TrainMonitor:
    def __init__(self):
        self.log_path = LOG_PATH
        self.total_iters = TOTAL_ITERS
        self.history = {
            "iter": [],
            "loss": [],
            "lr": []
        }
        self.latest_val = None  # 最新验证指标
        self.last_iter = 0      # 最后记录的迭代数
        self.start_time = time.time()

    def check_log_file(self):
        """检查日志文件是否存在"""
        if not os.path.exists(self.log_path):
            print(f"[错误] 日志文件不存在: {self.log_path}")
            print("请确认训练已启动，或修改LOG_PATH为正确的日志路径")
            return False
        return True

    def parse_log_line(self, line):
        """解析单行动日志，提取关键信息"""
        data = None
        # 匹配训练迭代日志（例如：[iter: 1000, lr: 0.0001998] l_pix: 0.2803）
        train_match = re.search(
            r"iter:\s*(\d+),\s*lr:\s*\(([0-9.e-]+)", line
        )
        loss_match = re.search(r"l_pix:\s*([0-9.e-]+)", line)

        if train_match and loss_match:
            iter_num = int(train_match.group(1).replace(",", ""))
            lr = float(train_match.group(2))
            loss = float(loss_match.group(1))
            data = {
                "type": "train",
                "iter": iter_num,
                "lr": lr,
                "loss": loss
            }

        # 匹配验证日志（例如：[val] PSNR: 28.5641 dB, SSIM: 0.8923 (iter: 1000)）
        val_match = re.search(
            r"PSNR:\s*([0-9.]+)\s*dB,\s*SSIM:\s*([0-9.]+).*iter:\s*(\d+)", line
        )
        if val_match:
            psnr = float(val_match.group(1))
            ssim = float(val_match.group(2))
            iter_num = int(val_match.group(3).replace(",", ""))
            data = {
                "type": "val",
                "iter": iter_num,
                "psnr": psnr,
                "ssim": ssim
            }

        return data

    def update_history(self, data):
        """更新训练历史数据"""
        if data["type"] == "train" and data["iter"] > self.last_iter:
            self.last_iter = data["iter"]
            # 保持历史数据量不超过HISTORY_SIZE
            if len(self.history["iter"]) >= HISTORY_SIZE:
                for key in self.history:
                    self.history[key].pop(0)
            self.history["iter"].append(data["iter"])
            self.history["loss"].append(data["loss"])
            self.history["lr"].append(data["lr"])

        elif data["type"] == "val":
            self.latest_val = data

    def calculate_eta(self):
        """计算预计剩余时间"""
        if not self.history["iter"]:
            return "未知"
        elapsed_time = time.time() - self.start_time
        iter_speed = self.last_iter / elapsed_time if elapsed_time > 0 else 0
        remaining_iters = self.total_iters - self.last_iter
        if iter_speed <= 0:
            return "未知"
        remaining_seconds = int(remaining_iters / iter_speed)
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def print_progress(self):
        """打印实时训练进度"""
        os.system("cls" if os.name == "nt" else "clear")  # 清空屏幕
        print("=" * 80)
        print(f"📊 BasicSR 训练监控 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 训练进度
        if self.history["iter"]:
            progress = (self.last_iter / self.total_iters) * 100
            bar_length = 50
            filled_length = int(bar_length * progress // 100)
            progress_bar = f"[{'█' * filled_length}{' ' * (bar_length - filled_length)}]"

            print(f"🚀 训练进度: {progress:.2f}% {progress_bar}")
            print(f"🔄 迭代次数: {self.last_iter:,}/{self.total_iters:,}")
            print(f"⏰ 预计剩余时间: {self.calculate_eta()}")
            print(f"📉 当前学习率: {self.history['lr'][-1]:.6f}")
            print(f"💥 当前L1损失: {self.history['loss'][-1]:.6f}")

        # 损失趋势图
        if len(self.history["iter"]) >= 2:
            print("\n📈 最近损失趋势（按迭代顺序）:")
            min_loss = min(self.history["loss"])
            max_loss = max(self.history["loss"])
            loss_range = max_loss - min_loss if max_loss > min_loss else 0.1
            trend = []
            for loss in self.history["loss"]:
                # 归一化损失值到10个字符长度的进度条
                normalized = (loss - min_loss) / loss_range
                bar = "■" * int((1 - normalized) * 10) + "□" * int(normalized * 10)
                trend.append(f"{loss:.4f} [{bar}]")
            # 每行显示5个趋势点
            for i in range(0, len(trend), 5):
                print("   " + "  ".join(trend[i:i+5]))

        # 验证指标
        if self.latest_val:
            print(f"\n✅ 最新验证（迭代 {self.latest_val['iter']:,}）:")
            print(f"   PSNR: {self.latest_val['psnr']:.4f} dB")
            print(f"   SSIM: {self.latest_val['ssim']:.4f}")

        print("\n" + "-" * 80)
        print(f"⌛ 下次刷新: {REFRESH_INTERVAL}秒后（按Ctrl+C退出）")

    def run(self):
        """启动监控主循环"""
        if not self.check_log_file():
            return

        print("监控启动成功！正在等待训练日志更新...")
        print(f"日志路径: {self.log_path}")
        print(f"总迭代次数: {self.total_iters:,}")
        print("-" * 80)

        # 打开日志文件并定位到末尾
        with open(self.log_path, "r", encoding="utf-8") as f:
            f.seek(0, 2)  # 移动到文件末尾
            try:
                while True:
                    line = f.readline()
                    if line:
                        data = self.parse_log_line(line)
                        if data:
                            self.update_history(data)
                    else:
                        # 日志无更新，刷新显示并等待
                        self.print_progress()
                        time.sleep(REFRESH_INTERVAL)
            except KeyboardInterrupt:
                print("\n" + "-" * 80)
                print("监控已手动停止！")

if __name__ == "__main__":
    monitor = TrainMonitor()
    monitor.run()