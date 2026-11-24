import os
from PIL import Image

# 固定输入输出路径（直接修改这里的路径即可）
INPUT_DIR = "./"   # 存放原始PNG图片的文件夹
OUTPUT_DIR = "./"  # 存放处理后图片的文件夹
TARGET_SIZE = (1280, 720)     # 目标尺寸：宽1280，高720
FILL_COLOR = (0, 0, 0)        # 填充颜色（黑色），如需白色可改为(255,255,255)

def resize_png(input_path, output_path):
    """将单张PNG图片调整为1280x720，保持比例并填充黑边"""
    try:
        with Image.open(input_path) as img:
            # 转换为RGBA以支持透明通道
            img = img.convert('RGBA')
            original_width, original_height = img.size

            # 计算缩放比例（保持原图比例，避免变形）
            scale = min(TARGET_SIZE[0]/original_width, TARGET_SIZE[1]/original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            # 高质量缩放
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 创建目标尺寸的画布
            canvas = Image.new('RGBA', TARGET_SIZE, (*FILL_COLOR, 255))  # 255表示不透明

            # 计算居中位置
            x = (TARGET_SIZE[0] - new_width) // 2
            y = (TARGET_SIZE[1] - new_height) // 2

            # 粘贴图片到画布中央（保留透明通道）
            canvas.paste(resized_img, (x, y), resized_img)

            # 转为RGB格式（去除透明通道，纯PNG可保留RGBA）
            canvas = canvas.convert('RGB')
            canvas.save(output_path, 'PNG')

            print(f"处理成功：{os.path.basename(input_path)} "
                  f"({original_width}x{original_height} → {new_width}x{new_height})")

    except Exception as e:
        print(f"处理失败 {os.path.basename(input_path)}：{str(e)}")

def batch_resize():
    """批量处理文件夹中的所有PNG图片"""
    # 创建输出目录（如果不存在）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 遍历输入目录中的PNG文件
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            resize_png(input_path, output_path)

    print("\n所有PNG图片处理完成！结果保存在：", OUTPUT_DIR)

if __name__ == "__main__":
    batch_resize()