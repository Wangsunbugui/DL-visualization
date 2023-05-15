from PIL import Image
import glob
import time
import torch
import numpy as np


def pil_read(img: str):
    img = Image.open(img)
    img_arr = np.array(img)  # 转换为np
    img_tensor = torch.from_numpy(img_arr)  # 转换为tensor  uint8
    return img_tensor


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'函数{func.__name__}运行了{end - start}秒')

    return wrapper


def map_paste_process(epoch, rows: int, cols: int, output_dir: str, input_dir: str, loss: float):
    # 与训练函数搭配使用 epoch用来命名拼完的特征图
    # rows cols 分别为行数列数
    # output_dir 输出图像路径
    # input_dir  输入图像路径
    # 使用例：map_paste_process(epoch, 2, 3, 'feature_map拼图', 'feature_map_pretrained', epoch_training_loss / num_batches)

    # 加载列表中的所有图像
    images = [Image.open(file) for file in glob.glob(f'feature_map/{input_dir}/*.png')]
    # 确定每个图像的大小
    image_width, image_height = images[0].size
    # 确定行数和列数
    # rows = 6
    # cols = 6
    # 创建具有白色背景的新图像
    canvas_width = cols * image_width
    canvas_height = rows * image_height
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    # 将图像粘贴到画布中
    i = 0  # 当前图像索引
    x = 0  # x 当前位置坐标
    y = 0  # y 当前位置坐标
    for row in range(rows):
        for col in range(cols):
            # 将当前图像粘贴到当前位置
            canvas.paste(images[i], (x, y))
            # 更新下一个图像的索引x和y
            i += 1
            x += image_width
            if i >= len(images):
                break
        if i >= len(images):
            break
        # 更新下一行图像的x和y
        x = 0
        y += image_height
    # 保存图片
    canvas.save(f'feature_map/{output_dir}/特征图集合{loss}.png')
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    print('拼图保存成功')
