import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


def img_norm(image_tensor):
    # 缩到0-1范围
    return (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min()) * 1500 - 500


def lenet_feature_map_process(model: torch.nn.Module, input_image: str):
    # model 选择模型
    # input_image '输入图像'的路径
    # 使用例：lenet_feature_map_process(LeNet5, '数字4.png')

    # 读取处理图片
    image = Image.open(input_image)
    image = image.convert('L')
    image_tensor = torch.from_numpy(np.array(image))  # 转换为tensor  uint8
    image_tensor = image_tensor.unsqueeze(0).type(torch.float32)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()  # torch.Size([1, 28, 28])

    def forward_hook(module, input, output):
        for i in range(output.shape[0]):  # output.shape[1] 'viridis'
            plt.imshow(output[i].cpu().detach(), cmap='gray', interpolation='spline16')
            plt.axis('off')
            plt.savefig(f"LeNet5_pic/{module.name}/{module.name}_{i}.png", bbox_inches='tight', dpi=10)
        plt.close()
        # torchvision.utils.save_image(output[i].cpu().detach(), f"LeNet5_pic/LeNet_{module.name}_{i}.png")
        return img_norm(output)

    handles = []
    for child in model.children():
        if not isinstance(child, torch.nn.Linear):
            handle = child.register_forward_hook(hook=forward_hook)
            handles.append(handle)

    output = model(image_tensor)
    for handle in handles:
        handle.remove()

    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    percentage = torch.softmax(output, dim=1) * 100
    formatted = ' '.join(['%.4f' % number for number in percentage[0]])
    print('预测概率为：', formatted)
    print('预测值为：', predicted.item())


def lenet_filter_map_process(param_list: list):
    # 参数要求：存放卷积核参数的列表,每一个元素为一个5x5的tensor值
    start_time = time.time()
    for i in range(len(param_list)):
        if i >= 13 and i % 13 == 0:
            plt.close()
        plt.imshow(param_list[i].cpu().detach(), cmap='gray', interpolation='spline16')
        plt.axis('off')
        plt.savefig(f"LeNet5_pic/filter/kernel{i + 1}.png", bbox_inches='tight', dpi=20)
    plt.close()
    print('卷积核图消耗时间: %.3f s' % (time.time() - start_time))


def alex_feature_map_process(model: torch.nn.Module, input_image: str, conv_layer: int, output_image_num: int):
    # model 选择模型
    # input_image '输入图像'的路径
    # conv_layer 哪一层卷积层处理 1-5
    # output_image 输出特征图的数量

    image = torchvision.io.read_image(input_image)
    image_tensor = image.unsqueeze(0).float()
    image_tensor = image_tensor.cuda()

    # 获取第一层卷积后的特征图（64个）
    feature_maps_1 = model.features[0](image_tensor)
    # 获取第二层卷积后的特征图（192个）
    feature_maps_2 = model.features[3](feature_maps_1)
    # 获取第三层卷积后的特征图（384个）
    feature_maps_3 = model.features[6](feature_maps_2)
    # 获取第四层卷积后的特征图（256个）
    feature_maps_4 = model.features[8](feature_maps_3)
    # 获取第五层卷积后的特征图（256个）
    feature_maps_5 = model.features[10](feature_maps_4)

    torch.save(feature_maps_1, "feature_map/feature_map_save_pth/feature_maps_1.pth")
    torch.save(feature_maps_2, "feature_map/feature_map_save_pth/feature_maps_2.pth")
    torch.save(feature_maps_3, "feature_map/feature_map_save_pth/feature_maps_3.pth")
    torch.save(feature_maps_4, "feature_map/feature_map_save_pth/feature_maps_4.pth")
    torch.save(feature_maps_5, "feature_map/feature_map_save_pth/feature_maps_5.pth")

    feature_maps_load = torch.load(f"feature_map/feature_map_save_pth/feature_maps_{conv_layer}.pth")

    for i in range(output_image_num):
        feature_map_load = feature_maps_load[0][i]
        # 将特征图保存为png文件
        if i % 10 == 0:
            print('保存png图片成功')
        torchvision.utils.save_image(feature_map_load, f"feature_map/feature_map_untrained/feature_map_load_{i}.png")
