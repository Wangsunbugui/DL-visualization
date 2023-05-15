import torch
from param_classify import para_classify
from All_nn_model import VGG, AlexNet
from datasets_pro import mnist_load
from featrue_map_pro import alex_feature_map_process, lenet_feature_map_process, img_norm, lenet_filter_map_process

# 设备
device = torch.device('cuda')


def lenet_initial():
    from All_nn_model import LeNet5
    start_time = time.time()
    # 模型
    LeNet5 = LeNet5()
    if torch.cuda.is_available():
        LeNet5 = LeNet5.cuda()
    # 损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_func = loss_func.cuda()
    # 优化器
    optim = torch.optim.Adam(LeNet5.parameters(), lr=0.01)
    print('LeNet5初始化完成 消耗时间: %.3f s' % (time.time() - start_time))
    return loss_func, optim, LeNet5


def lenet_train(batch_size, num_workers, Epoch, display_frequency,
                mnist_train_loader,
                LeNet5,
                optim, loss_func
                ):
    import time
    print('开始训练')
    # 对初始化模型处理输入图像
    lenet_feature_map_process(model=LeNet5, input_image='img/image.png')
    for epoch in range(Epoch):
        # 评估初始化
        epoch_training_loss = 0.0
        num_batches = 0
        count = 0
        total = 0  # 精度计算
        correct = 0
        train_time = time.time()
        for data in mnist_train_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = LeNet5(imgs)
            result_loss = loss_func(output, targets)
            optim.zero_grad()
            result_loss.backward()
            optim.step()

            # 计算损失
            epoch_training_loss += result_loss.item()
            num_batches += 1

            # 计算精度
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # 样本间隔计算
            count += batch_size
            # 每个epoch使用特定次数功能
            if 60000 / display_frequency - count < batch_size:
                print('特征图间隔时间: %.3f s' % (time.time() - train_time))
                train_time = time.time()
                print("特定功能, count=", count)
                count = 0
                print('精度:%d %%' % (100 * correct / total))
                # 重置精度计算
                total = 0
                correct = 0

                # 特征图产出
                start_time = time.time()
                lenet_feature_map_process(model=LeNet5, input_image='img/image.png')
                print('特征图消耗时间: %.3f s' % (time.time() - start_time))

                # 卷积核可视化
                param_list = []
                param_list = para_classify(LeNet5, param_list)
                lenet_filter_map_process(param_list)

                print("损失: ", epoch_training_loss / num_batches)

                print('#' * 80)
                # os.system('pause')
        print(f'批次数:{num_batches}')
        print("轮数: ", epoch + 1)
        print('> ' * 40)


if __name__ == '__main__':
    import os
    import time

    batch_size = 32  # batch_size = 16 frequency = 10 ——>特征图间隔约4s
    num_workers = 2
    Epoch = 30
    display_frequency = 20  # 展示频率(每个epoch展示次数)

    # LeNet5数据集加载
    print('启动训练')
    ini_time = time.time()
    mnist_train_loader = mnist_load(batch_size, num_workers)
    print('数据集加载消耗时间: %.3f s' % (time.time() - ini_time))

    # LeNet5初始化
    loss_func, optim, LeNet5 = lenet_initial()

    # LeNet5训练
    lenet_train(batch_size, num_workers, Epoch, display_frequency, mnist_train_loader,
                LeNet5,
                optim, loss_func)

# alex_net训练
#     for epoch in range(Epoch):  # 多次循环遍历整个数据集
#         print('当前轮数：', epoch)
#         running_loss = 0.0
#         for i, data in enumerate(alex_train_loader, 0):
#             # 获取输入；data是一个列表，包含输入和标签
#             inputs, labels = data
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#             # 梯度清零
#             optim_alex.zero_grad()
#             # 前向传播+反向传播+优化
#             outputs = AlexNet(inputs)
#             loss = loss_func(outputs, labels)
#             loss.backward()
#             optim_alex.step()
#             # 打印统计信息
#             running_loss = 0.0
#             running_loss += loss.item()
#             if i % 100 == 99:  # 每100批次打印一次
#                 print('[%d, %5d] 损失: %.3f' % (epoch + 1, i + 1, running_loss / 100))
#                 running_loss = 0.0
#                 print('训练完成')
#
#             # 测试模型
#             correct = 0
#             total = 0
#             with torch.no_grad():
#                 for data1 in alex_test_loader:
#                     images, labels = data1
#                     images = images.cuda()
#                     labels = labels.cuda()
#                     outputs = AlexNet(images)
#                     _, predicted = torch.max(outputs.data, 1)
#                     total += labels.size(0)
#                     correct += (predicted == labels).sum().item()
#
#             print('网络对10000张测试图像的准确性: %d %%' % (100 * correct / total))
#
#             # 每个类别的准确率
#             class_correct = list(0. for i in range(10))
#             class_total = list(0. for i in range(10))
#             with torch.no_grad():
#                 for data2 in alex_test_loader:
#                     images, labels = data2
#                     images = images.cuda()
#                     labels = labels.cuda()
#                     outputs = AlexNet(images)
#                     _, predicted = torch.max(outputs, 1)
#                     c = (predicted == labels).squeeze()
#                     for j in range(4):
#                         label = labels[j]
#                         class_correct[label] += c[j].item()
#                         class_total[label] += 1
#
#             for j in range(10):
#                 print('%5s的精确度 : %2d %%' % (alex_test_set.classes[j], 100 * class_correct[j] / class_total[j]))
