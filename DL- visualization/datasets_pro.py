import torch
import torchvision
from torch.utils.data import DataLoader


def mnist_load(batch_size, num_workers):
    # 数据增强方法
    transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.5,), (0.5,))])

    # MNIST数据集
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
    # valid_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
    # test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformImg)

    # MNIST数据集加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True, persistent_workers=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    #                               pin_memory=True,
    #                               drop_last=False, persistent_workers=True)
    # test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=num_workers)
    print("MNIST数据集加载完毕!")
    return train_dataloader


def alex_load():
    pass
# 初始化alex
# AlexNet = AlexNet()
# AlexNet = AlexNet.cuda()
# optim_alex = torch.optim.SGD(AlexNet.parameters(), lr=0.001, momentum=0.9)

    # 定义alexnet数据增强和归一化的转换方法
    # alex_transform_train = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomResizedCrop(224),  # 随机裁剪到指定大小并缩放回原始大小
    #     torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    #     torchvision.transforms.ToTensor(),  # 将PIL Image转换为张量
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    # ])

    # alex_transform_test = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(256),  # 缩放到指定大小
    #     torchvision.transforms.CenterCrop(224),  # 中心裁剪到指定大小
    #     torchvision.transforms.ToTensor(),  # 将PIL Image转换为张量
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    # ])

    # alexnet数据集CIFAR10
    # alex_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=alex_transform_train)
    # alex_train_loader = torch.utils.data.DataLoader(alex_train_set, batch_size=64, shuffle=True, num_workers=2)

    # alex_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=alex_transform_test)
    # alex_test_loader = torch.utils.data.DataLoader(alex_test_set, batch_size=30, shuffle=False, num_workers=2)
    # # alex数据集（自定义）
    # train_dataset_alex = torchvision.datasets.ImageFolder(root='data/archive/dataset/test_set',
    #                                                       transform=alex_transform)
    # # alex数据集加载器（自定义）
    # train_dataloader_alex = DataLoader(train_dataset_alex, batch_size=batch_size, shuffle=True,
    #                                    num_workers=num_workers, pin_memory=True,
    #                                    drop_last=True, persistent_workers=True)
