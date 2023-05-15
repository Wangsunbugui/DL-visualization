import torch


def para_classify(module: torch.nn.Module, param_list: list):
    # 参数为torch模型，与用来存放模型参数的列表
    for index, (name, param) in enumerate(module.named_parameters()):
        if 'conv' in name and 'bias' not in name:
            for i in range(param.shape[0]):  # 过滤器
                for j in range(param.shape[1]):  # 卷积核
                    param_list.append(param[i][j])
    return param_list

# 完整版
# def para_classify(module: torch.nn.Module, param_dic: dict):
#     # 参数为torch模型，与存放模型参数的字典
#     # 使用例： param_dic = para_classify(module=LeNet5, param_dic=param_dic)
#     # 该函数会破坏张量梯度 使用了tensor.detach().numpy()方法，大概只能在epoch之间使用
#     # 命名规范：f过滤器 k卷积核(矩阵) b偏置
#
#     for index, (name, param) in enumerate(module.named_parameters()):
#         # print(f'第{index + 1}层网络', name)
#         if 'conv' in name:
#             if 'bias' in name:
#                 # 对偏置项进行操作
#                 print(param)
#                 param_dic[name] = param
#             else:
#                 # 对权重矩阵操作
#                 for i in range(param.shape[0]):  # 过滤器
#                     for j in range(param.shape[1]):  # 卷积核
#                         print(f'第{i + 1}/{param.shape[0]}个过滤器的第{j + 1}/{param.shape[1]}个卷积核')
#                         print(param[i][j])
#                         param_dic[f'{name}_f{i}k{j}'] = param[i][j]
#         elif 'fc' in name:  # 全连接层
#             if 'bias' in name:
#                 # print(param)
#                 param_dic[f'{name}'] = param
#             else:
#                 for i in range(param.shape[0]):
#                     for j in range(param.shape[1]):
#                         # print(f'第{index}层网络的第{i + 1}个节点的第{j + 1}个权重:')
#                         # print(param[i][j])
#                         param_dic[f'{name}_i{i}j{j}'] = param[i][j]
#     return param_dic
