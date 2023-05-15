import base64
import io
import os

from PIL import Image
from flask import Flask, render_template, request, send_from_directory, jsonify

from Model_Train import lenet_train, lenet_initial
from datasets_pro import mnist_load

app = Flask(__name__)
image_folders = {
    'conv1': 6,
    'pool1': 6,
    'relu1': 6,
    'conv2': 16,
    'relu2': 16,
    'pool2': 16,
}


# 渲染页面
@app.route('/')
def index():
    return render_template('index.html')


# 前端按钮点击后开始训练
@app.route('/train', methods=['POST'])
def train():
    return {'status': 'Training started'}


# 前端手写数字画的图片传到后端来徐连
@app.route("/upload", methods=["POST"])
def upload():
    # 从 POST 请求中获取 Base64 编码的数据
    image_data = request.json["image"]

    # 将 Base64 编码的数据转换为 PNG 图像
    image_binary = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_binary))

    # 将 PNG 图像保存到文件中
    image = image.resize((28, 28))  # 调整图像大小为 28x28px
    image.save("img/image.png")

    return "Image uploaded successfully"


# 训练一个epoch 做几次特定的动作
# 特定动作里面可以有生成特征图 卷积核图
# 我写在外头是考虑到之后这些数据是可以给用户调节的
@app.route('/model_train')
def model_train():
    global mnist_train_loader
    global loss_func, optim, LeNet5
    batch_size = 16
    num_workers = 2
    Epoch = 3
    display_frequency = 10  # 展示频率(每个epoch展示次数)

    lenet_train(batch_size, num_workers, Epoch, display_frequency,
                mnist_train_loader=mnist_train_loader,
                LeNet5=LeNet5,
                optim=optim, loss_func=loss_func)


# 把特征图传输到前端对应url下
@app.route('/get_image/<folder>/<filename>')
def get_image(folder, filename):
    return send_from_directory(os.path.join('LeNet5_pic', folder), filename)


@app.route('/get_image_list')
def get_image_list():
    image_files = []
    for folder, num_images in image_folders.items():
        for i in range(num_images):
            image_files.append(f"{folder}/{folder}_{i}.png")
    # print(image_files)
    return {'images': image_files}


@app.route('/get-percentages')
def get_percentages():
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print("get_percentages()执行")
    return jsonify(percentages)


if __name__ == '__main__':
    batch_size = 16
    num_workers = 2  # numworkers 线程数量
    Epoch = 3
    display_frequency = 10  # 展示频率(每个epoch展示次数)

    mnist_train_loader = mnist_load(batch_size, num_workers)

    loss_func, optim, LeNet5 = lenet_initial()

    app.run(debug=True, port=5002)
