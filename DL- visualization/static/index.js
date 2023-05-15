// 按钮点击开始训练，执行函数
async function startTraining() {
    const response = await fetch('/train', { method: 'POST' });
    const data = await response.json();
    if (data.status === 'Training started') {
        console.log("updateImages开始!!!!!!!!!!!!!");
        // model_train();
        sendCanvas()
        updateImages();
        updatePercentages();
    }
}

async function endTraining() {
    const response = await fetch('/endtrain', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: 'shutdown' })
    })
}


// 在startTraining 中执行的函数，模型开始训练
function model_train() {
    fetch('/model_train', { method: 'GET' });
}

// 在startTraining 中执行的函数，对图片url进行更新
async function updateImages() {
    const response = await fetch('/get_image_list');
    const data = await response.json();
    console.log(data);

    // 获取所有的square的div，往里填入图片
    const squares = document.getElementsByClassName('square');
    console.log(squares);

    for (var i = 0; i < squares.length; i++) {
        squares[i].innerHTML = "";
        console.log("HTML删除");
    }

    for (var i = 0; i < squares.length; i++) {
        // var wlz = data.images[i]    ----->pool2/pool2_10.png
        const [folder, imageName] = data.images[i].split('/');
        const imageId = imageName.split('.')[0];
        const img = document.createElement('img');
        var timestamp = new Date().getTime(); // 获取当前时间戳
        img.src = `/get_image/${data.images[i]}?` + timestamp;
        img.alt = '';
        squares[i].appendChild(img)
        console.log("添加图片成功");
    }

    setTimeout(updateImages, 4000);  // 更新间隔，单位：毫秒
}

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// 将整个 Canvas 填充为黑色
ctx.fillStyle = '#000000';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// 将画笔设置为白色
ctx.strokeStyle = '#000000';
ctx.fillStyle = '#ffffff'
ctx.lineWidth = 2;

canvas.addEventListener("mousedown", function (e) {
    isDrawing = true;
    let rect = canvas.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    draw(x, y);
});

canvas.addEventListener("mousemove", function (e) {
    if (isDrawing) {
        let rect = canvas.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;
        draw(x, y);
    }
});

canvas.addEventListener("mouseup", function (e) {
    isDrawing = false;
});

function draw(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
}

// 传输canvas图片
function sendCanvas() {
    // 将 Canvas 转换为 Base64 编码的 URL
    let dataURL = canvas.toDataURL('image/png');

    // 将 Base64 编码的数据发送到后端
    fetch("/upload", {
        method: "POST",
        body: JSON.stringify({ image: dataURL }),
        headers: {
            "Content-Type": "application/json"
        }
    });
}

// 清空画布
let reSetCanvas = document.getElementById("clear");
reSetCanvas.onclick = function () {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // // 将整个 Canvas 填充为黑色
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // 将画笔设置为白色
    ctx.strokeStyle = '#000000';
    ctx.fillStyle = '#ffffff'
    ctx.lineWidth = 2;
};


// 请求获取数据，由后端接收请求响应，返回数据，最后展示在页面上
async function updatePercentages() {
    // 请求路由
    const response = await fetch('/get-percentages');
    // 获得返回的文件
    const percentages = await response.json();
    console.log(percentages);
    const perrect = document.querySelector('.perrect g');

    for (let i = 0; i < percentages.length; i++) {
        const childrenElement = perrect.children[i * 2 + 1];
        childrenElement.querySelector('tspan').innerHTML = `${percentages[i]}%`;
    }
}

updatePercentages();