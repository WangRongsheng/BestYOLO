const div_video = document.getElementById("div-video") // 视频层
const video01 = document.getElementById("video-rt"); // 视频

var tf = tf // 声明tf

// ------------------YOLOv5配置------------------
const weights = './static/yolov5n_web_model/model.json'; // 权重文件

// 类别名称（中文版）
const cls_names = ['人', '自行车', '汽车', '摩托车', '飞机', '公交车', '火车', '卡车', '船', '红绿灯', '消防栓', '停止标志',
    '停车收费表', '长凳', '鸟', '猫', '狗', '马', '羊', '牛', '象', '熊', '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带',
    '手提箱', '飞盘', '滑雪板', '单板滑雪', '运动球', '风筝', '棒球棒', '棒球手套', '滑板', '冲浪板', '网球拍', '瓶子', '红酒杯',
    '杯子', '叉子', '刀', '勺', '碗', '香蕉', '苹果', '三明治', '橙子', '西兰花', '胡萝卜', '热狗', '比萨', '甜甜圈', '蛋糕',
    '椅子', '长椅', '盆栽', '床', '餐桌', '马桶', '电视', '笔记本电脑', '鼠标', '遥控器', '键盘', '手机', '微波炉', '烤箱',
    '烤面包机', '洗碗槽', '冰箱', '书', '时钟', '花瓶', '剪刀', '泰迪熊', '吹风机', '牙刷'
]

// 类别名称（英文版）
// const cls_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
//     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
//     'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
//     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
//     'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
//     'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
//     'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
//     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
// ]



// ------------------摄像头切换------------------
const videoAll = document.getElementById("video-all"); // 摄像头列表

//获取摄像头的名称和设备id
let VideoAllInfo = []; // 摄像头列表
navigator.mediaDevices.enumerateDevices().then((devices) => {
    if (devices) {
        devices.forEach((value) => {
            if (value.kind == "videoinput") {
                VideoAllInfo.push(value);
            }
        });
        if (VideoAllInfo.length > 0) {
            let videoAllItem = "";
            VideoAllInfo.forEach((value) => {
                let ItemDom = `<option value="${value.deviceId}">${value.label}</option>`
                videoAllItem += ItemDom;
            });
            videoAll.innerHTML = videoAllItem;
        }
    }
});


// 颜色切换
function colorsSwitch() {
    let colors = [];

    // 深色系
    colors = ["#800000", "#808000", "#09713d", "#007480", "#000080", "#4B0080",
        "#FF7F50", "#CC5500", "#B87333", "#CC7722", "#704214", "#50C878", "#DE3163", "#003153"
    ];

    return colors;
}

// 选择预测框绘制颜色
function selectColor(index) {
    let colors = colorsSwitch(); // 切换颜色
    let i = index % 15; // 颜色循环
    let color = colors[i]; // 设置颜色
    return color;
}

//hex -> rgba
function hexToRgba(hex, opacity) {
    return 'rgba(' + parseInt('0x' + hex.slice(1, 3)) + ',' + parseInt('0x' + hex.slice(3, 5)) + ',' +
        parseInt('0x' + hex.slice(5, 7)) + ',' + opacity + ')';
}


// 绘制边界框和标签
async function renderPredictions(res) {
    $("div").remove(".div-bbox"); // 清除边界框
    $("span").remove(".span-label"); // 清除标签

    let [boxes, scores, classes, valid_detections] = res; // 获取检测信息

    let totalClasses = new Array(); // 类别数组

    let cls_index = 0; // 类别序号，用于分配颜色
    let num_box = 0; // 边界框ID

    for (let i = 0; i < valid_detections.dataSync()[0]; ++i) {
        // 坐标点
        let [x0, y0, x1, y1] = boxes.dataSync().slice(i * 4, (i + 1) * 4);

        // ------------------修复tf.js检测结果超出正常范围的bug------------------
        x0 = x0 < 0 || x0 > 1 ? parseInt(x0) : x0;
        x1 = x1 < 0 || x1 > 1 ? parseInt(x1) : x1;
        y0 = y0 < 0 || y0 > 1 ? parseInt(y0) : y0;
        y1 = y1 < 0 || y1 > 1 ? parseInt(y1) : y1;

        x0 = parseInt(Math.abs(x0) * video01.offsetWidth)
        x1 = parseInt(Math.abs(x1) * video01.offsetWidth)
        y0 = parseInt(Math.abs(y0) * video01.offsetHeight)
        y1 = parseInt(Math.abs(y1) * video01.offsetHeight)

        let cls = cls_names[classes.dataSync()[i]]; // 类别
        let score = scores.dataSync()[i].toFixed(2); // 置信度

        // 加入类别列表
        if (totalClasses.includes(cls) == false) {
            totalClasses[cls_index] = cls;
            cls_index += 1;
        }

        let color_index = totalClasses.indexOf(cls) // 类别索引值

        // ------------------检测框------------------
        const div_bbox = document.createElement("div");
        div_bbox.id = `${"div-bbox" + num_box}`;
        div_bbox.className = "div-bbox";
        div_bbox.style.position = "absolute";
        div_video.appendChild(div_bbox);

        div_bbox.style.width = (x1 - x0) + "px";
        div_bbox.style.height = (y1 - y0) + "px";
        div_bbox.style.border = "2px solid " + selectColor(color_index);

        // ------------------边界框位置------------------
        div_bbox.style.marginLeft = x0 + "px"; // x0
        div_bbox.style.marginTop = y0 + "px"; // y0


        // ------------------标签------------------
        let content = cls + " " + parseFloat(score).toFixed(2);
        const span_label = document.createElement('span');
        span_label.id = `${"span-label" + num_box}`;
        span_label.className = "span-label";
        span_label.style.backgroundColor = selectColor(color_index);

        // ------------------标签透明度设置------------------
        let rgba_label = hexToRgba(selectColor(i), parseFloat(0.5)); // 标签透明度0.5
        span_label.style.backgroundColor = rgba_label; // 设置标签背景颜色
        span_label.innerHTML = content; // 标签内容
        span_label.style.color = "#ffffff"; // 颜色
        span_label.style.position = "absolute";
        span_label.style.marginLeft = x0 + "px";
        span_label.style.marginTop = y0 + "px";
        span_label.style.display = "block";
        div_video.appendChild(span_label);

        num_box = num_box + 1;
    }

    // ------------------清除检测结果tensor------------------
    boxes.dispose();
    scores.dispose();
    classes.dispose();
    valid_detections.dispose();
}


// 检测帧
async function detectFrame(video, model) {
    // 模型输入尺寸
    let [modelWeight, modelHeight] = model.inputs[0].shape.slice(1, 3);

    // 输入, tf.tidy()防止内存溢出
    let input = tf.tidy(() => tf.image.resizeBilinear(tf.browser.fromPixels(video), [modelWeight, modelHeight]).div(255.0).expandDims(0));

    // 执行异步函数
    await model.executeAsync(input).then(res => {

        renderPredictions(res); // 画框

        requestAnimationFrame(() => {
            tf.dispose(res); // 清除预测tensor
            input.dispose(); // 清除输入tensor

            // 读取下一帧
            detectFrame(video, model); // 递归
        });
    });
}


// 启动函数
async function openCam(videoID) {
    // 摄像头切换
    let videoObj;
    if (videoID == "") {
        //设置默认获取的摄像头
        videoObj = {
            "video": true,
            "audio": false
        }
    } else {
        //切换摄像头
        videoObj = {
            "video": { deviceId: videoID },
            "audio": false
        };
    }

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // 启动webcam
        const webCamPromise = navigator.mediaDevices
            .getUserMedia(videoObj).then(stream => {
                window.stream = stream;
                video01.srcObject = stream; // 视频流
                return new Promise((resolve) => {
                    // 视频流异步加载
                    video01.onloadedmetadata = () => {
                        resolve();
                    };
                });
            });

        // 加载权重文件
        const modelPromise = tf.loadGraphModel(weights);
        Promise.all([modelPromise, webCamPromise])
            .then(values => {
                // 执行检测函数
                detectFrame(video01, values[0]);
            });
    }
}


// 切换摄像头
document.getElementById("video-all").onchange = () => {
    if (document.getElementById("video-all").children.length > 0) {
        let selIndex = document.getElementById("video-all").selectedIndex; // 获取当前设备的index值
        let selectedValue = document.getElementById("video-all").options[selIndex].value; // 获取选中的设备值
        // 切换摄像头
        openCam(selectedValue);
    }
}

// 摄像头开启
const webcam_open = document.getElementById("webcam-open");
webcam_open.onclick = () => {
    openCam(); // 启动设备
}
