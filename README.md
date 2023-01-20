

# 🌟改进

- [Backbone-ResNet18](https://github.com/WangRongsheng/BestYOLO/blob/main/models/backbone/resnet18.yaml) 对齐 [resnet18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18)
- [Backbone-RegNet_y_400mf](https://github.com/WangRongsheng/BestYOLO/blob/main/models/backbone/RegNety400.yaml) 对齐 [regnet_y_400mf](https://pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html#torchvision.models.regnet_y_400mf)
- [Backbone-MobileNetV3 small](https://github.com/WangRongsheng/BestYOLO/blob/main/models/backbone/MobileNetV3s.yaml) 对齐 [mobilenet_v3_small](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small)
- [Backbone-EfficientNet_B0](https://github.com/WangRongsheng/BestYOLO/blob/main/models/backbone/efficientnet_b0.yaml) 对齐 [efficientnet_b0](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0)
- [Backbone-ResNet34](https://github.com/WangRongsheng/BestYOLO/blob/main/models/backbone/resnet34.yaml) 对齐 [resnet34](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html#torchvision.models.resnet34)
- [Backbone-ResNet50](https://github.com/WangRongsheng/BestYOLO/blob/main/models/backbone/resnet50.yaml) 对齐 [resnet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50)
- [Backbone-EfficientNetV2_s](https://github.com/WangRongsheng/BestYOLO/blob/main/models/backbone/efficientnet_v2_s.yaml) 对齐 [efficientnet_v2_s](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s)

> 所有Backbone都支持开启预训练权重，只需添加`pretrained=True`到每个[common.py](https://github.com/WangRongsheng/BestYOLO/blob/main/models/common.py#L870) 的模型中！

|models|layers|parameters|model size|
|:-|:-|:-|:-|
|yolov5n|214|1766623|3.9|
|MobileNetV3s|313|2137311|4.7|
|efficientnet_b0|443|6241531|13.0|
|RegNety400|450|5000191|10.5|
|ResNet18|177|12352447|25.1|
|ResNet34|223|22460607|45.3|
|ResNet50|258|27560895|55.7|
|EfficientNetV2_s|820|22419151|45.8|

# 💻应用

- [TFjs部署使用](https://github.com/WangRongsheng/BestYOLO/tree/main/deploy/yolov5_tfjs_flask)

<div align="center"><img src="./images/tfjs1.png" /></div>

- [Pyqt GUI使用](https://github.com/WangRongsheng/BestYOLO/tree/main/deploy/gui)

<div align="center"><img src="./images/gui.png" /></div>

# 📋参考

- [https://github.com/ultralytics/yolov5/tree/v7.0](https://github.com/ultralytics/yolov5/tree/v7.0)
- [https://github.com/ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)
- [https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)
- [https://github.com/Gumpest/YOLOv5-Multibackbone-Compression](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression)
- [https://github.com/jizhishutong/YOLOU](https://github.com/jizhishutong/YOLOU)
- [https://github.com/Bobo-y/flexible-yolov5](https://github.com/Bobo-y/flexible-yolov5)
- [https://github.com/iscyy/yoloair](https://github.com/iscyy/yoloair)
- [https://github.com/WangQvQ/Yolov5_Magic](https://github.com/WangQvQ/Yolov5_Magic)
- [https://github.com/Hongyu-Yue/yoloV5_modify_smalltarget](https://github.com/Hongyu-Yue/yoloV5_modify_smalltarget)
- [https://github.com/wuzhihao7788/yolodet-pytorch](https://github.com/wuzhihao7788/yolodet-pytorch)
- [https://github.com/iscyy/yoloair2](https://github.com/iscyy/yoloair2)
- [https://github.com/positive666/yolo_research](https://github.com/positive666/yolo_research)
- [https://github.com/Javacr/PyQt5-YOLOv5](https://github.com/Javacr/PyQt5-YOLOv5)
- [https://github.com/yang-0201/YOLOv6_pro](https://github.com/yang-0201/YOLOv6_pro)
