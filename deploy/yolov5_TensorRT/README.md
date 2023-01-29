# 1.export onnx

下载yolov5源码(https://github.com/ultralytics/yolov5)
完成环境搭建，建议使用链接提供的docker(docker pull ultralytics/yolov5)
```C++
python export.py --weights yolov5s.pt --include onnx --simplify
```

# 2.onnx转TensorRT
```C++
 ./trtexec --onnx=/path_to/yolov5s.onnx --saveEngine=/path_to/yolov5s.engine
```

# 3.运行
```C++
mkdir build && cd build
cmake ..
make
./demo
```
