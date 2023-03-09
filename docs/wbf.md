# 安装

```python
pip install ensemble_boxes
```

# 推理

```python
python val.py --weights ./runs/train/yolov5s_auxota-head-TR/weights/best.pt --data data/data.yaml --save-txt --save-conf --task test --batch-size 1 --verbose --device 2 --name TR 

python val.py --weights ./runs/train/yolov5s_auxota-head-DCN/weights/best.pt --data data/data.yaml --save-txt --save-conf --task test --batch-size 1 --verbose --device 2 --name DCN 

or

...
```

# wbf

- [Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- [tph-yolov5/issues/3](https://github.com/cv516Buaa/tph-yolov5/issues/3)