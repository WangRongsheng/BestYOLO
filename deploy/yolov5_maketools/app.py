import os
from colorama import init,Fore,Back,Style
import xml.etree.ElementTree as ET
import pickle
from os import listdir, getcwd
from os.path import join
import xml.dom.minidom as xmldom
import random
import sys
import shutil
from ruamel import yaml
import re
import zipfile

def exist_folder(img_path, ann_path):
    if not os.path.exists(img_path):
          os.makedirs(img_path)
          print(Fore.GREEN + "创建JPEGImages文件夹成功")
    if not os.path.exists(ann_path):
          os.makedirs(ann_path)
          print(Fore.GREEN + "创建Annotations文件夹成功")

def demo_logo():
    print("\n/*********************************/")
    print("/---------------------------------/\n")
    print("  欢迎使用: YOLOv5数据制作助手   ")
    print("    Copyright 2022 王荣胜   ")
    print("\n/---------------------------------/")
    print("/*********************************/\n")

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    if w>=1:
        w=0.99
    if h>=1:
        h=0.99
    return (x,y,w,h)

def get_labels():
    annotation_path=r'./Annotations/'
    annotation_names=[os.path.join(annotation_path,i) for i in os.listdir(annotation_path)]
    labels = list()
    for names in annotation_names:
        xmlfilepath = names
        domobj = xmldom.parse(xmlfilepath)
        elementobj = domobj.documentElement
        subElementObj = elementobj.getElementsByTagName("object")
        for s in subElementObj:
            label=s.getElementsByTagName("name")[0].firstChild.data
            if label not in labels:
                labels.append(label)
    return labels

def voc2yolo(classes, rootpath, xmlname):
    xmlpath = rootpath + '/Annotations'
    xmlfile = os.path.join(xmlpath,xmlname)
    with open(xmlfile, "r", encoding='UTF-8') as in_file:
        txtname = xmlname[:-4]+'.txt'
        txtpath = rootpath + '/worktxt'
        if not os.path.exists(txtpath):
            os.makedirs(txtpath)
        txtfile = os.path.join(txtpath,txtname)
        with open(txtfile, "w+" ,encoding='UTF-8') as out_file:
            tree=ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            out_file.truncate()
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult)==1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w,h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def split_dataset():
    root_path = '.'
    xmlfilepath = root_path + '/Annotations'
    txtsavepath = root_path + '/ImageSets/Main'
    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)
    train_test_percent = 0.9  # 修改
    train_valid_percent = 0.9  # 修改
    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list = range(num)
    tv = int(num * train_test_percent)
    ts = int(num-tv) 
    tr = int(tv * train_valid_percent)
    tz = int(tv-tr) 
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and valid size:", tv)
    print("train size:", tr)
    print("test size:", ts)
    print("valid size:", tz)

    ftest = open(txtsavepath + '/test.txt', 'w')
    ftrain = open(txtsavepath + '/train.txt', 'w')
    fvalid = open(txtsavepath + '/valid.txt', 'w')

    ftestimg = open(txtsavepath + '/img_test.txt', 'w')
    ftrainimg = open(txtsavepath + '/img_train.txt', 'w')
    fvalidimg = open(txtsavepath + '/img_valid.txt', 'w')

    for i in list:
        name = total_xml[i][:-4] + '.txt' + '\n'
        # 修改
        imgname = total_xml[i][:-4] + '.jpg' + '\n'
        if i in trainval:
            if i in train:
                ftrain.write(name)
                ftrainimg.write(imgname)
            else:
                fvalid.write(name)
                fvalidimg.write(imgname)
        else:
            ftest.write(name)
            ftestimg.write(imgname)
            
    ftrain.close()
    fvalid.close()
    ftest.close()

    ftrainimg.close()
    fvalidimg.close()
    ftestimg.close()

def remove_dataset():
    img_txt_cg_train = []
    img_txt_cg_test = []
    img_txt_cg_valid = []
    label_txt_cg_train = []
    label_txt_cg_test = []
    label_txt_cg_valid = []

    path = './ImageSets/Main/'

    for line in open(path+"img_train.txt"):
        line=line.strip('\n')
        img_txt_cg_train.append(line)
    for line1 in open(path+"img_test.txt"):
        line1=line1.strip('\n')
        img_txt_cg_test.append(line1)
    for line2 in open(path+"img_valid.txt"):
        line2=line2.strip('\n')
        img_txt_cg_valid.append(line2)

    for line3 in open(path+"train.txt"):
        line3=line3.strip('\n')
        label_txt_cg_train.append(line3)
    for line4 in open(path+"test.txt"):
        line4=line4.strip('\n')
        label_txt_cg_test.append(line4)
    for line5 in open(path+"valid.txt"):
        line5=line5.strip('\n')
        label_txt_cg_valid.append(line5)

    new_dataset_train = './dataset/train/images/'
    new_dataset_test = './dataset/test/images/'
    new_dataset_valid = './dataset/valid/images/'

    new_dataset_trainl = './dataset/train/labels/'
    new_dataset_testl = './dataset/test/labels/'
    new_dataset_validl = './dataset/valid/labels/'

    if not os.path.exists(new_dataset_train):
        os.makedirs(new_dataset_train)
    if not os.path.exists(new_dataset_test):
        os.makedirs(new_dataset_test)
    if not os.path.exists(new_dataset_valid):
        os.makedirs(new_dataset_valid)
    if not os.path.exists(new_dataset_trainl):
        os.makedirs(new_dataset_trainl)
    if not os.path.exists(new_dataset_testl):
        os.makedirs(new_dataset_testl)
    if not os.path.exists(new_dataset_validl):
        os.makedirs(new_dataset_validl)

    fimg = './JPEGImages/'
    flable = './worktxt/'

    # 小数据建议：copy 大数据建议：move
    for i in range(len(img_txt_cg_train)):
        shutil.copy(fimg+str(img_txt_cg_train[i]),new_dataset_train)
        shutil.copy(flable+str(label_txt_cg_train[i]),new_dataset_trainl)
    for j in range(len(img_txt_cg_test)):
        shutil.copy(fimg+str(img_txt_cg_test[j]),new_dataset_test)
        shutil.copy(flable+str(label_txt_cg_test[j]),new_dataset_testl)
    for q in range(len(img_txt_cg_valid)):
        shutil.copy(fimg+str(img_txt_cg_valid[q]),new_dataset_valid)
        shutil.copy(flable+str(label_txt_cg_valid[q]),new_dataset_validl)

if __name__=='__main__':
    img_path = './JPEGImages'
    ann_path = './Annotations'
    rootpath='.'
    xmlpath=rootpath+'/Annotations'
    init(autoreset=True)
    
    demo_logo()
    exist_folder(img_path, ann_path)
    print('请把需要训练的图片放入JPEGImages文件夹！')
    print('请保证放入JPEGImages文件夹中的图片后缀都为JPG格式，否则请使用源码修改！')
    put_img = input('是否已放入？（yes 或者 no）：')
    if put_img == 'yes' or put_img == 'Yes' or put_img == 'YES':
        print('请把对应的标注文件放入Annotations文件夹！')
        put_ann = input('是否已放入？（yes 或者 no）：')
        if put_ann == 'yes' or put_ann == 'Yes' or put_ann == 'YES':
            classes = get_labels()
            list=os.listdir(xmlpath)
            for i in range(0,len(list)) :
                path = os.path.join(xmlpath,list[i])
                if ('.xml' in path)or('.XML' in path):
                    voc2yolo(classes, rootpath, list[i])
                else:
                    print('not xml file',i)
            print(Fore.GREEN + "格式转化成功！")
            split_dataset()
            print(Fore.GREEN + "数据划分成功！")
            remove_dataset()
            print(Fore.GREEN + "数据划重构成功！")

            desired_caps = {
                'train': './dataset/train/images',
                'val': './dataset/valid/images', 
                'test': './dataset/test/images', 
                'nc': int(len(classes)), 
                'names': str(classes)
            }

            with open('./dataset/data.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(desired_caps, f, Dumper=yaml.RoundTripDumper)
            print(Fore.RED + "请删除【dataset/data.yaml】文件中【names】的引号！")
            chose_version = input('目前提供了YOLOv5的7.0、6.1、6.0、5.0版本的代码，请选择版本（7、6.1、6、5）：')
            if chose_version == '6.1':
                f = zipfile.ZipFile("./yolov5_version/yolov5-6.1.zip",'r')
                for file in f.namelist():
                    f.extract(file,"./") 
                f.close()
                shutil.copytree('./dataset','./yolov5-6.1/dataset')
            if chose_version == '6':
                f = zipfile.ZipFile("./yolov5_version/yolov5-6.0.zip",'r')
                for file in f.namelist():
                    f.extract(file,"./") 
                f.close()
                shutil.copytree('./dataset','./yolov5-6.0/dataset')
            if chose_version == '5':
                f = zipfile.ZipFile("./yolov5_version/yolov5-5.0.zip",'r')
                for file in f.namelist():
                    f.extract(file,"./") 
                f.close()
                shutil.copytree('./dataset','./yolov5-5.0/dataset')
            if chose_version == '7.0':
                f = zipfile.ZipFile("./yolov5_version/yolov5-7.0.zip",'r')
                for file in f.namelist():
                    f.extract(file,"./") 
                f.close()
                shutil.copytree('./dataset','./yolov5-5.0/dataset')
            print(Fore.GREEN + "YOLOv5代码解压与数据集构入完成！")
