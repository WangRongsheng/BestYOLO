"""
fogging train and test datasets using synthetic fog algorithm
"""

import os, sys
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
import random
from copy import deepcopy

from synthetic_fog import SyntheticFog


class AugmentCrosswalkDataset(object):
    def __init__(self, source_path):
        self.sp = source_path  # source path
        p = Path(self.sp)
        self.tp = f'{p.parent}/fogged_{p.stem}'  # target path

        self.sf = SyntheticFog()  # synthetic fog object

    def augment(self, show=False):
        """augment train and test set in YOLOv5 format"""
        # 逐张进行增强
        sp = self.sp
        tp = self.tp
        print(f'fogged data will be saved to: {tp}')
        if os.path.exists(self.tp):

            shutil.rmtree(self.tp)
        os.makedirs(f'{self.tp}/train/images')
        os.makedirs(f'{self.tp}/test/images')
        os.makedirs(f'{self.tp}/train/labels')
        os.makedirs(f'{self.tp}/test/labels')

        for trte in ['train', 'test']:
            pi = f'{sp}/{trte}/images'  # path of images
            pl = f'{sp}/{trte}/labels'
            ti = f'{tp}/{trte}/images'
            tl = f'{tp}/{trte}/labels'

            imgs = [f'{x}' for x in os.listdir(pi) if x.endswith('.jpg')]
            #print(f'transform {trte} images, total: {len(imgs)}, transformed total: {2*len(img)}.')
            bar = tqdm(imgs)
            for i, img_name in enumerate(bar):
                img_path = f'{pi}/{img_name}'
                stem = Path(img_path).stem
                assert os.path.exists(img_path), f'img does not exists {img_path}'

                # 先拷贝原始图像和标注
                shutil.copy(img_path, f'{ti}/{img_name}')
                shutil.copy(f'{pl}/{stem}.txt', f'{tl}/{stem}.txt')

                # fogging
                img = cv2.imread(img_path)
                h, w, c = img.shape
                # random brightness and thickness
                br = np.clip(0.2 * np.random.randn() + 0.5, 0.1, 0.9)  # 0.1~0.9
                th = np.clip(0.01 * np.random.randn() + 0.05, 0.01, 0.09)
                normed_img = img.copy()/255.0
                fogged_img = self.sf.fogging_img(
                    normed_img, brightness=br, thickness=th, high_efficiency=True)
                fogged_img = np.array(fogged_img*255, dtype=np.uint8)

                # save fogged images and labels
                cv2.imwrite(f'{ti}/fogged_{img_name}', fogged_img)
                shutil.copy(f'{pl}/{stem}.txt', f'{tl}/fogged_{stem}.txt')

                if show:
                    print(f'img_name: {Path(img_path).name} img: {img.shape} br: {br} th: {th} max: {np.max(fogged_img)}')
                    self.show(img, name='src_img', wait=False)
                    self.show(fogged_img, name='fogged_img', wait=False)
                    if cv2.waitKey(0) == ord('q'):
                        break

                bar.set_description(f'Img and fogged img saved, {stem}.')

    def show(self, img, name='xx', wait=True):
        h, w, c = img.shape
        scale = 0.5
        show_img = cv2.resize(img, (int(w*scale), int(h*scale)))
        cv2.imshow(name, show_img)
        if wait:
            cv2.waitKey(0)

    def augment_testset(self, dir):
        """augment only test set"""
        self.sf.fogging_dir(sp=dir, tp=None, random_params=True, save_src_img=True)


if __name__ == '__main__':
    source_path = './data'
    acd = AugmentCrosswalkDataset(source_path)
    acd.augment(show=False)
    # test_imgs_path = '/home/zzd/datasets/crosswalk/testsets_1770/Images'
    # acd.augment_testset(test_imgs_path)











