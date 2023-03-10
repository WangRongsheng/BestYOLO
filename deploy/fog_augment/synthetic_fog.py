"""
直接运行程序可以测试合成雾气效果
Produced by: zhangzhengde@sjtu.edu.cn
"""
import os
import math
import cv2
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil


class SyntheticFog(object):
    def __init__(self):
        pass

    def __call__(self, show=False):
        img_path = '../example/fog_image/raw.jpg'
        # img_path = '../sources/IMG_6685.JPG'
        assert os.path.exists(img_path), f'error: img does not exists, {img_path}'
        img = cv2.imread(img_path)
        print(img.shape)
        img = img/255.0
        print(f'fogging...')
        t0 = time.time()
        br = 0.7
        th = 0.05
        fogged_img = self.fogging_img(
            img, brightness=br, thickness=th,
            high_efficiency=True)
        print(f'fogging time: {(time.time()-t0)*1000:.4f}ms')
        rf = 1  # resize factor
        img = cv2.resize(img, (int(img.shape[1]*rf), int(img.shape[0]*rf)))
        fogged_img = cv2.resize(fogged_img, ((int(fogged_img.shape[1]*rf)), (int(fogged_img.shape[0]*rf))))
        fogged_img = np.array(fogged_img*255, dtype=np.uint8)
        if show:
            cv2.imshow('src', img)
            cv2.imshow('fogged', fogged_img)
            cv2.waitKey(0)
        cv2.imwrite(f'../example/fog_image/fogged_br{br}_th{th}.jpg', fogged_img)

    def fogging_dir(self, sp, tp=None, random_params=True, brightness=None, thickness=None, save_src_img=False):
        """
        fogging images in a directory
        :param sp: str, source dir path
        :param tp: str, target dir path, tp is fogged_{sp} by default
        :param random_params: bool, use random brightness and fog thickness params if True
        :param brightness: float, 0.1 to 0.9, gray of synthetic fog, pure white fog if 1, dark fog if 0.
        :param thickness: float, 0.01 to 0.09, thickness of synthetic fog, the larger the value, the thicker the fog.
        :param save_src_img: save source image at the same time i.e. copy source imgs to tp
        :return: None, all fogged images will be saved to target dir path.
        """
        tp = tp if tp is not None else f'{Path(sp).parent}/fogged_{Path(sp).name}'
        if os.path.exists(tp):
            ipt = input(f'Target dir: {tp} exists, do you want to remove it and continue. [Yes]/No: ')
            if ipt in ['', 'Yes', 'Y', 'yes']:
                shutil.rmtree(tp)
            else:
                print('do nothing')
                exit()
        os.makedirs(f'{tp}')

        imgs = [x for x in os.listdir(sp) if str(Path(x).suffix).lower() in ['.jpg', '.bmp']]
        print(f'Fogging {len(imgs)} images in dir {sp}, \nfogged images will be save to {tp}.')
        bar = tqdm(imgs)
        for i, img_name in enumerate(bar):
            img_path = f'{sp}/{img_name}'
            # stem = Path(img_path).stem
            # suffix = Path(img_path).suffix

            if save_src_img:  # save source img
                shutil.copy(f'{sp}/{img_name}', f'{tp}/{img_name}')

            img = cv2.imread(img_path)
            h, w, c = img.shape
            normed_img = img.copy()/255.0

            if random_params:
                br = np.clip(0.2 * np.random.randn() + 0.5, 0.1, 0.9)  # 0.1~0.9
                th = np.clip(0.01 * np.random.randn() + 0.05, 0.01, 0.09)
            else:
                br = brightness
                th = thickness
                assert br is not None
                assert th is not None
            fogged_img = self.fogging_img(normed_img, br, th, high_efficiency=True)
            fogged_img = np.array(fogged_img * 255, dtype=np.uint8)
            cv2.imwrite(f'{tp}/fogged_{img_name}', fogged_img)

            bar.set_description(f'Fogged image saved, fogged_{img_name}')

    def fogging_img(self, img, brightness=0.7, thickness=0.05, high_efficiency=True):
        """
        fogging single image
        :param img: src img
        :param brightness: brightness
        :param thickness: fog thickness, without fog when 0, max 0.1,
        :param high_efficiency: use matrix to improve fogging speed when high_efficiency is True, else use loops
                low efficiency: about 4000ms, high efficiency: about 80ms, tested in (864, 1152, 3) img
        :return: fogged image
        """
        assert 0 <= brightness <= 1
        assert 0 <= thickness <= 0.1
        fogged_img = img.copy()
        h, w, c = fogged_img.shape
        if not high_efficiency:  # use default loop to fogging, low efficiency
            size = np.sqrt(np.max(fogged_img.shape[:2]))  # 雾化尺寸
            center = (h // 2, w // 2)  # 雾化中心
            # print(f'shape: {img.shape} center: {center} size: {size}')  # 33
            # d_list = []
            for j in range(h):
                for l in range(w):
                    d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                    # print(f'd {d}')
                    td = math.exp(-thickness * d)
                    # d_list.append(td)
                    fogged_img[j][l][:] = fogged_img[j][l][:] * td + brightness * (1 - td)
                # x = np.arange(len(d_list))
                # plt.plot(x, d_list, 'o')
                # if j == 5:
                #     break
        else:  # use matrix  # TODO: 直接使用像素坐标，距离参数不适用于大分辨率图像，会变成鱼眼镜头的样子. done.
            use_pixel = True
            size = np.sqrt(np.max(fogged_img.shape[:2])) if use_pixel else 1  # 雾化尺寸
            h, w, c = fogged_img.shape
            hc, wc = h // 2, w // 2
            mask = self.get_mask(h=h, w=w, hc=hc, wc=wc, pixel=use_pixel)  # (h, w, 2)
            d = -0.04 * np.linalg.norm(mask, axis=2) + size

            td = np.exp(-thickness * d)

            for cc in range(c):
                fogged_img[..., cc] = fogged_img[..., cc] * td + brightness*(1-td)

            # a = np.linalg.norm(mask, axis=2)
            # print(f'size: {fogged_img.shape} a: {a} max: {np.max(fogged_img)} {np.min(fogged_img)}')

            fogged_img = np.clip(fogged_img, 0, 1)  # 解决黑白噪点的问题
            # print(f'mask: {mask[:, :, 1]} {mask.shape}')
            # print(f'd: {d} {d.shape}')

        return fogged_img

    def get_mask(self, h, w, hc, wc, pixel=True):
        mask = np.zeros((h, w, 2), dtype=np.float32)
        if pixel:
            mask[:, :, 0] = np.repeat(np.arange(h).reshape((h, 1)), w, axis=1) - hc
            mask[:, :, 1] = np.repeat(np.arange(w).reshape((1, w)), h, axis=0) - wc
        else:
            mask[:, :, 0] = np.repeat(np.linspace(0, 1, h).reshape(h, 1), w, axis=1) - 0.5
            mask[:, :, 1] = np.repeat(np.linspace(0, 1, w).reshape((1, w)), h, axis=0) - 0.5
        return mask


if __name__ == '__main__':
    synf = SyntheticFog()
    synf(show=True)



