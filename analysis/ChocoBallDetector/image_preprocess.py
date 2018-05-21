
# -*- coding: utf-8 -*-

"""
目的
- アノテーション作業の前の一番最初の画像データの前処理
- 画像サイズを小さくする & 画像サイズを揃える
"""

import os
import glob
import numpy as np
from PIL import Image
import argparse


def main(args):
    img_files = glob.glob(os.path.join(args.img_dir, args.img_filter))
    print('image_dir : ', args.img_dir, ', filter : ', args.img_filter)
    print('image file number : ', len(img_files))

    """
    画像サイズが異なるものがあるが、縦横比は同じと仮定。
    高さを302に固定して、リサイズする。
    これで不具合出るようなら、強制的に(402, 302)でリサイズすれば良い。
    """
    height_size = 302
    for img_file in img_files:
        org_img = Image.open(img_file)
        img = org_img.copy()
        if img.height > img.width:  # 向きを一定にする
            img = img.rotate(90, expand=True)
        scale = float(height_size) / img.height
        res_img = img.resize((int(img.width*scale), height_size))
        res_img.save(os.path.join(args.out_dir, img_file.split('/')[-1]))
        print(img_file, np.array(org_img).shape, '->', np.array(res_img).shape)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparser')
    parser.add_argument('--img_dir', type=str, default='data/org_images')
    parser.add_argument('--out_dir', type=str, default='data/res_images')
    parser.add_argument('--img_filter', type=str, default='*.JPG')
    args = parser.parse_args()
    main(args)
