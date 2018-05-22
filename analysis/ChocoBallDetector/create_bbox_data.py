
# coding: utf-8

# # 目的
# - pascal VOC形式のxmlファイルで作ったアノテーションデータをnumpy.arrayに変換する
# - データ元はjpeg画像とlabelImgで作ったアノテーションデータ(xmlファイル)
#     - https://github.com/tzutalin/labelImg
#
# ## required
# - xmltodict
# ```
# $ pip install xmltodict
# ```
#

import numpy as np

import glob
import os
from PIL import Image

import xmltodict

import argparse


def getBBoxData(anno_file, classes, data_dir):
    """
    Read and Parse annotation file(Pascal VOC like XML)
    Args:
        anno_file : file path of annotation file (Pascal VOC like XML)
        classes : list of classes ([class1(str), class2, ...])
        data_dir : data directory path
    Returns:
        dictionaly{img, img_arr, bboxs, obj_names, obj_ids}
            img : image data (PIL object)
            img_arr : image data (nimpy array, [channel, height, width])
    """
    # xmlファイルのパース
    with open(anno_file) as fd:
        pars = xmltodict.parse(fd.read())
    ann_data = pars['annotation']
    # read image
    img = Image.open(os.path.join(data_dir, ann_data['filename']))
    img_arr = np.asarray(img).transpose(2, 0, 1).astype(
        np.float32)  # データ型を指定
    # BoundingBoxとクラス名を読む
    bbox_list = list()
    obj_names = list()
    for obj in ann_data['object']:
        bbox_list.append([obj['bndbox']['ymin'], obj['bndbox']
                          ['xmin'], obj['bndbox']['ymax'], obj['bndbox']['xmax']])
        obj_names.append(obj['name'])
    bboxs = np.array(bbox_list, dtype=np.float32)  # データ型を指定
    obj_names = np.array(obj_names)
    obj_ids = np.array(
        list(map(lambda x: classes.index(x), obj_names)), dtype=np.int32)
    return {'img': img, 'img_arr': img_arr, 'bboxs': bboxs, 'obj_names': obj_names, 'obj_ids': obj_ids}


def getBBoxDataSet(data_dir, classes):
    """
    Args:
        classes : list of classes ([class1(str), class2, ...])
        data_dir : data directory path
    Returns:
        imgs : image data set ([N, channel, height, width])
        bboxs : list of bounding box [np.array([ymin, xmin, ymax, xmax],[], ...), (), ...]
        obj_ids : object id [np.array([obj_id, ...]), (), ...]
    """
    # get annotetion file list
    os.path.join(data_dir, '*.xml')
    anno_files = glob.glob(os.path.join(data_dir, '*.xml'))
    #
    img_list = list()
    bboxs = list()
    obj_ids = list()
    for ann_file in anno_files:
        ret = getBBoxData(anno_file=ann_file,
                          classes=classes, data_dir=data_dir)
        img_list.append(ret['img_arr'])
        bboxs.append(ret['bboxs'])
        obj_ids.append(ret['obj_ids'])
    imgs = np.array(img_list)
    return (imgs, bboxs, obj_ids)


def getClasses(classes_file):
    classes = list()
    with open(classes_file) as fd:
        for one_line in fd.readlines():
            cl = one_line.split('\n')[0]
            classes.append(cl)
    return classes


def main(args):
    classes_file = args.classes
    data_dir = args.data_dir

    # カテゴリファイル
    classes = getClasses(classes_file)
    print('classes_file : ', classes_file)
    print('classes : ', classes)
    print('data_dir : ', data_dir)
    # Parse data
    imgs, bboxs, obj_ids = getBBoxDataSet(data_dir=data_dir, classes=classes)
    # save
    np.save(os.path.join(data_dir, 'images.npy'), imgs)
    np.save(os.path.join(data_dir, 'bounding_box_data.npy'), bboxs)
    np.save(os.path.join(data_dir, 'object_ids.npy'), obj_ids)
    print('images_shape', imgs.shape)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparser')
    parser.add_argument('--classes', type=str,
                        default='data/classes.txt')
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()
    main(args)
