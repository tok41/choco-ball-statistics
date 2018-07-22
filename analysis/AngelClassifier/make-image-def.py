# -*- coding: utf-8 -*-

"""
ラベル付きの画像定義ファイルを作成する。
"""

# Imports
import os
import sqlite3
import pandas as pd
import glob
import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_string("img_path", "images", "image data path")
tf.app.flags.DEFINE_string(
    "db_path", "../../data/choco-ball.db", "DB file path")


def main(argv):
    FLAGS = tf.app.flags.FLAGS
    print("image_data_path : {}".format(FLAGS.img_path))

    return 0


if __name__ == '__main__':
    main()
