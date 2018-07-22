# -*- coding: utf-8 -*-

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# command line arguments
#tf.app.flags.DEFINE_boolean("bool", True, "bool value")
#tf.app.flags.DEFINE_integer("int", 0, "int value")
tf.app.flags.DEFINE_string("img_path", "images", "image data path")


def main(argv):
    FLAGS = tf.app.flags.FLAGS
    print("image_data_path : {}".format(FLAGS.img_path))

    return 0


if __name__ == '__main__':
    tf.app.run()
