# -*- coding: utf-8 -*-

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# command line arguments
#tf.app.flags.DEFINE_boolean("bool", True, "bool value")
#tf.app.flags.DEFINE_integer("int", 0, "int value")
tf.app.flags.DEFINE_string("str", "str", "string value")


def main(argv):
    FLAGS = tf.app.flags.FLAGS
    print(FLAGS.str)  # test 後で消す

    return 0


if __name__ == '__main__':
    tf.app.run()
