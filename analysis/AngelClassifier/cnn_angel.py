# -*- coding: utf-8 -*-

# Imports
import os
import tensorflow as tf
import make_image_def as mid

tf.logging.set_verbosity(tf.logging.INFO)

# command line arguments
tf.app.flags.DEFINE_string(
    "train_list", "image_list_train.csv", "image data list(csv)")
tf.app.flags.DEFINE_string(
    "test_list", "image_list_test.csv", "image data list(csv)")
tf.app.flags.DEFINE_integer("n_batch", 5, "mini batch size")
tf.app.flags.DEFINE_integer("n_epoch", 10, "number of epoch")

tf.app.flags.DEFINE_string("img_path", "images", "image data path")
tf.app.flags.DEFINE_string(
    "db_path", "../../data/choco-ball.db", "DB file path")


# ###########################
# ### Create Image List
def createImageListFile(img_path, db_path, train_file, test_file, out_dir='.', train_rate=0.8):
    """
    訓練、テストの画像定義ファイルを作成する
    Args:
        img_path:画像ディレクトリのパス, string
        db_path:DBパス, string
        train_file, test_file:画像リストのファイル名, string
        out_dir:画像リストファイルの出力ディレクトリ, string, default='.'
        train_rate:学習データに使うデータの割合, float, default=0.8
    """
    df_img = mid.makeImageDefinition(img_path, db_path)
    print('image_num : {}'.format(df_img.shape[0]))
    df_train, df_test = mid.splitTrainTest(
        df_img_def=df_img, rate=train_rate, upsampling=True)
    train_list_path = os.path.join(out_dir, train_file)
    test_list_path = os.path.join(out_dir, test_file)
    df_train.to_csv(train_list_path, index=False, header=False)
    print("saved image_list_file : {}".format(train_list_path))
    df_test.to_csv(test_list_path, index=False, header=False)
    print("saved image_list_file : {}".format(test_list_path))


# ###########################
# ### Read Image Files
def read_csv(filename):
    """
    データファイルの定義ファイルを読み込むgeneratorを作る
    CSV形式(,区切り)で、一行に[filename, label]の想定
    Args:
        filename:Dataset Definition File(full path)
    """
    with open(filename, 'r') as f:
        for line in f.readlines():
            record = line.rstrip().split(',')
            image_file = record[0]
            label = int(record[1])
            yield image_file, label


def read_image(image_file, label):
    """
    Args:
        image_file:path of image file [string]
        label:label [int]
    Returns:
        image_tensor:Tensor with type uint8 with shape [height, width, num_channels]
        label:label [int]
    """
    contents = tf.read_file(image_file)
    image = tf.image.decode_image(contents)  # 画像データを[0,1)の範囲に変換
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # return tf.image.decode_image(contents), label
    return image, label


# ###########################
# ### Input data function
def input_fn(file_path, n_batch, n_epoch):
    """
    データ入力関数
    lambda式を使って引数を与えて入力する
    参考
    https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
    """
    def generator():
        return read_csv(file_path)  # generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.string, tf.int32),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([])))\
        .map(read_image)
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(n_epoch)
    dataset = dataset.batch(n_batch)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels


# ###########################
# ### Define Model
def cnn_classifier(images, reuse):
    with tf.variable_scope('cnn', reuse=reuse):
        x = tf.reshape(images, [-1, 1100, 1480, 3])
        conv1 = tf.layers.conv2d(
            x, filters=32, kernel_size=(10, 10),
            padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            conv1, pool_size=(10, 10), strides=(10, 10))
        conv2 = tf.layers.conv2d(
            pool1, filters=64, kernel_size=(5, 5),
            padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            conv2, pool_size=(5, 5), strides=(5, 5))
        pool2_flat = tf.contrib.layers.flatten(pool2)
        fc3 = tf.layers.dense(pool2_flat, 1024)
        fc4 = tf.layers.dense(fc3, 2)
    return fc4


# ###########################
# ### Main
def main(argv):
    FLAGS = tf.app.flags.FLAGS

    # 画像リストファイルの作成
    createImageListFile(FLAGS.img_path, FLAGS.db_path,
                        FLAGS.train_list, FLAGS.test_list)

    # Inference
    images, labels = input_fn(
        FLAGS.train_list, n_batch=FLAGS.n_batch, n_epoch=FLAGS.n_epoch)
    y_pred_op = cnn_classifier(images, reuse=False)

    # loss
    loss_op = tf.losses.softmax_cross_entropy(
        tf.one_hot(labels, depth=2),
        y_pred_op)

    # training
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(loss_op)

    # valiable initialization
    init = tf.global_variables_initializer()

    # Run-Graph
    sess = tf.Session()
    # initialize
    sess.run(init)
    try:
        while True:
            _, loss, imgs = sess.run((train_op, loss_op, images))
            print("{}, {}".format(loss, imgs.shape))
    except tf.errors.OutOfRangeError:
        pass

    return 0


if __name__ == '__main__':
    tf.app.run()
