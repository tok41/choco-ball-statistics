# -*- coding: utf-8 -*-

# Imports
import tensorflow as tf
import make_image_def as mid

tf.logging.set_verbosity(tf.logging.INFO)

# command line arguments
tf.app.flags.DEFINE_string(
    "train_list", "image_list_train.csv", "image data list(csv)")
tf.app.flags.DEFINE_string(
    "test_list", "image_list_test.csv", "image data list(csv)")
tf.app.flags.DEFINE_integer("n_batch", 10, "mini batch size")
tf.app.flags.DEFINE_integer("n_epoch", 1, "number of epoch")

tf.app.flags.DEFINE_string("img_path", "images", "image data path")
tf.app.flags.DEFINE_string(
    "db_path", "../../data/choco-ball.db", "DB file path")


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
    return tf.image.decode_image(contents), label


# ###########################
# ### Create DataSet
def input_fn(file_path, n_batch):
    """
    Estimatorに流すためのデータ入力関数
    Estimatorには、lambda式を使って引数を与えて入力する
    参考
    https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
    """
    def generator():
        return read_csv(file_path)  # generator

    dataset = tf.data.Dataset.from_generator(
        generator, (tf.string, tf.int32), (tf.TensorShape([]), tf.TensorShape([])))\
        .map(read_image)
    dataset = dataset.shuffle(n_batch)
    dataset = dataset.batch(n_batch)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels


# ###########################
# ### Main
def main(argv):
    FLAGS = tf.app.flags.FLAGS

    # 画像リストファイルの作成
    df_img = mid.makeImageDefinition(FLAGS.img_path, FLAGS.db_path)
    print('image_num : {}'.format(df_img.shape[0]))
    df_train, df_test = mid.splitTrainTest(
        df_img_def=df_img, rate=0.8, upsampling=True)
    df_train.to_csv(FLAGS.train_list, index=False, header=False)
    print("saved image_list_file : {}".format(FLAGS.train_list))
    df_test.to_csv(FLAGS.test_list, index=False, header=False)
    print("saved image_list_file : {}".format(FLAGS.test_list))

    # ここからinput_fnに移動予定
    # ** train dataset
    def generator(): return read_csv(FLAGS.train_list)  # generator

    dataset = tf.data.Dataset.from_generator(
        generator, (tf.string, tf.int32), (tf.TensorShape([]), tf.TensorShape([])))\
        .map(read_image)
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(FLAGS.n_epoch)
    dataset = dataset.batch(FLAGS.n_batch)
    iterator = dataset.make_one_shot_iterator()
    value = iterator.get_next()
    # ここまでinput_fnに移動予定

    # ここからinput_fnに移動予定
    # ** test dataset
    def generator_test(): return read_csv(FLAGS.test_list)  # generator

    dataset_test = tf.data.Dataset.from_generator(
        generator_test, (tf.string, tf.int32), (tf.TensorShape([]), tf.TensorShape([])))\
        .map(read_image)
    dataset_test = dataset_test.shuffle(10000)
    dataset_test = dataset_test.repeat(FLAGS.n_epoch)
    dataset_test = dataset_test.batch(FLAGS.n_batch)
    iterator_test = dataset_test.make_one_shot_iterator()
    value_test = iterator_test.get_next()
    # ここまでinput_fnに移動予定

    # Run-Graph
    sess = tf.Session()
    try:
        while True:
            img, label = sess.run(value)
            print("{}, {}".format(img.shape, label))
    except tf.errors.OutOfRangeError:
        pass

    try:
        while True:
            img, label = sess.run(value_test)
            print("{}, {}".format(img.shape, label))
    except tf.errors.OutOfRangeError:
        pass

    return 0


if __name__ == '__main__':
    tf.app.run()
