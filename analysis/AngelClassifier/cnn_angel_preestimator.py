# -*- coding: utf-8 -*-

# Imports
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# command line arguments
tf.app.flags.DEFINE_string(
    "train_list", "image_list_train.csv", "image data list(csv)")
tf.app.flags.DEFINE_string(
    "test_list", "image_list_test.csv", "image data list(csv)")
tf.app.flags.DEFINE_integer("n_batch", 4, "mini batch size")
tf.app.flags.DEFINE_integer("n_epoch", 1, "number of epoch")


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
    # return tf.image.decode_image(contents), label
    image = tf.image.decode_image(contents)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label
    # return image, tf.one_hot(label, depth=2)


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

    # Use DNNClassifier
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],  # Two layers, each with 10 neurons
        n_classes=3,
        model_dir=PATH)
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./output")

    # Set up logging for predictions
    # cnn_model_fnで定義した予測値の確率表現の名称
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=4)

    # ***** Train
    mnist_classifier.train(
        input_fn=lambda: input_fn(
            FLAGS.train_list, FLAGS.n_batch),  # train_input_fn,
        steps=100,             # 20,000回モデルを更新する
        hooks=[logging_hook])    # loggong_hookを引数に指定して、train中にhookされるようにする
    eval_results = mnist_classifier.evaluate(
        input_fn=lambda: input_fn(FLAGS.test_list, FLAGS.n_batch))
    print(eval_results)

    # # ここからinput_fnに移動予定
    # def generator(): return read_csv(FLAGS.train_list)  # generator
    #
    # dataset = tf.data.Dataset.from_generator(
    #     generator, (tf.string, tf.int32), (tf.TensorShape([]), tf.TensorShape([])))\
    #     .map(read_image)
    # dataset = dataset.shuffle(10000)
    # dataset = dataset.repeat(FLAGS.n_epoch)
    # dataset = dataset.batch(FLAGS.n_batch)
    # iterator = dataset.make_one_shot_iterator()
    # value = iterator.get_next()
    # # ここまでinput_fnに移動予定
    #
    # # Run-Graph
    # sess = tf.Session()
    # try:
    #     while True:
    #         img, label = sess.run(value)
    #         print("{}, {}".format(img.shape, label))
    #         print("img_range : {} - {}".format(img.min(), img.max()))
    # except tf.errors.OutOfRangeError:
    #     pass

    return 0


if __name__ == '__main__':
    tf.app.run()
