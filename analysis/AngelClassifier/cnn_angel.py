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
# ### Model
def cnn_model_fn(features, labels, mode):
    """
    Model Function
    """
    # ### Input Layer
    input_layer = tf.reshape(features, [-1, 1100, 1480, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[10, 10],
        strides=(1, 1),
        padding="same",  # 入力と同じ大きさになるように0パディングする
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[10, 10], strides=10)
    # pool1 [batch, 110, 148, 32]]
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # conv2 [batch_size, 110, 148, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=5)
    # pool2 [batch_size, 28, 37, 64] ???
    # Fully-Connected Layer
    pool2_flat = tf.reshape(pool2, [-1, 163328])
    fc3 = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(
        inputs=fc3, units=2, activation=tf.nn.relu)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# ###########################
# ### Main
def main(argv):
    FLAGS = tf.app.flags.FLAGS

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
