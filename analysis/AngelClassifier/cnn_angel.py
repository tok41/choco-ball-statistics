# -*- coding: utf-8 -*-

# Imports
import os
import tensorflow as tf
import make_image_def as mid

tf.logging.set_verbosity(tf.logging.INFO)

# command line arguments
tf.app.flags.DEFINE_string("log_dir", "output", "log directory")
tf.app.flags.DEFINE_string(
    "train_list", "image_list_train.csv", "image data list(csv)")
tf.app.flags.DEFINE_string(
    "test_list", "image_list_test.csv", "image data list(csv)")
tf.app.flags.DEFINE_integer("n_batch", 10, "mini batch size")
tf.app.flags.DEFINE_integer("n_epoch", 5, "number of epoch")
tf.app.flags.DEFINE_integer("valid_step", 2, "validation interval")

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
# ### Data Class
class DataSet:
    def __init__(self, file_path, n_batch):
        self.file_path = file_path
        self.n_batch = n_batch

    def get_input_op(self):
        """
        データ入力関数
        """
        def generator():
            return self.read_csv(self.file_path)  # generator
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.string, tf.int64),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([])))\
            .map(self.read_image)
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.n_batch)
        iterator = dataset.make_initializable_iterator()
        images, labels = iterator.get_next()
        self.iterator = iterator
        return images, labels

    def read_csv(self, filename):
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

    def read_image(self, image_file, label):
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
        return image, label


# ###########################
# ### Define Model
def cnn_classifier(images, reuse):
    """
    Model
    """
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.name_scope('input'):
            x = tf.reshape(images, [-1, 275, 370, 3])  # [-1, 1100, 1480, 3]
            tf.summary.image('input', x, 10)  # for visualize at tensor-board
        with tf.name_scope('conv1'):
            conv1 = tf.layers.conv2d(
                x, filters=32, kernel_size=(5, 5),
                padding='same', activation=tf.nn.relu)
        with tf.name_scope('pool1'):
            pool1 = tf.layers.max_pooling2d(
                conv1, pool_size=(5, 5), strides=(5, 5))
        with tf.name_scope('conv2'):
            conv2 = tf.layers.conv2d(
                pool1, filters=64, kernel_size=(3, 3),
                padding='same', activation=tf.nn.relu)
        with tf.name_scope('pool2'):
            pool2 = tf.layers.max_pooling2d(
                conv2, pool_size=(3, 3), strides=(3, 3))
            pool2_flat = tf.contrib.layers.flatten(pool2)
        with tf.name_scope('fc3'):
            fc3 = tf.layers.dense(pool2_flat, 1024)
        with tf.name_scope('output'):
            out = tf.layers.dense(fc3, 2)
    return out


# ###########################
# ### Evaluation Function
def evaluation(sess, features, labels):
    # Inference
    y_pred = cnn_classifier(features, reuse=True)
    # validate
    correct = tf.equal(tf.argmax(y_pred, 1), labels)
    n_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # buffer
    total_acc = 0.0
    total_features = 0
    try:
        while True:
            imgs, nc, ac = sess.run([features, n_correct, accuracy])
            total_acc += nc
            total_features += imgs.shape[0]
    except tf.errors.OutOfRangeError:
        pass
    return (total_acc/total_features)


def evaluate(features, labels):
    with tf.name_scope('evaluate_test'):
        # Inference
        y_pred = cnn_classifier(features, reuse=True)
        # validate
        correct = tf.equal(tf.argmax(y_pred, 1), labels)
        n_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return n_correct, accuracy


# ###########################
# ### Main
def main(argv):
    FLAGS = tf.app.flags.FLAGS

    # 画像リストファイルの作成
    createImageListFile(FLAGS.img_path, FLAGS.db_path,
                        FLAGS.train_list, FLAGS.test_list)

    # create dataset
    with tf.name_scope('train_input'):
        train_dataset = DataSet(FLAGS.train_list, n_batch=FLAGS.n_batch)
        train_images, train_labels = train_dataset.get_input_op()
    with tf.name_scope('test_input'):
        valid_dataset = DataSet(FLAGS.test_list,  n_batch=FLAGS.n_batch)
        valid_images, valid_labels = valid_dataset.get_input_op()

    # Inference
    y_pred = cnn_classifier(train_images, reuse=False)

    # loss
    with tf.name_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(
            tf.one_hot(train_labels, depth=2),
            y_pred)
        tf.summary.scalar('loss', loss)

    # training
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

    # evaluating
    with tf.name_scope('evaluate'):
        correct = tf.equal(tf.argmax(y_pred, 1), train_labels)
        n_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # evaluate test data
    n_correct_valid, accuracy_valid = evaluate(valid_images, valid_labels)
    tf.summary.scalar('accuracy_valid', accuracy_valid)

    # initializer
    init = tf.global_variables_initializer()
    # Run-Graph
    sess = tf.Session()
    # log merged
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    # initialize
    sess.run(init)
    for epoch in range(FLAGS.n_epoch):
        sess.run(train_dataset.iterator.initializer)
        sess.run(valid_dataset.iterator.initializer)
        loss_epoch = 0.0
        acc_epoch = 0.0
        n_images = 0
        try:
            while True:
                # training operation
                _, imgs, tmp_loss, tmp_nc, tmp_acc, summary = sess.run(
                    [train, train_images, loss, n_correct, accuracy, merged])
                train_writer.add_summary(summary, epoch)
                loss_epoch += tmp_loss
                acc_epoch += tmp_nc
                n_images += imgs.shape[0]

        except tf.errors.OutOfRangeError:
            loss_epoch = loss_epoch/n_images
            acc_epoch = acc_epoch/n_images
            print('epoch:{}/{}, images:{}, mean_loss:{}, mean_accuracy:{}'.format(
                epoch, FLAGS.n_epoch, n_images, loss_epoch, acc_epoch))
            pass

        if epoch % FLAGS.valid_step == 0:
            # validating operation
            sess.run(train_dataset.iterator.initializer)
            sess.run(valid_dataset.iterator.initializer)
            total_acc = 0.0
            total_features = 0
            try:
                while True:
                    imgs, nc, ac, summary_valid = sess.run(
                        [valid_images, n_correct_valid, accuracy_valid, merged])
                    test_writer.add_summary(summary_valid, epoch)
                    total_acc += nc
                    total_features += imgs.shape[0]
            except tf.errors.OutOfRangeError:
                print('accuracy[epoch:{}]:{}'.format(
                    epoch, total_acc/total_features))
                pass
            # acc_train = evaluation(sess, train_images, train_labels)
            # acc_valid = evaluation(sess, valid_images, valid_labels)
            # print('accuracy[epoch:{}]:{}, {}'.format(
            #     epoch, acc_train, acc_valid))

    return 0


if __name__ == '__main__':
    tf.app.run()
