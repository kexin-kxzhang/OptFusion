import glob
import torch
import numpy as np
import tensorflow as tf


class CriteoLoader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 39
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }

    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label']
        files = glob.glob(self.tfrecord_path + "{}*".format(data_type))
        if not files:
            raise ValueError("no criteo files")
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x, y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            yield x, y

class AvazuLoader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 24
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }

    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label']
        files = glob.glob(self.tfrecord_path + "{}*".format(data_type))
        if not files:
            raise ValueError("no avazu files")
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x, y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            yield x, y

class KDD12loader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 11
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }

    def get_data(self, data_type, batch_size = 1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label']
        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
                        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x,y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            yield x, y

def count_examples(data_loader):
    train_iter = data_loader.get_data("train", batch_size=1000)
    step = 0
    for x, y in train_iter:
        step += 1
    print("train examples: {}000".format(step))

    val_iter = data_loader.get_data("valid", batch_size=1000)
    step = 0
    for x, y in val_iter:
        step += 1
    print("valid examples: {}000".format(step))

    test_iter = data_loader.get_data("test", batch_size=1000)
    step = 0
    for x, y in test_iter:
        step += 1
    print("test examples: {}000".format(step))

def count_features(data_loader):
    max_id = 0
    train_iter = data_loader.get_data("train", batch_size=10000)
    step = 0
    for x, y in train_iter:
        if max_id < np.max(x.cpu().numpy().tolist()):
            max_id = np.max(x.cpu().numpy().tolist())
        step += 1
        if step % 1000 == 0:
            print("step: {} ".format(step))
    print("train max_id: {} ".format(max_id))

    val_iter = data_loader.get_data("valid", batch_size=10000)
    step = 0
    for x, y in val_iter:
        if max_id < np.max(x.cpu().numpy().tolist()):
            max_id = np.max(x.cpu().numpy().tolist())
        step += 1
        if step % 1000 == 0:
            print("step: {} ".format(step))
    print("valid max_id: {} ".format(max_id))

    test_iter = data_loader.get_data("test", batch_size=10000)
    step = 0
    for x, y in test_iter:
        if max_id < np.max(x.cpu().numpy().tolist()):
            max_id = np.max(x.cpu().numpy().tolist())
        step += 1
        if step % 1000 == 0:
            print("step: {} ".format(step))
    print("test max_id: {} ".format(max_id))


def test_criteo_loader(data_path):
    data_loader = CriteoLoader(data_path)
    print("criteo data_path: {} ".format(data_path))

    count_examples(data_loader)
    count_features(data_loader)
    # criteo_2
    # train examples: 36672500
    # valid examples: 4585000
    # test examples: 4585000
    # train max_id: 6780367
    # valid max_id: 6780367
    # test max_id: 6780367

def test_avazu_loader(data_path):

    data_loader = AvazuLoader(data_path)
    print("avazu data_path: {} ".format(data_path))
    count_examples(data_loader)
    count_features(data_loader)
    # threshold_5
    # train examples: 32343200
    # valid examples: 4043000
    # test examples: 4043000
    # train max_id: 1544271
    # valid max_id: 1544271
    # test max_id: 1544271
    # threshold_2
    # train examples: 32344000
    # valid examples: 4043000
    # test examples: 4043000


if __name__ == "__main__":
    test_avazu_loader(data_path="../dataset/avazu/threshold_1/")




