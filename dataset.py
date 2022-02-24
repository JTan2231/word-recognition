import os
import json
import pandas as pd
import matplotlib.pyplot as pl
import tensorflow as tf

from random import sample, randrange, shuffle

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
RES = (48, 174) # average x, y
SPLIT = 0.9
LIMIT = True
TOP = 5000

def get_paths(filepath):
    fp = filepath
    if fp[-1] != '/':
        fp += '/'

    return [fp+s for s in os.listdir(fp)]

def preprocess(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, RES)

    return image, label

def gen_dataset(files, augment=None):
    dataset = tf.data.Dataset.from_tensor_slices(files).map(preprocess, num_parallel_calls=AUTOTUNE)
    if augment != None:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)

    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

def gen_tfrec_dataset(files, augment=None):
    def tfrec_preprocess(example):
        feature = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64)
        }
        
        example = tf.io.parse_single_example(example, feature)
        image = example['image']
        label = example['label']
        
        image = tf.io.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, RES)

        return image, label

    dataset = tf.data.TFRecordDataset(files).map(tfrec_preprocess, num_parallel_calls=AUTOTUNE)
    if augment is not None:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)

    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE).repeat()

def get_tfrec_dataset():
    p = "/home/joey/python/tensorflow/projects/mjsynth/data/tfrec/"
    folders = [p+'0/', p+'1/', p+'2/', p+'3/']
    filenames = [f+s for f in folders for s in os.listdir(f)]
    shuffle(filenames)

    split = int(SPLIT*len(filenames))
    train_split = filenames[:split]
    val_split = filenames[split:]

    train_dataset = gen_tfrec_dataset(train_split)
    val_dataset = gen_tfrec_dataset(val_split)

    return train_dataset, val_dataset, 10000*len(filenames)

def get_dataset(filename="top5k.csv"):
    """with open("top10k_relabeled.json", "r") as top5k:
        t5k = json.load(top5k)

    images = { 'filepaths': [], 'label': [] }
    for key, value in t5k.items():
        images['filepaths'] += value['filepaths']
        images['label'] += [value['label'] for _ in range(len(value['filepaths']))]

    data = list(zip(images['filepaths'], images['label']))"""

    df = pd.read_csv(f"data/csv/{filename}")
    data = list(zip(list(df['filename']), list(df['label'])))
    print("Shuffling...")
    shuffle(data)

    images, labels = zip(*data)

    split = int(SPLIT*len(images))
    train_images = list(images[:split])#[images.pop(randrange(len(images))) for _ in range(int(SPLIT*len(images)))]
    train_labels = list(labels[:split])

    val_images = list(images[split:])
    val_labels = list(labels[split:])

    print("Splitting into (filepath, label)...")
    train_data = (train_images, train_labels)
    val_data = (val_images, val_labels)

    train_dataset = gen_dataset(train_data)
    val_dataset = gen_dataset(val_data)

    return train_dataset, val_dataset, len(images)
