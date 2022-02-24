import os
import json
import pandas as pd
import tensorflow as tf

from random import shuffle

def get_word(filepath):
    s = filepath.split('/')[-1].split('_')
    return s[-2].lower()

def gen_word_freq():
    data_path = "/home/joey/python/tensorflow/projects/mjsynth/data/images/"
    p1_folders = [data_path+s+'/' for s in os.listdir(data_path)]

    i = 0
    word_counts = dict()
    for p1_folder in p1_folders:
        p2_folders = [p1_folder+s+'/' for s in os.listdir(p1_folder)]
        for p2_folder in p2_folders:
            filenames = [p2_folder+s for s in os.listdir(p2_folder)]
            for filename in filenames:
                name = get_word(filename)

                value = word_counts.get(name, [0, []])
                value[0] += 1
                value[1].append(filename)

                word_counts[name] = value

                print(f"{i+1}        \r", end='')
                i += 1

    print("-------------------")

    with open("word_freq.json", "w") as f:
        json.dump(word_counts, f)

    print(f"{i+1} words logged")

def gen_topnk():
    print("loading word counts...")
    with open("word_freq.json", "r") as f:
        word_counts = json.load(f)

    print("sorting word counts...")
    sorted_word_counts = { k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1][0], reverse=True) }

    intervals = [5000*i for i in range(1, 18)]

    for interval in intervals:
        topnk = dict()
        for x, (k, v) in enumerate(sorted_word_counts.items()):
            if x == interval:
                break

            topnk[k] = v

            print(f"{x}       \r", end='')

        with open(f"top{interval//1000}k.json", "w") as f:
            json.dump(topnk, f)

        print(f"top{interval//1000}k.json")

sizes = [0, 617912, 1200276, 1765543, 2318920, 2861901, 3395462, 3920709, 4437902, 4946609, 5447885, 5940653, 6425171, 6900897, 7366941, 7821812, 8262369, 8681829]

def gen_tfrecs():
    def preprocess(filepath):
        image = tf.io.read_file(filepath)
        #image = tf.io.decode_jpeg(image)

        return image

    def _bytes_feature(value):
      """Returns a bytes_list from a string / byte."""
      if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(image, label):
        feature = {
            'image': _bytes_feature(image),
            'label': _int64_feature(label)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

        return example_proto.SerializeToString()

    print("loading word counts...")
    with open("data/json/word_freq.json", "r") as f:
        word_counts = json.load(f)

    print("sorting word counts...")
    sorted_word_counts = { k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1][0], reverse=True) }
    labeled_words = dict()
    filenames = []
    for i, (word, value) in enumerate(sorted_word_counts.items()):
        labeled_words[word] = i
        filenames += value[1]

        print(f"{i+1}      \r", end='')

    for x in range(len(sizes)):
        record_count = 0
        written = 0
        writer = tf.io.TFRecordWriter(f"data/tfrec/{x}/{record_count}.tfrec")
        if x == len(sizes)-1:
            fn = filenames[sizes[x]:]
        else:
            fn = filenames[sizes[x]:sizes[x+1]]

        shuffle(fn)

        for i, filename in enumerate(fn):
            print(f"{written} of {len(fn)} images written     \r", end='')
            if (i+1) % 10000 == 0:
                record_count += 1
                del writer
                writer = tf.io.TFRecordWriter(f"data/tfrec/{x}/{record_count}.tfrec")

            image = preprocess(filename)
            label = labeled_words[get_word(filename)]
            example = serialize_example(image, label)

            writer.write(example)
            written += 1

def gen_csv():
    intervals = [5000*i for i in range(1, 18)]
    for interval in intervals:
        print("--------------------")
        print("loading word counts...")
        with open(f"data/json/top{interval//1000}k.json", "r") as f:
            word_counts = json.load(f)

        print("sorting word counts...")
        sorted_word_counts = { k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1][0], reverse=True) }
        labeled_words = dict()
        filenames = []
        for i, (word, value) in enumerate(sorted_word_counts.items()):
            labeled_words[word] = i
            filenames += value[1]
                
            print(f"{i+1}      \r", end='')

        labels_for_filenames = [labeled_words[get_word(f)] for f in filenames]

        csv = { 'filename': filenames, 'label': labels_for_filenames }
        df = pd.DataFrame.from_dict(csv)
        df.to_csv(f"data/csv/top{interval//1000}k.csv", index=False)
        print(f"top{interval//1000}.csv")

gen_tfrecs()
