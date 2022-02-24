import random
random.seed(0)

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

tf.random.set_seed(0)

from dataset import get_tfrec_dataset, RES, BATCH_SIZE, SPLIT

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

train_dataset, val_dataset, total_file_count = get_tfrec_dataset()

CLASSES = 88171

def get_sequential():
    return keras.Sequential([layers.Conv2D(64, 5, padding='same', activation='gelu'),
                             layers.AveragePooling2D(),
                             layers.BatchNormalization(),
                             layers.Conv2D(128, 5, padding='same', activation='gelu'),
                             layers.AveragePooling2D(),
                             layers.BatchNormalization(),
                             layers.Conv2D(256, 3, padding='same', activation='gelu'),
                             layers.AveragePooling2D(),
                             layers.BatchNormalization(),
                             layers.Conv2D(512, 3, padding='same', activation='gelu'),
                             layers.BatchNormalization(),
                             layers.GlobalAveragePooling2D()])


def get_model():
    backbone = keras.applications.EfficientNetB0(include_top=False, weights=None, pooling='avg')
    #backbone = get_sequential()

    inp = keras.Input((*RES, 3))
    x = backbone(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    #x1 = layers.Dense(5000, activation='softmax')(x)

    #model = keras.Model(inp, x1)
    #model.load_weights("weights5k.h5")

    x2 = layers.Dense(10000, activation='softmax')(x)
    model = keras.Model(inp, x2)
    model.load_weights("weights10k.h5")

    x3 = layers.Dense(20000, activation='softmax')(x)
    model = keras.Model(inp, x3)

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                       optimizer=keras.optimizers.Adam(),
                       metrics=['accuracy'])
    model.summary()

    return model

tboard = keras.callbacks.TensorBoard(log_dir="logs/20k/", update_freq="batch")
checkpoint = keras.callbacks.ModelCheckpoint("weights20k.h5", save_weights_only=True, save_freq='epoch', verbose=1)

model = get_model()
print(f"FILE COUNT: {total_file_count}")
model.fit(train_dataset, validation_data=val_dataset, steps_per_epoch=int(SPLIT*total_file_count)//BATCH_SIZE, epochs=1, callbacks=[tboard, checkpoint])
