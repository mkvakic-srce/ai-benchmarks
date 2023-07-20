# sources
# - https://lambdalabs.com/blog/tensorflow-2-0-tutorial-05-distributed-training-multi-node
# - http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-tf-multi-eng.html
# - https://keras.io/guides/distributed_training

import os
import sys
import time
import numpy as np

import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, batch_size, samples, epochs, gpus):
        self.start = time.time()
        self.batch_size = batch_size
        self.samples = samples
        self.epochs = epochs
        self.gpus = gpus

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()
        return

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time()-self.start
        img_sec = self.samples/elapsed
        nranks =  int(os.environ['PMI_SIZE'])
        ngpus = len(self.gpus)
        if os.environ['PMI_RANK'] == '0':
            lines = [
                'batch_size %s' % self.batch_size,
                'samples %s' % self.samples,
                'epochs %s' % self.epochs,
                'elapsed %0.2f' % elapsed,
                'imgsec %0.2f' % img_sec,
                'local_world_size %s' % ngpus,
                'world_size %s' % nranks,
            ]
            print(','.join(lines))
        return

def main():


    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    batch_size = int(os.environ['BATCH_SIZE'])
    samples = batch_size*100
    epochs = int(os.environ['EPOCHS'])

    data = tf.random.uniform([samples, 224, 224, 3])
    target = tf.random.uniform([samples, 1], minval=0, maxval=999, dtype=tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((data, target)).batch(batch_size)

    with strategy.scope():
        model = tf.keras.applications.ResNet50(weights=None)
        model.compile(optimizer=tf.optimizers.SGD(),
                      loss=tf.losses.SparseCategoricalCrossentropy(),
                      metrics='accuracy')

    callbacks = [CustomCallback(batch_size, samples, epochs, gpus)]
    model.fit(dataset,
              callbacks=callbacks,
              verbose=0,
              epochs=epochs,
              batch_size=batch_size)

if __name__ == '__main__':
    main()
