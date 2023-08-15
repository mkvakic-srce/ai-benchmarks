# sources
# - https://lambdalabs.com/blog/tensorflow-2-0-tutorial-05-distributed-training-multi-node
# - http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-tf-multi-eng.html
# - https://keras.io/guides/distributed_training

import os
import sys
import time
import numpy as np

import tensorflow as tf

import horovod
import horovod.tensorflow.keras as hvd

class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, batch_size, samples, epochs, gpus, size):
        self.start = time.time()
        self.batch_size = batch_size
        self.samples = samples
        self.epochs = epochs
        self.gpus = gpus
        self.size = size

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()
        return

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time()-self.start
        img_sec = self.samples/elapsed*self.size
        ngpus = len(self.gpus)
        nranks = os.environ['PMI_SIZE'] if 'PMI_SIZE' in os.environ else os.environ['OMPI_COMM_WORLD_SIZE']
        rank = os.environ['PMI_RANK'] if 'PMI_RANK' in os.environ else os.environ['OMPI_COMM_WORLD_RANK']
        if rank == '0':
            lines = [
                'batch_size %s' % self.batch_size,
                'samples %s' % self.samples,
                'epochs %s' % self.epochs,
                'elapsed %0.2f' % elapsed,
                'imgsec %0.2f' % img_sec,
                'local_world_size %s' % ngpus,
                'world_size %s' % self.size,
            ]
            print(','.join(lines))
        return

def main():

    # vars
    batch_size = int(os.environ['BATCH_SIZE'])
    samples = batch_size*10
    epochs = int(os.environ['EPOCHS'])

    # horovod
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # data
    data = tf.random.uniform([samples, 224, 224, 3])
    target = tf.random.uniform([samples, 1], minval=0, maxval=999, dtype=tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((data, target))
    dataset = dataset.batch(batch_size)

    # model
    loss = tf.losses.SparseCategoricalCrossentropy()
    optimizer = tf.optimizers.SGD()
    optimizer_distributed = hvd.DistributedOptimizer(optimizer,
                                                     compression=hvd.Compression.none)
    model = tf.keras.applications.ResNet50(weights=None)
    model.compile(loss=loss,
                  optimizer=optimizer_distributed,
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)

    # fit
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                 hvd.callbacks.MetricAverageCallback(),
                 CustomCallback(batch_size,
                                samples,
                                epochs, gpus,
                                hvd.size())]
    verbose = 2 if hvd.rank() == 0 else 0
    model.fit(dataset,
              callbacks=callbacks,
              verbose=verbose,
              epochs=epochs)

if __name__ == '__main__':
    main()
