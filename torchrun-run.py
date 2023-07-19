# source
# - https://pytorch.org/tutorials/intermediate/dist_tuto.html
# - https://pytorch.org/vision/main/generated/torchvision.datasets.FakeData.html
# - https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/#setting-up-the-same-model-with-distributeddataparallel

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.models import resnet50
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

def main():

    # vars
    epochs = int(os.environ['EPOCHS'])
    batch_size = int(os.environ['BATCH_SIZE'])
    samples = 256*1000
    epochs = 3

    # init
    dist.init_process_group("nccl")
    rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])

    # model
    model = resnet50(weights=None)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # data
    dataset = FakeData(samples,
                       num_classes=1000,
                       transform=ToTensor())
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        shuffle=False,
                        num_workers=1,
                        pin_memory=True)

    # train
    for epoch in range(epochs):
        start = time.time()
        for batch, (images, labels) in enumerate(loader):
            images = images.to(rank)
            labels = labels.to(rank)
            outputs = model(images)
            classes = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (global_rank == 0) and (batch%10 == 0):
                print('epoch: %3d, batch: %3d/%3d, loss: %0.4f' % (epoch+1,
                                                                   batch,
                                                                   len(loader),
                                                                   loss.item()))
        if (global_rank == 0):
            elapsed = time.time()-start
            img_sec = samples/elapsed
            lines = [
                'batch_size %s' % batch_size,
                'samples %s' % samples,
                'epochs %s' % epochs,
                'elapsed %0.2f' % elapsed,
                'imgsec %0.2f' % img_sec,
                'local_world_size %s' % os.environ['LOCAL_WORLD_SIZE'],
                'world_size %s' % os.environ['WORLD_SIZE'],
            ]
            print(','.join(lines))

    # clean
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
