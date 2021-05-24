#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch2caffe import pytorch2caffe

from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg16',help='net type')
    parser.add_argument('-weights', type=str, default='./checkpoint/vgg16/Wednesday_19_May_2021_14h_43m_41s/vgg16-10-regular.pth', help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=10, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    #for n_iter, (image, label) in enumerate(cifar100_test_loader):
        #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
        # print(image.shape)

    name = './torch2caffe/vgg_caffe'
    #dummy_input = torch.randn(1, 3, 32, 32)
    dummy_input=torch.ones([1,3,32,32])
    pytorch2caffe.trans_net(net, dummy_input, name)
    pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))
