import torch
from torchvision.models import resnet
from pytorch2caffe import pytorch2caffe

name='resnet18'
resnet18=resnet.resnet18()
resnet18.eval()
dummy_input=torch.ones([1,3,224,224])


pytorch2caffe.trans_net(resnet18, dummy_input, name)
pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))
