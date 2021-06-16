import torch
from torchvision.models import resnet

name='resnet50'
resnet18=resnet.resnet50()
resnet18.eval()
dummy_input=torch.ones([1,3,256,256])


torch.save(resnet18.state_dict(),name)
