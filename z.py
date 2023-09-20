import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models


class Model(nn.Module):
    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu
        model = models.mobilenet_v2(pretrained=True)
        for parma in model.parameters():
            parma.requires_grad = False
        self.model = model
        self.model.classifier[1] = nn.Linear(1280, 21, bias=True)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        # x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.model(x)
        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


if __name__ == '__main__':
    VGG16 = Model()
    input = torch.ones((1, 3, 500, 500))
    output = VGG16(input)
    print(output.shape)
    print(VGG16)
