import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as model
from torchsummary import summary
from collections import OrderedDict
import torch.nn.init as init
from dataset import voc_loader as loader
import torchvision.transforms as transforms


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSD(nn.Module):
    def __init__(self, pretrained=True):
        super(SSD, self).__init__()

        # module list 로 둔 이유는
        self.vgg = nn.ModuleList(vgg(cfg['300'], 3))
        # self.vgg = vgg(cfg['300'], 3)
        if pretrained:
            vgg_weights = torch.load('./vgg16_reducedfc.pth')
            self.vgg.load_state_dict(vgg_weights)

        conv_8_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        conv_8_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)
        conv_9_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
        conv_9_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        conv_10_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        conv_10_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,)
        conv_11_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        conv_11_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)

        self.extras = nn.ModuleList([conv_8_1,  nn.ReLU(True),
                                     conv_8_2,  nn.ReLU(True),
                                     conv_9_1,  nn.ReLU(True),
                                     conv_9_2,  nn.ReLU(True),
                                     conv_10_1,  nn.ReLU(True),
                                     conv_10_2,  nn.ReLU(True),
                                     conv_11_1,  nn.ReLU(True),
                                     conv_11_2,  nn.ReLU(True)])

        self.L2Norm = L2Norm(512, 20)
        self.dbox = dbox['300']

        loc4_3 = nn.Conv2d(in_channels=512, out_channels=self.dbox[0] * 4, kernel_size=3, padding=1)
        loc7 = nn.Conv2d(in_channels=1024, out_channels=self.dbox[1] * 4, kernel_size=3, padding=1)
        loc8_2 = nn.Conv2d(in_channels=512, out_channels=self.dbox[2] * 4, kernel_size=3, padding=1)
        loc9_2 = nn.Conv2d(in_channels=256, out_channels=self.dbox[3] * 4, kernel_size=3, padding=1)
        loc10_2 = nn.Conv2d(in_channels=256, out_channels=self.dbox[4] * 4, kernel_size=3, padding=1)
        loc11_2 = nn.Conv2d(in_channels=256, out_channels=self.dbox[5] * 4, kernel_size=3, padding=1)

        cls4_3 = nn.Conv2d(in_channels=512, out_channels=self.dbox[0] * 21, kernel_size=3, padding=1)
        cls7 = nn.Conv2d(in_channels=1024, out_channels=self.dbox[1] * 21, kernel_size=3, padding=1)
        cls8_2 = nn.Conv2d(in_channels=512, out_channels=self.dbox[2] * 21, kernel_size=3, padding=1)
        cls9_2 = nn.Conv2d(in_channels=256, out_channels=self.dbox[3] * 21, kernel_size=3, padding=1)
        cls10_2 = nn.Conv2d(in_channels=256, out_channels=self.dbox[4] * 21, kernel_size=3, padding=1)
        cls11_2 = nn.Conv2d(in_channels=256, out_channels=self.dbox[5] * 21, kernel_size=3, padding=1)

        self.loc = nn.ModuleList([loc4_3,
                                  loc7,
                                  loc8_2,
                                  loc9_2,
                                  loc10_2,
                                  loc11_2])

        self.cls = nn.ModuleList([cls4_3,
                                   cls7,
                                   cls8_2,
                                   cls9_2,
                                   cls10_2,
                                   cls11_2])

    def forward(self, x):
        """
        여기서 output 은 결과적으로 loc prediction 과 cls prediction 입니다.
        :param x: input 이미지 batch data
        :return: loc, cls 입니다.
        """

        features = []  # 필요한 feature 를 담는 부분 : conv4_3 , ... , conv11_2
        loc = []
        cls = []

        for i in range(22):
            x = self.vgg[i](x)

        conv4_3 = self.L2Norm(x)
        features.append(conv4_3) # conv 4_3

        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)

        conv7 = x
        features.append(conv7)  # conv 7

        for i, ex_layer in enumerate(self.extras):
            x = ex_layer(x)
            # if i % 4 == 3:  # 4개의 한번꼴로
            #     features.append(x)
            if i == 3:
                conv8_2 = x    # conv 8_2
                features.append(conv8_2)
            if i == 7:
                conv9_2 = x    # conv 9_2
                features.append(conv9_2)
            if i == 11:
                conv10_2 = x    # conv 10_2
                features.append(conv10_2)
            if i == 15:
                conv11_2 = x    # conv 11_2
                features.append(conv11_2)

        for feature, l, c in zip(features, self.loc, self.cls):

            loc.append(l(feature).permute(0, 2, 3, 1).contiguous())
            cls.append(c(feature).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # torch.Size([B, 34928])
        cls = torch.cat([o.view(o.size(0), -1) for o in cls], 1)  # torch.Size([B, 183372])

        loc = loc.view(loc.size(0), -1, 4),  # torch.Size([B, 8732, 4])
        cls = cls.view(cls.size(0), -1, 21),  # torch.Size([B, 8732, 21])

        return (loc, cls)


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, input_ch):
    '''
    fc-reduced vgg 사용하기 위해서 vgg 만드는 부분!
    :param cfg: str arr : vgg 의 구성을 나타내는 string array
    :return: nn.module 들의 layer 리스트!
    '''
    layers = []
    in_channels = input_ch

    # feature extractor

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
            layers += [nn.ReLU(inplace=True)]
            in_channels = v

    # classifier

    layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(512, 1024, kernel_size=3, dilation=6, padding=6)]
    layers += [nn.ReLU(True)]
    layers += [nn.Conv2d(1024, 1024, kernel_size=1)]
    layers += [nn.ReLU(True)]

    # return nn.Sequential(*layers)
    return layers


cfg = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}

dbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor()
         ])

    # 얘는 img 에 대해서만 하는거임

    root_dir = "D:\Data\\voc\\2012"
    trainset = loader.VOC_loader(root_dir, transform=transform)

    # # img
    # # 이미지 하나씩 가져오는 부분
    # for i in range(len(trainset)):
    #     image, annotation, scale = trainset[i]
    #
    #     img = np.array(image)
    #     img = img.transpose((1, 2, 0))
    #     img *= 255
    #     img = img[:, :, ::-1].astype(np.uint8)
    #
    #     annotation = np.array(annotation)
    #     annotation[:, 0] *= scale[1]
    #     annotation[:, 2] *= scale[1]
    #     annotation[:, 1] *= scale[0]
    #     annotation[:, 3] *= scale[0]
    #
    #     for i in range(len(annotation)):
    #         img = cv2.rectangle(img, (int(annotation[i][0]), int(annotation[i][1])),
    #                             (int(annotation[i][2]), int(annotation[i][3])), (255, 255, 0), 3)
    #
    #     cv2.imshow('image', img)
    #     cv2.waitKey(0)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SSD().to(device)

    for epoch in range(10):

        running_loss = 0.0
        for i, (images, labels, _) in enumerate(trainloader):
            images = images.to(device)
            # labels = labels.to(device)

            out = net(images)
            print(out)
            out = 1








