import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as model
from torchsummary import summary
from collections import OrderedDict

from dataset import voc_loader as loader
import torchvision.transforms as transforms


class Test_model(nn.Module):
    def __init__(self):
        super(Test_model, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU())),
            ('conv2', nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU())),
            ('conv3', nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=6, dilation=6)))

        ]))

    def forward(self, x):
        x = self.features(x)
        return x


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

    def forward(self, x):
        for layer in self.vgg:
            x = layer(x)

        for ex_layer in self.extras:

            x = ex_layer(x)

        return x


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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SSD().to(device)
    print(net.vgg)
    summary(net, (3, 300, 300))

    # net = torchvision.modecls.vgg16(pretrained=True).to(device)
    # NET2 = Test_model().to(device)
    # # summary(net, (3, 300, 300))
    #
    # vgg_conv4_3 = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features.children())[: -8]).to(device)
    # # summary(vgg_conv4_3, (3, 300, 300))
    #
    # summary(NET2, (3, 300, 300))


    # for i, (name, module) in enumerate(net.named_children()):
    #     print(i)
    #     print(name)
    #     print(module)
        # md = module

        # for i, layer in enumerate(module):
        #     if i == 21:
        #         print(layer)






