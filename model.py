import torch
import torch.nn as nn
import torchvision
import torchvision.models as model

from dataset import voc_loader as loader
import torchvision.transforms as transforms


class AlexNetConv4(nn.Module):
    def __init__(self):
        super(AlexNetConv4, self).__init__()


    def forward(self, x):
        x = self.features(x)
        return x



class SSD(nn.Module):
    def __init__(self):
        super(SSD, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(model.vgg16_bn(pretrained=True).features.children())[:-3]
        )

    def forward(self, x):
        self.vgg_net(x)
        return




if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((600, 1000)),
            transforms.ToTensor()
         ])

    # 얘는 img 에 대해서만 하는거임

    root_dir = "D:\Data\\voc\\2012"
    trainset = loader.VOC_loader(root_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

    net = SSD()

    for i, (name, module) in enumerate(net.named_children()):
        # print(i)
        # print(name)
        # print(module)
        # md = module

        for i, layer in enumerate(module):
            if i == 3:
                print(layer)








    # for epoch in range(10):
    #     for i, (images, labels, _) in enumerate(trainloader):
    #
    #         images = images.cuda()
    #         # labels = labels.cuda()
    #
    #         bbox, cls, f = region_proposal_net(images)
    #
    #         print("bbox: ", bbox)
    #         print("cls: ", cls)
    #         # print(labels)


