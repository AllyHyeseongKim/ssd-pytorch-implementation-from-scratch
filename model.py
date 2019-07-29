import torch
import torch.nn as nn
import torchvision.models as model

from dataset import voc_loader as loader
import torchvision.transforms as transforms


class RPN(nn.Module):
    def __init__(self):

        super(RPN, self).__init__()
        # vgg 16 가져오기
        vgg_net = model.vgg16(pretrained=True)
        modules = list(vgg_net.children())[:-1]  # delete the last fc layer.
        modules = list(modules[0])[:-1]  # delete the last pooling layer

        self.vggnet = nn.Sequential(*modules)
        for module in list(self.vggnet.children())[:10]:

            # print("fix weight", module)
            # 10까지는 학습이 안되도록 설정!

            for param in module.parameters():
                param.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU()
        )

        # 9 anchor * 2 classfier (object or non-object) each grid
        self.conv1 = nn.Conv2d(512, 2 * 9, kernel_size=1, stride=1)

        # 9 anchor * 4 coordinate regressor each grids
        self.conv2 = nn.Conv2d(512, 4 * 9, kernel_size=1, stride=1)
        self.softmax = nn.Softmax()

    def forward(self, images):

        features = self.vggnet(images)
        features = self.conv(features)
        logits, rpn_bbox_pred = self.conv1(features), self.conv2(features)

        height, width = features.size()[-2:]
        logits = logits.squeeze(0).permute(1, 2, 0).contiguous()  # (1, 18, H/16, W/16) => (H/16 ,W/16, 18)
        logits = logits.view(-1, 2)  # (H/16 ,W/16, 18) => (H/16 * W/16 * 9, 2)

        rpn_cls_prob = self.softmax(logits)
        rpn_cls_prob = rpn_cls_prob.view(height, width, 18)  # (H/16 * W/16 * 9, 2)  => (H/16 ,W/16, 18)
        rpn_cls_prob = rpn_cls_prob.permute(2, 0, 1).contiguous().unsqueeze(0) # (H/16 ,W/16, 18) => (1, 18, H/16, W/16)

        return rpn_bbox_pred, rpn_cls_prob, logits


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((600, 1000)),
            transforms.ToTensor()
         ])

    # 얘는 img 에 대해서만 하는거임
    trainset = loader.VOC_loader(transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    region_proposal_net = RPN().cuda()

    for epoch in range(10):
        for i, (images, labels, _) in enumerate(trainloader):

            images = images.cuda()
            # labels = labels.cuda()

            bbox, cls, f = region_proposal_net(images)

            print("bbox: ", bbox)
            print("cls: ", cls)
            # print(labels)


