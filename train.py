import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import voc_loader as loader
import model
import torch.nn as nn
import torch.optim as optim
import visdom
import time
import utils.anchors as anchor


# 1. device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. visdom
# vis = visdom.Visdom()

# 3. data set 정의
transform = transforms.Compose(
    [
        transforms.Resize((600, 1000)),
        transforms.ToTensor(),
    ])

train_set = loader.VOC_loader(transform=transform)
# 4. data loader 정의
train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

# 5. model 정의
net = model.RPN().to(device)

# 6. loss 정의


# 7. optimizer 정의
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# 8. anchor 정의 anchor : (H/16 * W/16 * 9, 4)
all_anchors_boxes = anchor.get_anchors((14, 14), anchor)

# 9. train
for epoch in range(10):
    for i, (images, labels, _) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        bbox, cls, f = net(images)
