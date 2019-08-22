import torch
import time
import torchvision.transforms as transforms
import torch.utils.data as data
from dataset import voc_loader as loader
import model
import torch.optim as optim
from loss import MultiBoxLoss
from model import SSD
from utils import *
import visdom

# 1. device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. visdom
vis = visdom.Visdom()

# 3. data set 정의
transforms_list = photometric_distort()
transform = transforms.Compose(transforms_list)
root_dir = "D:\Data\\voc\\2012"
train_set = loader.VOC_loader(root_dir, transform=transform)
# 4. data loader 정의
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=32,
                                           collate_fn=train_set.collate_fn,
                                           shuffle=True,
                                           num_workers=0)

# 5. model 정의
net = SSD().to(device)
net.train()

# 6. loss 정의
criterion = MultiBoxLoss().to(device)

# 7. optimizer 정의
optimizer = optim.Adam(net.parameters(), lr=0.001)
total_step = len(train_loader)

# 8. train
for epoch in range(30):

    epoch_time = time.time()
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = [l.to(device) for l in labels]
        # labels = labels.to(device)

        optimizer.zero_grad()
        bbox, cls = net(images)
        loss = criterion(bbox[0], cls[0], labels)

        vis.line(X=torch.ones((1, 1)).cpu() * i + epoch * train_set.__len__() / 32,
                 Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                 win='loss',
                 update='append',
                 opts=dict(xlabel='step',
                           ylabel='Loss',
                           title='training loss',
                           legend=['Loss'])
                 )

        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}'
                  .format(epoch + 1, 30, i + 1, total_step, loss.item(), time.time() - epoch_time))

        # step 별로 저장
        # torch.save(net.state_dict(), './saves/ssd.{}.{}.ckpt'.format(epoch, i + 1))

    # 각 epoch 별로 저장
    torch.save(net.state_dict(), './saves/ssd.{}.pth.tar'.format(epoch + 1))





