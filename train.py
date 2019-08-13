import torch
import time
import torchvision.transforms as transforms
import torch.utils.data as data
from dataset import voc_loader as loader
import model
import torch.optim as optim
from loss import MultiBoxLoss
from model import SSD

# 1. device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. visdom
# vis = visdom.Visdom()

# 3. data set 정의
transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor()
         ])

root_dir = "D:\Data\\voc\\2012"
trainset = loader.VOC_loader(root_dir, transform=transform)
# 4. data loader 정의
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=2, collate_fn=trainset.collate_fn,
                                          shuffle=True, num_workers=0)

# 5. model 정의
net = SSD().to(device)

# 6. loss 정의
criterion = MultiBoxLoss().to(device)

# 7. optimizer 정의
optimizer = optim.Adam(net.parameters(), lr=0.001)


total_step = len(trainloader)

# 9. train
for epoch in range(10):

    epoch_time = time.time()
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = [l.to(device) for l in labels]
        # labels = labels.to(device)

        optimizer.zero_grad()
        bbox, cls = net(images)
        loss = criterion(bbox[0], cls[0], labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}'
                  .format(epoch + 1, 10, i + 1, total_step, loss.item(), time.time() - epoch_time))



