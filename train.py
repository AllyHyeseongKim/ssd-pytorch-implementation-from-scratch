import torch
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
        transforms.Resize((600, 1000)),
        transforms.ToTensor(),
    ])

train_set = loader.VOC_loader(transform=transform)
# 4. data loader 정의
train_loader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

# 5. model 정의
net = SSD().to(device)

# 6. loss 정의
criterion = MultiBoxLoss().to(device)

# 7. optimizer 정의
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 9. train
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        bbox, cls, f = net(images)
