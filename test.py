import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from dataset.voc_loader import VOC_loader
from model import SSD
import numpy as np


# 1. device config
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 2. test data set
transform = transforms.Compose([transforms.Resize(size=(300, 300)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_set = VOC_loader(root="D:\Data\\voc\\2007", year='2007_test', image_set='train', phase='TEST', transform=transform)

# 3.test loader
test_loader = data.DataLoader(dataset=test_set,
                              batch_size=1,
                              collate_fn=test_set.collate_fn)

# 4. model load
net = SSD().to(device)
net.load_state_dict(torch.load('./saves/ssd.4.ckpt'))
net.eval()


# 5. test
with torch.no_grad():

    for image, target in test_loader:
        # image