import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from xml.etree.ElementTree import parse
import torchvision.transforms as transforms

# dataset 정보 담고있는 dictionary
DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


class VOC_loader(data.Dataset):
    def __init__(self,
                 root='../data',
                 year='2012',
                 download=False,
                 image_set='train',
                 transform=None,
                 target_transform=None):
        """
        voc detection data loader
        :param root (string) : voc data 저장하는 폴더
        :param year (string) :
        :param download (bool) :
        :param img_set (string) :
        :param transform (callable) :
        :param target_transform (callable) :
        """

        super(VOC_loader, self).__init__()
        # download
        self.root = root
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.image_set = image_set

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        # 이름 넣어주는 부분 이름 .tar 제거 부분
        voc_root = os.path.join(self.root, self.filename.split('.')[0])

        voc_root = os.path.join(voc_root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')
        print(image_dir)
        print(annotation_dir)

        # transform 추가 부분
        self.transform = transform
        self.target_transform = target_transform

        img_list = os.listdir(image_dir)
        anno_list = os.listdir(annotation_dir)
        self.images = [os.path.join(image_dir, x) for x in img_list]
        self.annotations = [os.path.join(annotation_dir, x) for x in anno_list]

        # 같지 않으면 error 를 내라.
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc(self.annotations[index])

        # Image.open 의 size 는 w, h 순서이다.
        # annotation 맞춰주기 위함
        old_w, old_h = image.size
        print("old_h, old_w:", old_h, old_w)

        # transform 적용
        if self.transform is not None:
            image = self.transform(image)
            scale = (600/old_h, 1000/old_w)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, scale

    def __len__(self):
        return len(self.images)

    def parse_voc(self, xml_file_path):

        VOC_CLASSES = ['background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

        tree = parse(xml_file_path)
        root = tree.getroot()

        ret = []

        for obj in root.iter("object"):

            # 'name' tag 에서 멈추기
            name = obj.find('./name')
            # bbox tag 에서 멈추기
            bbox = obj.find('./bndbox')
            x_min = bbox.find('./xmin')
            y_min = bbox.find('./ymin')
            x_max = bbox.find('./xmax')
            y_max = bbox.find('./ymax')

            # from str to int
            x_min = float(x_min.text)
            y_min = float(y_min.text)
            x_max = float(x_max.text)
            y_max = float(y_max.text)

            ret.append([x_min, y_min, x_max, y_max, VOC_CLASSES.index(name.text)])

        return ret


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((600, 1000)),
            transforms.ToTensor(),
         ])
    # 얘는 img 에 대해서만 하는거임

    trainset = VOC_loader(transform=transform)

    # 이미지 하나씩 가져오는 부분
    for i in range(len(trainset)):
        image, annotation, scale = trainset[i]

        print(scale)

        img = np.array(image)
        img = img.transpose((1, 2, 0))
        img *= 255
        img = img[:, :, ::-1].astype(np.uint8)

        annotation = np.array(annotation)
        annotation[:, 0] *= scale[1]
        annotation[:, 2] *= scale[1]
        annotation[:, 1] *= scale[0]
        annotation[:, 3] *= scale[0]

        for i in range(len(annotation)):
            img = cv2.rectangle(img, (int(annotation[i][0]), int(annotation[i][1])), (int(annotation[i][2]), int(annotation[i][3])), (255, 255, 0), 3)

        cv2.imshow('image', img)
        cv2.waitKey(0)
