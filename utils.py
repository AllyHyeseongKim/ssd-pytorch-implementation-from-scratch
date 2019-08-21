import torch
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision.transforms.functional as FT


def photometric_distort():
    transforms_list = []
    # random brightness
    if random.random() > 0.5:
        br = transforms.ColorJitter(brightness=random.uniform(0.5, 1.5))
        transforms_list.append(br)
    if random.random() > 0.5:
        cont = transforms.ColorJitter(contrast=random.uniform(0.5, 1.5))
        transforms_list.append(cont)
    if random.random() > 0.5:
        sat = transforms.ColorJitter(saturation=random.uniform(0.5, 1.5))
        transforms_list.append(sat)
    if random.random() > 0.5:
        hue = transforms.ColorJitter(hue=abs(random.uniform(-18 / 255., 18 / 255.)))
        transforms_list.append(hue)
    # a = abs(random.uniform(-18 / 255., 18 / 255.))
    # print(a)
    transforms_list.append(transforms.ToTensor())

    return transforms_list


def random_expand(image, target, mean):

    if random.random() > 0.5:
        return image, target

    """
    image 와 target 이 들어오면 zoom out 기능으로 이미지를 작게 만드는 것!
    :param image: torch.tensor : input 이미지, size : torch.Size([3, 333, 500])
    :param target: torch.tensor : [[x1, y1, x2, y2, c]...] 로 이루어진 target
    :param mean : np.array : 나머지를 채우는 mean 값
    :return: zoom out 해서 나오는 부분 img, target
    """
    # step 1 ) 랜덤으로 1~4 scale 중에 뽑는다. --> 얼마나 커지는지를 알려주는 것을 랜덤으로 뽑는다.
    scale = random.uniform(1, 4)

    # step 2 ) 새로운 left right top bottom 을 구한다.

    old_h = image.size(1)
    old_w = image.size(2)
    new_h = int(old_h * scale)
    new_w = int(old_w * scale)

    # 아래와 같이 새로운 image 생성
    #  --------------------------
    # |          |               |
    # |          | old_h         |
    # |          |               |
    # |----------                |
    # |   old_ w                 | new_h
    # |                          |
    # |                          |
    # |                          |
    #  --------------------------
    #            new_w

    # 좌 우 상 하 를 새로 정해서 new_w 와 new_h 의 중간에 넣도록 정하는 부분.
    left = random.randint(0, new_w-old_w)
    right = left + old_w
    top = random.randint(0, new_h-old_h)
    bottom = top + old_h

    # 원래의 image 위치 변경
    #  --------------------------
    # |                          |
    # |       ---------          |
    # |      |         |         |
    # |      |         | old_h   |  new_h
    # |      |         |         |
    # |       ---------          |
    # |         old_ w           |
    # |                          |
    #  --------------------------
    #            new_w

    # step 3 ) 나머지 부분을 fill 한다 mean 으로

    # tensor 로 바꾸어 주는 부분
    mean = torch.tensor(mean) # torch.Size([3]) --> [3, 1, 1] 로 변경 해 주어야 함.
    mean = mean.unsqueeze(1).unsqueeze(1)

    new_image = torch.ones([3, new_h, new_w], dtype=torch.float)
    new_image *= mean

    # mean visualization 보는 부분
    # new_image_vis = new_image.permute(1, 2, 0)
    # plt.figure('mean_image')
    # plt.imshow(new_image_vis)
    # plt.show()

    # step 4 ) new 이미지 위에 올린다.
    new_image[:, top:bottom, left:right] = image

    # 올린 이미지 vis
    # new_image_vis2 = new_image.permute(1, 2, 0)
    # plt.figure('mix_image')
    # plt.imshow(new_image_vis2)
    # plt.show()

    # step 5 ) target 변환 부분
    target[:, :4] = target[:, :4] + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    # 더해진 이미지
    # new_image_vis2 = new_image.permute(1, 2, 0)
    # plt.figure('mix_image')
    # plt.imshow(new_image_vis2)
    # # 1개만 실험 여기서  Rectangle 은 (x1, y1, w, h)
    # plt.gca().add_patch(Rectangle((target[0][0], target[0][1]), target[0][2]-target[0][0], target[0][3]-target[0][1],
    #                               linewidth=1, edgecolor='r', facecolor='none'))
    # plt.show()

    return new_image, target


def random_crop(image, target):

    """
    zoom in 느낌으로 더 땡기는 부분?
    :param image:
    :param target:
    :return:
    """
    old_h = image.size(1)
    old_w = image.size(2)

    # step 1 ) minimum overlap 을 구한다.
    while True:
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 1 / 6 확률로 안한다.
        # test 용
        min_overlap = 0.3
        if min_overlap is None:
            return image, target

        # 저자의 구현에서 최대 시도횟수를 50으로 적용하였다.
        max_trials = 50
        for _ in range(max_trials):

            # step 2 ) 랜덤으로 w, h 의 scale 을 구한후 새로운 w, h 를 구한다.
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(old_h * scale_h)
            new_w = int(old_w * scale_w)

            # step 3 ) aspect ration 를 제한한다.
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue  # 다시 for 문으로 진입.

            # step 4 ) crop 한 left right top bottom 생성!

            # 아래와 같이 새로운 image 생성 --> 가운데로~
            #  --------------------------
            # |          |               |
            # |          | new_h         |
            # |          |               |
            # |----------                |
            # |   new_w                  | old_h
            # |                          |
            # |                          |
            # |                          |
            #  --------------------------
            #            old_w
            #
            #            |
            #            |
            #            |
            #            V
            #
            #  --------------------------
            # |                          |
            # |       ---------          |
            # |      |         |         |
            # |      |         | new_h   |  old_h
            # |      |         |         |
            # |       ---------          |
            # |         new_w            |
            # |                          |
            #  --------------------------
            #            old_w

            left = random.randint(0, old_w-new_w)
            right = left + new_w
            top = random.randint(0, old_h-new_h)
            bottom = top + new_h
            # crop 부분 정의
            crop = torch.FloatTensor([left, top, right, bottom])

            # step 5 ) crop 의 iou 가 min_overlap 보다 큰지 확인
            overlap = find_jaccard_overlap(crop.unsqueeze(0), target[:, :4])  # (1, n_objects)
            overlap = overlap.squeeze(0)  # (n_objects)

            # min_overlap 이 더 작으면 다시
            if overlap.max().item() < min_overlap:
                continue

            # step 6) 새로 crop 한 이미지가 중심점이 또한 object 의 중간에 위치 하는지 확인
            bb_centers = (target[:, :2] + target[:, 2:4]) / 2.  # (n_objects, 2) --> find (c_x, c_y)

            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * \
                              (bb_centers[:, 1] > top) * (bb_centers[:, 1] < bottom)
            # (n_objects) 모두 0 혹은 1 : torch.uint8 의 값과  >, < 비교를 하면 boolean 연산이 된다.

            # 하나도 안 포함되어 있다면 즉 모두 0이라면
            if not centers_in_crop.any():
                continue  # for 문의 처음으로 다시

            # a = torch.tensor([1, 0, 1, 1], dtype=torch.uint8), 이고
            # target 이 [[1, 2, 3, 4,], [5, 6, 7, 8], [1, 2, 3, 4,], [5, 6, 7, 8]] 이면
            # target[a, :] 에서 a 의 0 부분은 사라져서
            # 결국 [3, 4] 의 tensor 인 [[1, 2, 3, 4,],[1, 2, 3, 4,], [5, 6, 7, 8]] 가 남는다.

            # object 의 중간에 위치하면 남겨둔다~
            target = target[centers_in_crop, :]

            # step 7) Crop image
            image = image[:, top:bottom, left:right]  # (3, new_h, new_w)
            # step 8) Crop target
            # 만약 bbox 보다 작은 crop 을 잡았다면, 아래의 처리를 해 주어야 한다.
            target[:, :2] = torch.max(target[:, :2], crop[:2])  # crop[:2] is [left, top]
            target[:, :2] -= crop[:2]
            target[:, 2:4] = torch.min(target[:, 2:4], crop[2:])  # crop[2:] is [right, bottom]
            target[:, 2:4] -= crop[:2]

            # image vis
            # image_vis3 = image.permute(1, 2, 0)
            # plt.figure('rand_crop')
            # plt.imshow(image_vis3)
            # plt.gca().add_patch(
            #     Rectangle((target[0][0], target[0][1]), target[0][2] - target[0][0], target[0][3] - target[0][1],
            #               linewidth=1, edgecolor='r', facecolor='none'))
            # plt.show()

            return image, target


def random_mirror(image, target):
    """
    이미지를 랜덤으로 flip 시키거나 안 시키는 함수
    :param image:
    :param target:
    :return:
    """
    # if random.random() > 0.5:
    #     return image, target

    # Convert Torch tensor to PIL image
    image = FT.to_pil_image(image)

    # step 1 ) image flip
    image = FT.hflip(image)
    image = FT.to_tensor(image)

    # step 2 ) target flip
    target = target
    target[:, 0] = image.size(2) - target[:, 0] - 1
    target[:, 2] = image.size(2) - target[:, 2] - 1

    target[:, :4] = target[:, [2, 1, 0, 3]]

    # image_vis4 = image.permute(1, 2, 0)
    # plt.figure('flip')
    # plt.imshow(image_vis4)
    # plt.gca().add_patch(
    #     Rectangle((target[0][0], target[0][1]), target[0][2] - target[0][0], target[0][3] - target[0][1],
    #               linewidth=1, edgecolor='r', facecolor='none'))
    # plt.show()

    return image, target


def resize(image, target, size=(300, 300)):

    old_h = image.size(1)
    old_w = image.size(2)
    old_scale = torch.FloatTensor([old_w, old_h, old_w, old_h]).unsqueeze(0)

    # step 1 ) image resize
    image = FT.to_pil_image(image)
    image = FT.resize(image, size)
    image = FT.to_tensor(image)

    # step 2 ) target size
    target[:, :4] = target[:, :4] / old_scale
    size = torch.FloatTensor([size[1], size[0], size[1], size[0]]).unsqueeze(0)
    target[:, :4] = target[:, :4] * size

    # image_vis4 = image.permute(1, 2, 0)
    # plt.figure('resize')
    # plt.imshow(image_vis4)
    # # 1개만 실험 여기서  Rectangle 은 (x1, y1, w, h)
    # plt.gca().add_patch(
    #     Rectangle((target[0][0], target[0][1]), target[0][2] - target[0][0], target[0][3] - target[0][1],
    #               linewidth=1, edgecolor='r', facecolor='none'))
    # plt.show()

    return image, target


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


if __name__ == "__main__":
    photometric_distort()