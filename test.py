import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from dataset.voc_loader import VOC_loader
from model import SSD
import numpy as np
from loss import create_prior_boxes
import torch.nn.functional as F
from utils import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



def detect_objects(predicted_locs, predicted_scores, min_score=0.01, max_overlap=0.45, top_k=200):
    """
    Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

    For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

    :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
    :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
    :param min_score: minimum threshold for a box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :return: detections (boxes, labels, and scores), lists of length batch_size
    """
    batch_size = predicted_locs.size(0)
    priors_cxcy = create_prior_boxes()
    n_priors = priors_cxcy.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted boxes to
        decoded_locs = cxcy_to_xy(
            gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

        # Lists to store boxes and scores for this image
        image_boxes = list()
        image_labels = list()
        image_scores = list()

        max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

        # Check for each class
        for c in range(1, 20):
            # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = predicted_scores[i][:, c]  # (8732)
            score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
            class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

            # Sort predicted boxes and scores by scores
            class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
            class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

            # Find the overlap between predicted boxes
            overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

            # Non-Maximum Suppression (NMS)

            # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
            # 1 implies suppress, 0 implies don't suppress
            suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box is already marked for suppression
                if suppress[box] == 1:
                    continue

                # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                # Find such boxes and update suppress indices
                suppress = torch.max(suppress, overlap[box] > max_overlap)
                # The max operation retains previously suppressed boxes, like an 'OR' operation

                # Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = 0

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[1 - suppress])
            image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
            image_scores.append(class_scores[1 - suppress])

        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            image_labels.append(torch.LongTensor([0]).to(device))
            image_scores.append(torch.FloatTensor([0.]).to(device))

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        # Keep only the top k objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]  # (top_k)
            image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
            image_labels = image_labels[sort_ind][:top_k]  # (top_k)

        # Append to lists that store predicted boxes and scores for all images
        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


def detection(loc, cls, min_score=0.01, max_overlap=0.45, top_k=200):
    """
    model 의 출력을 가지고 실제 bbox 를 찾는 부분
    :param loc:             torch.tensor [B, 8732, 4] - location 출력물
    :param cls:             torch.tensor [B, 8732, 21] -  classification 출력물
    :param prior_cxcy:      torch.tensor [8732, 4] - default boxes
    :param min_score:       a float - class 를 판별하는 최소의 score 기준
    :param max_overlap:
    :param top_k:
    :return:
    """
    # step 1. loc, cls 바꿔줌
    batch_size = loc.size(0)
    prior_cxcy = create_prior_boxes()
    n_priors = prior_cxcy.size(0)
    pre_cls = F.softmax(cls, dim=2)  # [B, 8732, 21] 마지막 dim 에서 0 ~ 1 의 확률로 변화

    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    # 오류검사 8732 개 잘 있나.
    assert n_priors == loc.size(1) == pre_cls.size(1)

    for i in range(batch_size):

        # loc 는 bbox regression 을 통해서 나온 애들이 나오기 때문에, 그것을 prior 를 연산을 해주어서 (x1, y1, x2, y2) 로
        # 바꿔줌!
        loc = cxcy_to_xy(gcxgcy_to_cxcy(loc[i], prior_cxcy))

        # step 2. 각 class 는 최소 score 보다 높아야 인정 해 준다.
        n_class = 20  # voc 에 한하여

        image_boxes = list()
        image_labels = list()
        image_scores = list()

        for c in range(1, n_class):
            class_scores = cls[i][:, c]  # torch.Size([1, 8732])
            class_idx = class_scores > min_score
            sum_idx = class_idx.sum().item()

            # 진짜 1도 아니라고 하면 걍 넘어가기~
            if sum_idx == 0:
                continue

            # 있는 부분만 남기고 나머지 지우기 -- index 로 계산한당.
            class_scores = class_scores[class_idx]  # 처음 해봤을 때 3100 개로 줄음
            loc = loc[class_idx]  # loc 도 있는 부분만 봐야지 ~

            # step 3. nmx
            sorted_class, sort_idx = class_scores.sort(dim=0, descending=True)
            sorted_loc = loc[sort_idx]

            # 자기 자신과의 overlap 실시
            overlap = find_jaccard_overlap(sorted_loc, sorted_loc)  # torch.Size([3193, 4]) torch.Size([3193, 4])
            # overlap 은 (3193, 3193)

            suppress = torch.zeros((sum_idx), dtype=torch.uint8).to(device)  # (n_qualified)

            for box in range(sorted_loc.size(0)):
                # If this box is already marked for suppression
                if suppress[box] == 1:
                    continue

                # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                # Find such boxes and update suppress indices
                suppress = torch.max(suppress, overlap[box] > max_overlap)
                # The max operation retains previously suppressed boxes, like an 'OR' operation

                # Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = 0

            # Store only unsuppressed boxes for this class
            image_boxes.append(sorted_loc[1 - suppress])
            image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
            image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            image_labels.append(torch.LongTensor([0]).to(device))
            image_scores.append(torch.FloatTensor([0.]).to(device))

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        # Keep only the top k objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]  # (top_k)
            image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
            image_labels = image_labels[sort_ind][:top_k]  # (top_k)

        # Append to lists that store predicted boxes and scores for all images
        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores


# 1. device config
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 2. test data set -- test 시는 resize 와 normalize 만 해 준다.
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
net.load_state_dict(torch.load('./saves/ssd.1.pth.tar'))
net.eval()


# 5. test
with torch.no_grad():

    for image, target in test_loader:

        # 사실 target 은 필요 없둠
        # print(image.size())  # image : torch.Size([1, 3, 300, 300])
        # print(len(target))  # target : list ( torch.Size([object_n, 5]) ) len(list) = 1
        # image

        # 각각의 tensor 들을 gpu 에 올리는 부분
        image = image.to(device)

        # target = [t.to(device) for t in target]

        loc, cls = net(image)

        # tuple to tensor
        loc = loc[0]  # torch.Size([1, 8732, 4])
        cls = cls[0]  # torch.Size([1, 8732, 21])

        pred_boxes, pred_labels, pred_scores = detect_objects(loc, cls)

        # pred_boxes, pred_labels, pred_scores = detection(loc, cls)
        print(pred_boxes)

        bbox = pred_boxes[0][0]

        image_vis = image.squeeze(0).permute(1, 2, 0)
        plt.figure('result')
        plt.imshow(image_vis.cpu())
        plt.gca().add_patch(Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                      linewidth=1, edgecolor='r', facecolor='none'))
        plt.show()











