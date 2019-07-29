import numpy as np
import torch
import faster_rcnn.utils.anchors as anchors


def rpn_targets(all_anchors_boxes, gt_boxes_c, im_info, args):

    # 전체 anchor box 에서 gt 와 비슷한 애들 뽑아오는 부분!
    """
    Arguments:
        all_anchors_boxes (Tensor) : (H/16 * W/16 * 9, 4)
        gt_boxes_c (Ndarray) : (# gt boxes, 5) [x, y, x`, y`, class]
        im_info (Tuple) : (Height, Width, Channel, Scale)
        args (argparse.Namespace) : global arguments

    Return:
        labels (Ndarray) : (H/16 * W/16 * 9,)
        bbox_targets (Ndarray) : (H/16 * W/16 * 9, 4)
        bbox_inside_weights (Ndarray) : (H/16 * W/16 * 9, 4)
    """

    # it maybe H/16 * W/16 * 9
    num_anchors = all_anchors_boxes.shape[0]  # ex) 갯수는 1764 개

    # im_info : (H, W, C, S)
    # height, width = im_info[:2]  # ex) image 는 224, 224,
    height, width = 600, 1000

    # only keep anchors inside the image
    _allowed_border = 0

    # np.where 은  index 찾는것이다. 내부의 조건을 만족하는 !
    inds_inside = np.where(
        (all_anchors_boxes[:, 0] >= -_allowed_border) &
        (all_anchors_boxes[:, 1] >= -_allowed_border) &
        (all_anchors_boxes[:, 2] < width + _allowed_border) &  # width
        (all_anchors_boxes[:, 3] < height + _allowed_border)  # height
    )[0]
    # np.where 이 tuple 형태로 나타내어져서 [0] 부분에는 index의 array 가 들어가고
    # [1] 부분은 잘 모르겠당.ㅎ
    #  찾는것이다. 내부의 조건을 만족하는 !
    print(inds_inside)

    # keep only inside anchors
    inside_anchors_boxes = all_anchors_boxes[inds_inside, :]
    assert inside_anchors_boxes.shape[0] > 0, '{0}x{1} -> {2}'.format(height, width, num_anchors)

    # label: 1 is positive, 0 is negative, -1 is don't care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if isinstance(inside_anchors_boxes, np.ndarray):
        if torch.cuda.is_available():
            inside_anchors_boxes = torch.from_numpy(inside_anchors_boxes).cuda()

    x1 = inside_anchors_boxes.select(1, 0).clone()
    y1 = inside_anchors_boxes.select(1, 1).clone()
    x2 = inside_anchors_boxes.select(1, 2).clone()
    y2 = inside_anchors_boxes.select(1, 3).clone()

    print(x1)
    # overlaps = bbox_overlaps(inside_anchors_boxes, gt_boxes_c[:, :-1]).cpu().numpy()


if __name__ == "__main__":
    anchor = anchors.anchor
    feature = torch.zeros((10, 512, 60, 40))
    # feature 에서 필요한것은 H, W 뿐
    all_anchors = anchors.get_anchors(feature=feature, anchor=anchor)
    # print(all_anchors)

    rpn_targets(all_anchors, (600, 1000, 3, 224), 0, 0)