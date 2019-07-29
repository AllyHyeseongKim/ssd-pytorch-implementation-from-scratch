import torch
import numpy as np
import utils.anchors as anchors


# torch tensors version
def bbox_overlaps(boxes, gt_boxes):
    """
    두 bbox 의 overlap 즉 IOU 를 뽑아내는 함수
    :param bbox1 (torch.tensor) : bbox1
    :param bbox2 (torch.tensor): bbox2
    :return: 두개의 bbox 의 interaction of union
    """

    # 만약 numpy 로 들어왔다면 텐서로 변환해주는 부분
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes).cuda()
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = torch.from_numpy(gt_boxes).cuda()

        # boxes : some props boxes
        # gt_boxes : ground truth box

    oo = []

    for b in gt_boxes:
        # 각각 0,1,2,3 col을 복사.
        x1 = boxes.select(1, 0).clone()
        y1 = boxes.select(1, 1).clone()
        x2 = boxes.select(1, 2).clone()
        y2 = boxes.select(1, 3).clone()

        # [1764] 의 size() 를 갖는 torch.tensor([0, 1, 1, 0  .... , 1, 0])
        a1 = torch.lt(x1, b[0])
        b1 = torch.lt(y1, b[1])
        a2 = torch.gt(x2, b[2])
        b2 = torch.gt(x2, b[3])

        # masking
        x1[a1] = b[0]
        y1[b1] = b[1]
        x2[a2] = b[2]
        y2[b2] = b[3]

        # w 구하기
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # intersection area
        inter = torch.mul(w, h).float()

        # total area
        aarea = torch.mul((boxes.select(1, 2) - boxes.select(1, 0) + 1),
                          (boxes.select(1, 3) - boxes.select(1, 1) + 1)).float()
        barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

        # intersection over union overlap
        o = torch.div(inter, (aarea + barea - inter))

        # set invalid entries to 0 overlap
        o[w.lt(0)] = 0
        o[h.lt(0)] = 0

        oo += [o]

    return torch.cat([o.view(-1, 1) for o in oo], 1)


if __name__ == "__main__":
    anchor = anchors.anchor
    feature = torch.zeros([10, 3, 14, 14])
    anc = anchors.get_anchors(feature=feature, anchor=anchor)

    gt = [[104.0, 78.0, 375.0, 183.0, 1], [133.0, 88.0, 197.0, 123.0, 1], [195.0, 180.0, 213.0, 229.0, 15], [26.0, 189.0, 44.0, 238.0, 15]]
    overlaps = bbox_overlaps(anc, gt)
    print(overlaps)

    test, _ = overlaps.max(dim=1)

    # label 만들때 쓰는구나
    argmax_overlaps = overlaps.argmax(dim=1)
    max_overlaps = overlaps[np.arange(len(anc)), argmax_overlaps]  # anchor와 가장 많이 겹치는 IOU

    # if torch.all(torch.eq(max_overlaps, test)):
    #     print("same")

    gt_argmax_overlaps = overlaps.argmax(dim=0)  # gtbox와 가장많이 겹치는 anchor

    # gtbox와 가장많이 겹치는 anchor의 index(gt_argmax_overlaps)와 그 IOU(gt_max_overlaps)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    print(argmax_overlaps)










