import torch
from torch import Tensor
from losses.iou import box_iou, generalized_box_iou, box_area, s_box_iou, ma_box_iou_t2b, ma_box_iou_b2t, ma_box_iou_ex_points


if __name__ == "__main__":
    boxes1 = torch.Tensor([[9, 3, 13, 11], [11, 3, 15, 11]])  # GT
    # boxes1 = torch.Tensor([[230, 280, 440, 900]])  # GT
    # boxes1 = torch.Tensor([[9, 3, 13, 11]])  # GT
    boxes2 = torch.Tensor([[8, 2, 12, 10], [10, 4, 14, 12], [9.5, 2, 14, 10]])  # proposals
    # boxes2 = torch.Tensor([[188, 156, 398, 776]])  # proposals
    # boxes2 = torch.Tensor([[272, 404, 482, 1024]])  # proposals

    base_ious = box_iou(boxes1, boxes2)
    print(f"iou is：{base_ious}")

    gious = generalized_box_iou(boxes1, boxes2)
    print(f'giou is: {gious}')

    # sious = s_box_iou(boxes1, boxes2)
    # print(f'siou is: {sious}')

    # maiou1 = ma_box_iou_t2b(boxes1, boxes2)
    # print(f'maiou top to bottom is: {maiou1}')
    # maiou2 = ma_box_iou_b2t(boxes1, boxes2)
    # print(f'maiou bottom to top is: {maiou2}')

    maiou3 = ma_box_iou_ex_points(boxes1, boxes2)
    print(f'maiou ma_box_iou_ex_points is: {maiou3}')

