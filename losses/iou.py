import torch
from torch import Tensor
import os
# from utils.utils import write_text
from defaults import get_default_cfg


def print_debug(*args, **kwargs):
    flag = True
    # flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def octagon_area(boxes: Tensor) -> Tensor:
    pass


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    cfg = get_default_cfg()
    output_dir = cfg.OUTPUT_DIR
    # print("using iou here in iou.py")
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)
    # print(f"iou is: {iou}")
    # write_text(sentence="Computing iou {}".format(iou), fpath=os.path.join(output_dir, 'iou.txt'))
    return iou


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    # print("using giou here in iou.py")
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    areai = whi[:, :, 0] * whi[:, :, 1]
    return iou - (areai - union) / areai


# mask-aware iou, top to bottom
def ma_box_iou_t2b(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    # box1 GT, box2 proposals
    mask_ratio = 0.6
    mask_boxes1 = boxes1.clone()
    mask_boxes1[:, 3] = mask_boxes1[:, 1] + (mask_boxes1[:, 3] - mask_boxes1[:, 1]) * mask_ratio
    print_debug(f'boxes1: {boxes1}')
    print_debug(f'mask boxes1: {mask_boxes1}')

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    print_debug(f'lt {lt}')
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    print_debug(f'rb {rb}')
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    print_debug(f'wh {wh}')
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    print_debug(f'inter {inter}')
    union = area1[:, None] + area2 - inter
    # iou = inter / union

    lt_mask = torch.max(mask_boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    print_debug(f'lt_mask {lt_mask}')
    rb_mask = torch.min(mask_boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    print_debug(f'rb_mask {rb_mask}')
    wh_mask = (rb_mask - lt_mask).clamp(min=0)  # [N,M,2]
    print_debug(f'wh_mask {wh_mask}')
    inter_mask = wh_mask[:, :, 0] * wh_mask[:, :, 1]  # [N,M]
    print_debug(f'inter_mask {inter_mask}')
    ma_iou = (1 / mask_ratio) * inter_mask / union
    return ma_iou


# mask-aware iou,  bottom to top
def ma_box_iou_b2t(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    # box1 GT, box2 proposals
    mask_ratio = 0.6
    mask_boxes1 = boxes1.clone()
    mask_boxes1[:, 1] = mask_boxes1[:, 3] - (mask_boxes1[:, 3] - mask_boxes1[:, 1]) * mask_ratio
    print_debug(f'boxes1: {boxes1}')
    print_debug(f'mask boxes1: {mask_boxes1}')

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    print_debug(f'lt {lt}')
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    print_debug(f'rb {rb}')
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    print_debug(f'wh {wh}')
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    print_debug(f'inter {inter}')
    union = area1[:, None] + area2 - inter
    # iou = inter / union

    lt_mask = torch.max(mask_boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    print_debug(f'lt_mask {lt_mask}')
    rb_mask = torch.min(mask_boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    print_debug(f'rb_mask {rb_mask}')
    wh_mask = (rb_mask - lt_mask).clamp(min=0)  # [N,M,2]
    print_debug(f'wh_mask {wh_mask}')
    inter_mask = wh_mask[:, :, 0] * wh_mask[:, :, 1]  # [N,M]
    print_debug(f'inter_mask {inter_mask}')
    ma_iou = (1 / mask_ratio) * inter_mask / union
    return ma_iou


# mask-aware iou with for extreme points, ------Bottom-up object detection by grouping extreme and center points
def ma_box_iou_ex_points(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    # box1 GT, box2 proposals
    # box1[:,0] = {boxes1[:, 0]}'  # box x1
    # box1[:,1] = {boxes1[:, 1]}'  # box y1
    # box1[:,2] = {boxes1[:, 2]}'  # box x2
    # box1[:,3] = {boxes1[:, 3]}'  # box y2

    # gt_tl: (x1, y1), gt_br: (x2, y2)
    # gt_tl: (x_l, y_t)  gt_br: (x_r, y_b)
    # (x_t, y_t), (x_l ,y_l), (x_b, y_b), (x_r, y_r)

    mask_boxes1 = boxes1.clone()

    # height, width of GT boxes
    gt_h = boxes1[:, 3] - boxes1[:, 1]
    gt_w = boxes1[:, 2] - boxes1[:, 0]
    vertical_ratio = 0.17
    horizontal_ratio = 0.17

    x_l = mask_boxes1[:, 0]
    y_l = (mask_boxes1[:, 1] + mask_boxes1[:, 3]) / 3
    x_t = (mask_boxes1[:, 2] + mask_boxes1[:, 0]) / 2
    y_t = mask_boxes1[:, 1]
    x_r = mask_boxes1[:, 2]
    y_r = (mask_boxes1[:, 1] + mask_boxes1[:, 3]) / 3
    x_b = (mask_boxes1[:, 2] + mask_boxes1[:, 0]) / 2
    y_b = mask_boxes1[:, 3]

    print_debug(f'boxes1: {boxes1}')

    area1 = box_area(boxes1)
    print_debug(f'gt area: {area1}')
    area2 = box_area(boxes2)
    # mask_tl_area = gt_w * (1 - horizontal_ratio * 2) * gt_h * (1 - vertical_ratio * 2) / 2
    # mask_br_area = gt_w * (1 - horizontal_ratio * 2) * gt_h * (1 - vertical_ratio * 2) / 2
    mask_area = area1 - gt_w * (1 - horizontal_ratio * 2) * gt_h * (1 - vertical_ratio * 2) / 2
    print_debug(f'mask_area: {mask_area}')

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    print_debug(f'lt {lt}')
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    print_debug(f'rb {rb}')
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    print_debug(f'wh {wh}')
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    print_debug(f'inter {inter}')
    union = area1[:, None] + area2 - inter

    # mask_ratio = torch.zeros_like(boxes1)
    mask_ratio = mask_area / area1
    print_debug(f'mask_ratio {mask_ratio}')

    lt_mask = torch.max(mask_boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    print_debug(f'lt_mask {lt_mask}')
    rb_mask = torch.min(mask_boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    print_debug(f'rb_mask {rb_mask}')
    wh_mask = (rb_mask - lt_mask).clamp(min=0)  # [N,M,2]
    print_debug(f'wh_mask {wh_mask}')
    inter_mask = wh_mask[:, :, 0] * wh_mask[:, :, 1]  # [N,M]
    print_debug(f'inter_mask {inter_mask}')
    ones = torch.ones_like(mask_ratio)
    ma_iou = (1 / mask_ratio[0]) * inter_mask / union

    return ma_iou


def s_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    # box1 GT, box2 proposals
    print_debug("using siou here in iou.py")

    print_debug(f'GT {boxes1}')
    print_debug(f'proposal {boxes2}')

    # compute mean of GT_y1
    mean_y1 = torch.mean(boxes1[:, 1])
    print_debug(f'mean of GT y1: {mean_y1}')

    # mask condition
    print_debug('Computing mask condition--------------------------------------')
    mask = boxes2[:, 1].gt(mean_y1)
    print_debug(f'mask: {mask}')
    unmask = ~mask  # tensor bool invert
    print_debug(f'unmask: {unmask}')

    # apply mask on proposal box
    print_debug('Apply mask on proposal box--------------------------------------')
    boxes2_unmask = boxes2[unmask]
    boxes2_mask = boxes2[mask]
    print_debug(f'unmask proposal {boxes2_unmask}')
    print_debug(f'mask proposal {boxes2_mask}')

    area1 = box_area(boxes1)
    print_debug(f'area1 GT {area1}')
    area2 = box_area(boxes2)
    print_debug(f'area2 proposal {area2}')
    print_debug(f'area2 proposal shape {area2.size()}')

    print_debug('Computing mask area on proposal box--------------------------------------')
    area2_unmask = box_area(boxes2_unmask)
    area2_mask = box_area(boxes2_mask)
    print_debug(f'area2_unmask proposal {area2_unmask}')
    print_debug(f'area2_unmask proposal shape {area2_unmask.size()}')
    print_debug(f'area2_mask {area2_mask}')
    print_debug(f'area2_mask proposal shape {area2_mask.size()}')
    area2_cat = torch.cat((area2_unmask, area2_mask), -1)
    print_debug(f'area2_cat {area2_cat}')
    print_debug(f'area2_cat proposal shape {area2_cat.size()}')

    # original lt
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    print_debug(f'lt {lt}')
    print_debug(f'lt shape {lt.size()}')

    #  lt with mask
    print_debug('Computing mask lt on proposal box--------------------------------------')
    lt_unmask = torch.max(boxes1[:, None, :2], boxes2_unmask[:, :2])  # [N,M,2]
    print_debug(f'lt_unmask {lt_unmask}')
    print_debug(f'lt_unmask shape {lt_unmask.size()}')
    lt_mask = torch.max(boxes1[:, None, :2], boxes2_mask[:, :2])  # [N,M,2]
    print_debug(f'lt_mask {lt_mask}')
    print_debug(f'lt_mask shape {lt_mask.size()}')
    #  concat mask and unmask
    lt_cat = torch.cat((lt_unmask, lt_mask), 1)
    print_debug(f'lt_cat {lt_cat}')
    print_debug(f'lt_cat shape {lt_cat.size()}')

    # original rb
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    print_debug(f'rb {rb}')
    print_debug(f'rb shape {rb.size()}')

    #  rb with mask
    print_debug('Computing mask rb on proposal box--------------------------------------')
    rb_unmask = torch.min(boxes1[:, None, 2:], boxes2_unmask[:, 2:])  # [N,M,2]
    print_debug(f'rb_unmask {rb_unmask}')
    rb_mask = torch.min(boxes1[:, None, 2:], boxes2_mask[:, 2:])  # [N,M,2]
    print_debug(f'rb_mask {rb_mask}')
    rb_cat = torch.cat((rb_unmask, rb_mask), 1)
    print_debug(f'rb_cat {rb_cat}')
    print_debug(f'rb_cat shape {rb_cat.size()}')

    # original wh
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    print_debug(f'wh {wh}')
    print_debug(f'wh size {wh.size()}')
    wh2 = (rb_cat - lt_cat).clamp(min=0)
    print_debug(f'wh2 {wh2}')
    print_debug(f'wh2 size {wh2.size()}')

    # wh with mask
    print_debug('Computing mask wh on proposal box--------------------------------------')
    wh_unmask = (rb_unmask - lt_unmask).clamp(min=0)  # [N,M,2]
    print_debug(f'wh_unmask {wh_unmask}')
    wh_mask = (rb_mask - lt_mask).clamp(min=0)  # [N,M,2]
    print_debug(f'wh_mask {wh_mask}')
    wh_cat = torch.cat((wh_unmask, wh_mask), 1)
    print_debug(f'wh_cat {wh_cat}')
    print_debug(f'wh_cat shape {wh_cat.size()}')

    # original inter
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    print_debug(f'inter {inter}')
    print_debug(f'inter size {inter.size()}')
    inter2 = wh2[:, :, 0] * wh2[:, :, 1]  # [N,M]
    print_debug(f'inter2 {inter2}')
    print_debug(f'inter2 size {inter2.size()}')

    # inter with mask
    print_debug('Computing mask inter on proposal box--------------------------------------')
    inter_unmask = wh_unmask[:, :, 0] * wh_unmask[:, :, 1]  # [N,M]
    print_debug(f'inter_unmask {inter_unmask}')
    inter_mask = wh_mask[:, :, 0] * wh_mask[:, :, 1]  # [N,M]
    print_debug(f'inter_mask {inter_mask}')
    inter_cat = torch.cat((inter_unmask, inter_mask), 1)
    print_debug(f'inter_cat {inter_cat}')
    print_debug(f'inter_cat shape {inter_cat.size()}')

    # original iou
    iou = inter / (area1[:, None] + area2 - inter)
    print_debug(f'iou: {iou}')
    print_debug(f'iou size: {iou.size()}')
    iou2 = inter / (area1[:, None] + area2_cat - inter2)
    print_debug(f'iou2: {iou2}')
    print_debug(f'iou2 size: {iou2.size()}')

    # iou with mask
    print_debug('Computing mask iou on proposal box--------------------------------------')
    iou_unmask = inter_unmask / (area1[:, None] + area2_unmask - inter_unmask)
    print_debug(f'iou_unmask {iou_unmask}')
    print_debug(f'iou_unmask shape {iou_unmask.size()}')
    iou_mask = inter_mask / (area1[:, None] + area2_mask - inter_mask) - 0.02
    print_debug(f'iou_mask {iou_mask}')
    print_debug(f'iou_mask shape {iou_mask.size()}')
    iou_cat = torch.cat((iou_unmask, iou_mask), 1)
    print_debug(f'iou_cat {iou_cat}')
    print_debug(f'iou_cat shape {iou_cat.size()}')

    # print(f'box1[:,0] = {boxes1[:, 0]}')  # box x1
    # print(f'box1[:,1] = {boxes1[:, 1]}')  # box y1
    # print(f'box1[:,2] = {boxes1[:, 2]}')  # box x2
    # print(f'box1[:,3] = {boxes1[:, 3]}')  # box y2
    # zero = torch.zeros_like(boxes2[:, 1])
    # print(f'zero: {zero}')
    # zero[:] = mean_y1
    # print(f'zero: {zero}')
    # print(boxes2[:, 1] < zero)
    # print(boxes2.unsqueeze(-1))
    # ones = torch.ones_like(boxes2[mask])
    # print_debug(f'ones {ones}')
    # boxes2[mask][:, 1] = boxes2[mask][:, 1] - 1
    # print(boxes2[mask])
    # boxes2[mask] = torch.sub(boxes2[mask], ones)
    # boxes2[mask][:, 3] = boxes2[mask][:, 3].sub_(ones[:, 3])
    # torch.sub(boxes2[mask], ones)
    # print(torch.sub(boxes2[mask][:, 1], ones))
    # print_debug(f'after change {boxes2}')

    # for i in range(0, boxes2.unsqueeze(0)):

    # if boxes2[:, 1] > zero:  # make sure proposal top above GT
    #     if float(abs(boxes2[:, 0] - boxes1[:, 0]) / (boxes1[:, 2] - boxes1[:, 0])) <= 0.1:
    #         if float((boxes2[:, 1] - boxes1[:, 1]) / (boxes1[:, 3] - boxes1[:, 1])) <= 0.1:
    #             iou = iou + 0.05

    return iou_cat
