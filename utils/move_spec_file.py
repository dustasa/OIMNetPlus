# -- coding: utf-8 --
# txt readline, move spec imgs to another file
import os.path as osp
import shutil


def _load_queries(img_dir, ann_dir, img_cp_dir, ann_cp_dir):
    query_info = osp.join(img_cp_dir, "prw_seg.txt")
    with open(query_info, "rb") as f:
        raw = f.readlines()
        # print(f"raw is: {raw}")

    queries = []
    for line in raw:
        linelist = str(line, "utf-8").split("\n")
        img_name = linelist[0] + ".jpg"
        # print(f'img name: {img_name}')
        src_img_name = img_dir + img_name
        # print(f"src_img_name is： {src_img_name}")
        shutil.copy(src_img_name, ann_cp_dir)

    return queries


if __name__ == '__main__':
    # train 从train中移动
    img = '/home/aousn/dataset/PRW_mini/frames/'
    ann = 'home/aousn/dataset/PRW_mini/frames/annotations/'
    img_cp = '/home/aousn/dataset/segment dataset/PRW_seg/'
    ann_cp = '/home/aousn/dataset/segment dataset/PRW_seg/frame_train/'
    _load_queries(img_dir=img, ann_dir=ann, img_cp_dir=img_cp, ann_cp_dir=ann_cp)
