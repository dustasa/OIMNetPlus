# -- coding: utf-8 --
import numpy as np
# import pandas as pd
import os.path as osp
import shutil
import labelme
from labelme.cli import json_to_dataset


def main(file_dir, json_filename):
    file_json = osp.join(file_dir, json_filename)
    # print(f'file_json: {file_json}')
    with open(file_json, 'rb') as f:
        g = f.read()
        linelist = str(g, "utf-8").split("\n")
        raw = f.readlines()
        # g_ = (file_dir + g)
        # print(f"g is: {g}")
        # print(f"linelist is: {linelist}")
    for line in linelist:
        # linelist = str(line, "utf-8").split("\n")
        # json_name = linelist[line]
        # print(f'json_name: {line}')
        g_ = osp.join(file_dir, line)
        json_to_dataset(g_)
        print(f'g_: {g_}')


if __name__ == '__main__':
    # train 从train中移动
    file_dir = '/home/aousn/git-repo/OIMNetPlus/datasets/PRW_seg/train/frame_train_mix/'
    json_filename = 'json.txt'
    main(file_dir, json_filename)
