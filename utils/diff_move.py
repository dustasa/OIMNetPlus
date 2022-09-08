# -- coding: utf-8 --
import os
import shutil


# 将in_path文件夹里的所有文件复制到out_path中
def copy_file(in_path, out_path):
    if not os.path.isdir(in_path):
        shutil.copy(in_path, out_path)
    else:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        names = os.listdir(in_path)
        for name in names:
            in_path_2 = in_path + '\\' + name
            out_path_2 = out_path + '\\' + name
            copy_file(in_path_2, out_path_2)


# 对比两个文件夹下的文件，如果某文件仅存在于一个文件夹，则直接将此文件复制到same_path
# 如果某文件在两个文件夹中都存在，则把该文件复制到difference_path
def compare_copy(dir_path_1, dir_path_2, difference_path, same_path):
    names_1 = os.listdir(dir_path_1)
    names_2 = os.listdir(dir_path_2)
    for name in names_1:
        path_1 = dir_path_1 + name
        path_2 = dir_path_2 + name
        Difference = difference_path + name
        Same = same_path + name
        if not os.path.exists(path_2):
            # 复制文件夹或文件
            copy_file(path_1, Difference)
            print(f'copy {path_1} to {difference_path}')
        else:
            if not os.path.isdir(path_1):
                shutil.copy(path_1, Same)
                print(f'copy {path_1} to {same_path}')

    for name in names_2:
        path_1 = dir_path_1 + name
        path_2 = dir_path_2 + name
        Difference = difference_path + name
        if not os.path.exists(path_1):
            # 复制文件夹或文件
            copy_file(path_2, Difference)
            print(f'copy {path_2} to {difference_path}')
    print('done')


if __name__ == '__main__':
    dir_path_1 = '/home/aousn/dataset/PRW_mini/frames/'
    dir_path_2 = '/home/aousn/dataset/PRW_mini/frames_test/'
    difference_path = '/home/aousn/dataset/PRW_mini/frames_train/'
    same_path = '/home/aousn/dataset/PRW_mini/4/'
    compare_copy(dir_path_1, dir_path_2, difference_path, same_path)
