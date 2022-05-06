import os
import cv2
import numpy as np
import random

data_path = "D:/Study/Thesis/LiTS/datasets-png"
dst_path = "D:/Study/Thesis/LiTS/balanced-datasets/liver"
sub_folder = os.listdir(data_path)

for folder in sub_folder:
    seg_path = os.path.join(data_path, folder, 'liver_seg/liver_seg')
    ct_path = os.path.join(data_path, folder, 'ct/ct')
    file_names = os.listdir(seg_path)
    liver_slices = []
    non_liver_slices = []
    for file_name in file_names:
        liver_seg = cv2.imread(os.path.join(seg_path, file_name))
        if liver_seg.sum() != 0:
            liver_slices.append(file_name)
        else:
            non_liver_slices.append(file_name)
    random.shuffle(non_liver_slices)
    non_liver_slices = non_liver_slices[:len(liver_slices)]
    balanced_list = liver_slices + non_liver_slices
    print(f'Number of liver slices in {folder}set: {len(liver_slices)}')
    for file_name in balanced_list:
        ct_img = cv2.imread(os.path.join(ct_path, file_name.replace('segmentation', 'volume')), cv2.IMREAD_ANYDEPTH)
        liver_seg_img = cv2.imread(os.path.join(seg_path, file_name))

        cv2.imwrite(os.path.join(dst_path, folder, 'ct/ct', file_name.replace('segmentation', 'volume')), ct_img)
        cv2.imwrite(os.path.join(dst_path, folder, 'liver_seg/liver_seg', file_name), liver_seg_img)
