import os
import numpy as np
import cv2

from utils import utils
from tensorflow.keras.utils import Progbar

data_path = 'D:/Study/Thesis/LiTS/datasets-png/test/ct/ct/'
masks_path = 'D:/Study/Thesis/LiTS/datasets-png/test/liver_seg/liver_seg/'
tumor_masks_path = 'D:/Study/Thesis/LiTS/datasets-png/test/tumor_seg/tumor_seg/'
dst_data_path = 'D:/Study/Thesis/LiTS/liver-crops-test/ct/'
dst_masks_path = 'D:/Study/Thesis/LiTS/liver-crops-test/liver_seg/'
dst_tumor_path = 'D:/Study/Thesis/LiTS/liver-crops-test/tumor_seg/'

crop_dict = {}

# Load data
filenames, _ = utils.split_filenames_train_val(data_path, val_prec=0)
all_filenames_split = utils.split_to_patients(filenames)

# Display data info
num_patients = len(all_filenames_split)
num_files = sum([len(x) for x in all_filenames_split])
print('number of samples: ', num_files)
print('num patients: ', num_patients)

orig_height, orig_width = (512, 512)
num_channels = 3
n = 0
a = Progbar(num_files)

for i in range(num_patients):
    # load data to array
    img_filenames = all_filenames_split[i]
    mask_arr = np.zeros((len(img_filenames), orig_width, orig_height))
    tumor_mask_arr = np.zeros((len(img_filenames), orig_width, orig_height))
    img_arr = np.zeros((len(img_filenames), orig_width, orig_height, num_channels))
    for k, filename in enumerate(img_filenames):
        img_arr[k] = cv2.imread(os.path.join(data_path, filename))
        mask_arr[k] = cv2.imread(os.path.join(masks_path, filename.replace('volume', 'segmentation')), cv2.IMREAD_GRAYSCALE)
        tumor_mask_arr[k] = cv2.imread(os.path.join(tumor_masks_path, filename.replace('volume', 'segmentation')), cv2.IMREAD_GRAYSCALE)

    # 3D CC
    seg_liver_CC = utils.get_CC_largerThanTh(np.where(mask_arr > 0, 1, 0), dbg=False)

    # 3-D Crop coordinates
    (h1, h2, w1, w2) = utils.get_crop_coordinates_3D(seg_liver_CC, dbg=False)

    # Crop in 3D
    crop_img_arr = img_arr[:, h1:h2, w1:w2, :]
    crop_mask_arr = mask_arr[:, h1:h2, w1:w2]
    crop_tumor_mask_arr = tumor_mask_arr[:, h1:h2, w1:w2]

    for k, filename in enumerate(img_filenames):
        mask_liver = crop_mask_arr[k]
        mask_tumor = crop_tumor_mask_arr[k]
        if mask_liver.max() > 0:  # liver exists

            crop_img = crop_img_arr[k]
            crop_dict[filename] = (h1, h2, w1, w2)

            if mask_liver.max() == 1:
                # save 10% remaining liver only for false samples in data:
                if np.random.random() > 0.1:
                    continue # --> in 90% of the times- continue without saving

            img_name = filename
            cv2.imwrite(os.path.join(dst_data_path, img_name), crop_img.astype('uint8'))
            cv2.imwrite(os.path.join(dst_masks_path, img_name.replace('volume', 'segmentation')), mask_liver)
            cv2.imwrite(os.path.join(dst_tumor_path, img_name.replace('volume', 'segmentation')), mask_tumor)

        n += 1
        a.update(n)

    # Save crops data dict
    try:
        import cPickle as pickle
    except ImportError:  # python 3.x
        import pickle

    with open(dst_masks_path + '/crop_list.p', 'wb') as fp:
        pickle.dump(crop_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    fp.close()