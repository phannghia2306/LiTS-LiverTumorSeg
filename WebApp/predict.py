from cv2 import norm
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model, load_model
from keras.applications.resnet import ResNet50
from architecture import *
from utils import *
import tqdm
import time
from flask import Flask, redirect, url_for, render_template, request, session, flash, get_flashed_messages, send_file

def HU_clipping(array, slope, inter, range = (0, 255)):
    HU_array = array #* slope + inter
    HU_array[HU_array < range[0]] = range[0]
    HU_array[HU_array > range[1]] = range[1]
    norm_array = HU_array #- inter) / slope
    norm = (norm_array - range[0]) / (range[1] - range[0])
    norm = np.rot90(norm)
    # norm = np.flip(norm, axis=0)
    return norm

def check_data(array):
    shape = array.shape
    if shape[0] == shape[1]:
        return "channel last"
    if shape[2] == shape[1]:
        return "channel first"

def split_and_get_idx_all(list, idx):
    val_list = []
    for val in list:
        val_list.append(int((val.split('_')[idx]).split('-')[1]))
    return val_list

def get_unique_indices(filenams, idx=0):
    patient_indices = split_and_get_idx_all(filenams, idx)
    return list(set(patient_indices))
            
def convert_volume_to_nifti(volume_arr,
                            filenames,
                            output_dir,
                            orig_volume_dir='D:/Study/Thesis/Workspace/',
                            test_mode=True):
    import nibabel as nib

    volume_idx = get_unique_indices(filenames)[0]
    slice_indices = [int(filename.split('_')[-1][:-4]) for filename in filenames] # list of existing segmentation slices

    test_vol_filename = 'volume-{}.nii'.format(volume_idx)
    # test_vol_filename = 'segmentation-{}.nii'.format(volume_idx)
    if test_mode:
        test_seg_filename = 'test-segmentation-{}.nii'.format(volume_idx)
    else:
        test_seg_filename = 'segmentation-{}.nii'.format(volume_idx)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Processing test volume: ', test_vol_filename, ' --> ', test_seg_filename)

    # Load original volume
    curr_seg_data = nib.load(os.path.join(orig_volume_dir, test_vol_filename))# pointer/copy???
    # seg_vol = curr_seg_data.get_data().astype('uint8')

    seg_vol = np.zeros(curr_seg_data.shape).astype('uint8')

    for i, img in enumerate(volume_arr):
        curr_idx = slice_indices[i]
        # Convert Radio to neuro
        curr_seg = np.fliplr(np.transpose(img))
        seg_vol[:, :, curr_idx-1] = curr_seg

    new_img = nib.Nifti1Image(seg_vol, curr_seg_data.affine, curr_seg_data.header)
    nib.save(new_img, os.path.join(output_dir, test_seg_filename))

def liver_predict(nii_path, model_path, save_path, input_shape = (512, 512, 3), batch_size = 1):
    nii_file = nib.load(nii_path)
    arr = nii_file.get_data()
    slope = nii_file.dataobj.slope
    inter = nii_file.dataobj.inter
    shape = arr.shape
    if check_data(arr) == "channel last":
        image_arr = np.zeros((shape[2], 512, 512, 3))
        for i in range(shape[2]):
            norm_arr = HU_clipping(arr[:,:,i], slope=slope, inter=inter, range=(-160, 240))
            norm_arr = np.expand_dims(norm_arr, axis=2)
            norm_img = np.concatenate((norm_arr,norm_arr, norm_arr), axis=2)
            image_arr[i,:,:,:] = norm_img
    if check_data(arr) == "channel first":
        image_arr = np.zeros((shape[0], 512, 512, 3))
        for i in range(shape[0]):
            norm_arr = HU_clipping(arr[i,:,:], slope=slope, inter=inter, range=(-160, 240))
            norm_arr = np.expand_dims(norm_arr, axis=2)
            norm_img = np.concatenate((norm_arr,norm_arr, norm_arr), axis=2)
            image_arr[i,:,:,:] = norm_arr
    arr_with_batch = np.zeros((image_arr.shape[0]//batch_size + 1, batch_size, 512, 512, 3))
    id = 0
    print("Create batch")
    for i in tqdm.tqdm(range(image_arr.shape[0]//batch_size + 1)):
        for j in range(batch_size):
            arr_with_batch[i, j, :, :, :] = image_arr[id,:,:,:]
            if id < image_arr.shape[0]-1:
                id+=1
    model = build_resnet50_unet(input_shape)
    model.load_weights(model_path)
    predicted_arr = np.zeros((shape[2], 512, 512))
    print("Start predict")
    id = 0
    for step in tqdm.tqdm(range(arr_with_batch.shape[0])):
        predict_arr = model.predict(arr_with_batch[step,:,:,:,:], verbose = 1)
        for j in range(batch_size):
            predicted_arr[id,:,:] = predict_arr[j,:,:,0]
            predicted_arr[id,:,:] = np.rot90(predicted_arr[id,:,:])
            predicted_arr[id,:,:] = np.rot90(predicted_arr[id,:,:])
            predicted_arr[id,:,:] = np.rot90(predicted_arr[id,:,:])
            
            if id < image_arr.shape[0]-1:
                id+=1
            else:
                break
    predicted_arr = np.around(predicted_arr)
    predicted_arr = np.asarray(predicted_arr, np.uint8)
    liver_channel_first = predicted_arr
    liver_channel_last = np.transpose(predicted_arr, (1, 2, 0))

    nii_result_file = nib.Nifti1Image(predicted_arr, nii_file.affine, nii_file.header)
    file_name= os.path.basename(nii_path)
    liver_name = file_name.replace("volume", "liver_mask")
    nib.save(nii_result_file, os.path.join(save_path, liver_name))
    return  liver_channel_first, liver_channel_last  

def liver_tumor_predict(nii_path = "", liver_model = None, tumor_model = None,
                        mode = "None",# or "tumor_seg", "liver_seg"
                        save_path = "", batch_size = 1, 
                        input_channel_mode="channel last",# or "channel first" 
                        output_channel_mode = "channel last", # or "channel first"
                        save = False,
                        liver_model_input_shape = (512,512,3), 
                        tumor_model_input_shape = (512,512,3)):
    # if mode = "liver seg" then  return liver_channel_first, liver_channel_last
    # if mode = "tumor seg" then  return liver_channel_first,tumor_channel_first, liver_channel_last,tumor_channel_last
    # if save = True then save the .nii file to the save_path
    
    nii_file = nib.load(nii_path)
    arr = nii_file.get_data()
    slope = nii_file.dataobj.slope
    inter = nii_file.dataobj.inter
    shape = arr.shape

    if input_channel_mode == "channel last":
        image_arr = np.zeros((shape[2], liver_model_input_shape[0], liver_model_input_shape[1], liver_model_input_shape[2]))
        for i in range(shape[2]):
            norm_arr = HU_clipping(arr[:,:,i], slope=slope, inter=inter, range=(-160, 240))
            norm_arr = np.expand_dims(norm_arr, axis=2)
            norm_img = np.concatenate((norm_arr,norm_arr, norm_arr), axis=2)
            image_arr[i,:,:,:] = norm_img
    if input_channel_mode == "channel first":
        image_arr = np.zeros((shape[0], liver_model_input_shape[0], liver_model_input_shape[1], liver_model_input_shape[2]))
        for i in range(shape[0]):
            norm_arr = HU_clipping(arr[i,:,:], slope=slope, inter=inter, range=(-160, 240))
            norm_arr = np.expand_dims(norm_arr, axis=2)
            norm_img = np.concatenate((norm_arr,norm_arr, norm_arr), axis=2)
            image_arr[i,:,:,:] = norm_arr
    arr_with_batch = np.zeros((image_arr.shape[0]//batch_size + 1, batch_size, liver_model_input_shape[0], liver_model_input_shape[1], liver_model_input_shape[2]))
    id = 0
    # flash("Create batch for {}".format(os.path.basename(nii_path)))
    print("Create batch for {}".format(os.path.basename(nii_path)))
    for i in tqdm.tqdm(range(image_arr.shape[0]//batch_size + 1)):
        for j in range(batch_size):
            arr_with_batch[i, j, :, :, :] = image_arr[id,:,:,:]
            if id < image_arr.shape[0]-1:
                id+=1
    predicted_arr = np.zeros((shape[2], liver_model_input_shape[0], liver_model_input_shape[1]))
    # flash("Start predict liver")
    print("Start predict liver")
    id = 0
    for step in tqdm.tqdm(range(arr_with_batch.shape[0])):
        predict_arr = liver_model.predict(arr_with_batch[step,:,:,:,:], verbose = 1)
        for j in range(batch_size):
            predicted_arr[id,:,:] = predict_arr[j,:,:,0]
            predicted_arr[id,:,:] = np.rot90(predicted_arr[id,:,:])
            predicted_arr[id,:,:] = np.rot90(predicted_arr[id,:,:])
            predicted_arr[id,:,:] = np.rot90(predicted_arr[id,:,:])
            
            if id < image_arr.shape[0]-1:
                id+=1
            else:
                break
    # print(type(predicted_arr[0,0,0]), np.min(predicted_arr), np.max(predicted_arr))
    predicted_arr = np.around(predicted_arr)
    # print(type(predicted_arr[0,0,0]), np.min(predicted_arr), np.max(predicted_arr))
    predicted_arr = np.asarray(predicted_arr, np.uint8)
    # print(type(predicted_arr[0,0,0]), np.min(predicted_arr), np.max(predicted_arr))
    # print(predicted_arr)
    liver_channel_first = predicted_arr
    liver_channel_last = np.transpose(liver_channel_first, (1, 2, 0))
    if mode == "liver_seg":
        if save:
            # save to nii file
            nii_result_file = nib.Nifti1Image(liver_channel_last, nii_file.affine, nii_file.header)
            file_name= os.path.basename(nii_path)
            liver_name = file_name.replace("volume", "liver_mask")
            nib.save(nii_result_file, os.path.join(save_path, liver_name))
        if output_channel_mode == "channel first":
            return liver_channel_first
        if output_channel_mode == "channel last":
            return liver_channel_last

    liver_exist = []
    for i in range(liver_channel_first.shape[0]):
        if (np.max(liver_channel_first[i,:,:]) == 1):
            liver_exist.append(i)
    # ---------------------------------------------------------------------------------------
    # Crop 3D
    h_min, h_max, w_min, w_max = get_crop_coordinates_3D(liver_channel_first, pad_size=1)
    liver_cropped_arr = np.zeros(shape=(len(liver_exist), h_max-h_min+1, w_max-w_min+1))
    channel_first_arr = np.transpose(arr, (2,0,1))
    for i in range(len(liver_exist)):
        liver_cropped_arr[i,:,:] = channel_first_arr[liver_exist[i], h_min:h_max+1, w_min:w_max+1]
    liver_cropped_arr_resize = np.zeros((liver_cropped_arr.shape[0], tumor_model_input_shape[0], tumor_model_input_shape[1]))
    for i in range(liver_cropped_arr.shape[0]):
        liver_cropped_arr_resize[i,:,:] = cv2.resize(liver_cropped_arr[i,:,:], (tumor_model_input_shape[0], tumor_model_input_shape[1]), interpolation = cv2.INTER_AREA)
    # Tumor prediction

    image_arr = np.zeros((liver_cropped_arr_resize.shape[0], tumor_model_input_shape[0], tumor_model_input_shape[1], tumor_model_input_shape[2]))
    for i in range(liver_cropped_arr_resize.shape[0]):
        norm_arr = HU_clipping(liver_cropped_arr_resize[i,:,:], slope=slope, inter=inter, range=(-160, 240))
        norm_arr = np.expand_dims(norm_arr, axis=2)
        norm_img = np.concatenate((norm_arr,norm_arr, norm_arr), axis=2)
        image_arr[i,:,:,:] = norm_arr
    arr_with_batch = np.zeros((image_arr.shape[0]//batch_size + 1, batch_size, tumor_model_input_shape[0], tumor_model_input_shape[1], tumor_model_input_shape[2]))
    id = 0
    # flash("Create batch for tumor segmentation")
    print("Create batch for tumor segmentation")
    for i in tqdm.tqdm(range(image_arr.shape[0]//batch_size + 1)):
        for j in range(batch_size):
            arr_with_batch[i, j, :, :, :] = image_arr[id,:,:,:]
            if id < image_arr.shape[0]-1:
                id+=1
    predicted_arr = np.zeros((shape[2], tumor_model_input_shape[0], tumor_model_input_shape[1]))
    # flash("Start predict tumor")
    print("Start predict tumor")
    id = 0
    for step in tqdm.tqdm(range(arr_with_batch.shape[0])):
        predict_arr = tumor_model.predict(arr_with_batch[step,:,:,:,:], verbose = 1)
        for j in range(batch_size):
            predicted_arr[liver_exist[id],:,:] = predict_arr[j,:,:,0]
            predicted_arr[liver_exist[id],:,:] = np.rot90(predicted_arr[liver_exist[id],:,:])
            predicted_arr[liver_exist[id],:,:] = np.rot90(predicted_arr[liver_exist[id],:,:])
            predicted_arr[liver_exist[id],:,:] = np.rot90(predicted_arr[liver_exist[id],:,:])
            
            if id < image_arr.shape[0]-1:
                id+=1
            else:
                break
    predicted_arr = np.around(predicted_arr)
    predicted_arr = np.asarray(predicted_arr, np.uint8)
    org_size_predicted_arr = np.zeros(liver_channel_first.shape)
    for i in range(org_size_predicted_arr.shape[0]):
        org_size_predicted_arr[i, h_min:h_max+1, w_min:w_max+1] = cv2.resize(predicted_arr[i,:,:], (w_max-w_min+1, h_max-h_min+1))

    tumor_channel_first = org_size_predicted_arr
    tumor_channel_first[liver_channel_first == 0] = 0
    tumor_channel_last = np.transpose(tumor_channel_first, (1, 2, 0))
    if mode == "tumor_seg":
        if save:
            # save to nii file
            nii_result_file = nib.Nifti1Image(tumor_channel_last, nii_file.affine, nii_file.header)
            file_name= os.path.basename(nii_path)
            liver_name = file_name.replace("volume", "tumor_mask")
            nib.save(nii_result_file, os.path.join(save_path, liver_name))
        if output_channel_mode == "channel first":
            return liver_channel_first,tumor_channel_first
        if output_channel_mode == "channel last":
            return liver_channel_last,tumor_channel_last
    
    merge_channel_first = liver_channel_first
    merge_channel_first[tumor_channel_first == 1] = 2
    merge_channel_last = np.transpose(merge_channel_first, (1, 2, 0))


    if save:
        # save to nii file
        nii_result_file = nib.Nifti1Image(merge_channel_last, nii_file.affine, nii_file.header)
        file_name= os.path.basename(nii_path)
        liver_name = file_name.replace("volume", "segmentation")
        nib.save(nii_result_file, os.path.join(save_path, liver_name))
        print("{} is saved successfully".format(liver_name))
        flash("{} is saved successfully".format(liver_name))

    if (output_channel_mode == "channel first"):
        return liver_channel_first, tumor_channel_first, merge_channel_first
    if (output_channel_mode == "channel last"):
        return liver_channel_last, tumor_channel_last, merge_channel_last
    



# liver_model_path = "./Liver_Lesions_Segmentaiton/models/liver_weights_best.hdf5"
# save_path = "./Liver_Lesions_Segmentaiton/files/predicted"
# tumor_model_path = "./Liver_Lesions_Segmentaiton/models/tumor_weights_best.hdf5"
# nii_path="./Liver_Lesions_Segmentaiton/files/uploaded/volume-130.nii"

# input_shape = (512, 512, 3)
# liver_model = build_resnet50_unet(input_shape)
# liver_model.summary()
# liver_model.load_weights(liver_model_path)

# input_shape = (320, 320, 3)
# tumor_model = build_resnet50_unet1(input_shape)
# tumor_model.summary()
# tumor_model.load_weights(tumor_model_path)



# liver_mask, tumor_mask, merged_mask = liver_tumor_predict(nii_path, liver_model, tumor_model, save_path, batch_size = 6, 
#                         input_channel_mode="channel last", output_channel_mode = "channel last")

# print(liver_mask.shape, np.min(liver_mask), np.max(liver_mask))
# print(tumor_mask.shape, np.min(tumor_mask), np.max(tumor_mask))
# print(merged_mask.shape, np.min(merged_mask), np.max(merged_mask))

# plt.subplot(1,3,1)
# plt.imshow(liver_mask[:,:,500])
# plt.subplot(1,3,2)
# plt.imshow(tumor_mask[:,:,500])
# plt.subplot(1,3,3)
# plt.imshow(merged_mask[:,:,500])
# plt.show()

# liver_model = load_model('./Liver_Lesions_Segmentaiton/models/liver_weights_best.h5')
# tumor_model = load_model('./Liver_Lesions_Segmentaiton/models/tumor-liver-crops-512x512_weights_best.h5')
# tumor_polar_model = load_model('Liver_Lesions_Segmentaiton/models/tumor-liver-crops-polar-512x512_weights_best.h5')

# data_path = "./Liver_Lesions_Segmentaiton/files/test_data/Over_270MB"
# output_dir = "./Liver_Lesions_Segmentaiton/files/submit"

# nii_list = os.listdir(data_path)
# nii_path_list = []
# for filename in nii_list:
#     nii_path_list.append(os.path.join(data_path, filename))
# save_path = output_dir
# # nii_path_list
# len(nii_path_list)

# for nii_path in nii_path_list:

# liver_tumor_predict(nii_path="./Liver_Lesions_Segmentaiton/files/uploaded/volume-0.nii", liver_model=liver_model, tumor_model=tumor_model,
#                           mode = "None",# or "tumor_seg", "liver_seg"
#                           save_path ="Liver_Lesions_Segmentaiton/files/submit", batch_size = 4, 
#                           input_channel_mode="channel last",# or "channel first" 
#                           output_channel_mode = "channel last", # or "channel first"
#                           save = True,
#                           liver_model_input_shape = (512,512,3), 
#                           tumor_model_input_shape = (512,512,3))

# nii_file = nib.load("./Liver_Lesions_Segmentaiton/files/submit/test-liver_tumor_mask-30.nii")
# arr = nii_file.get_data()
# print(arr.shape, np.min(arr), np.max(arr))
# plt.subplot(1,2,1)
# plt.imshow(arr[:,:,580])
# plt.show()