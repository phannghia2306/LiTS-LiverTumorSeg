import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def neuroToRadio(vol, flip_flag):
    """ Change from neurological to radiological orientation. """
    vol = np.transpose(vol, axes=(1, 0, 2))
    if flip_flag:
        vol = np.fliplr(vol)
    vol = np.flipud(vol)

    return vol

def adjust_HU_value(volume, x=240, y=-160):
    # The upper grey level x and the lower grey level y
    volume[volume > x] = x  # above x will be white 
    volume[volume < y] = y  # below y will be black
    return volume
    
def normalize(volume):
    # Normalize into range [0, 1]
    vol_max = volume.max()
    vol_min = volume.min()
    volume = (volume - vol_min) / (vol_max - vol_min)
    return volume

def nifti_to_png(nifti_path, file_name, png_path, vol_mode = True):
    volume_path = os.path.join(nifti_path, file_name)
    load_volume = nib.load(volume_path)

    volume = neuroToRadio(load_volume.get_fdata(), 0)
    # w, h: slice size; d: number of slices in volume
    (w, h, d) = volume.shape

    if vol_mode: # Save CT images
        # HU value adjusment
        adj_vol = adjust_HU_value(volume)
        # Normalize volume into range [0, 1]
        norm_vol = normalize(adj_vol)

        for i in range(d):
            slice = norm_vol[:,:,i] * 65535
            slice_path = os.path.join(png_path, file_name.replace('.nii', '_' + str(i) + '.png'))
            cv2.imwrite(slice_path, slice.astype(np.uint16)) 
    else: # Save segmentation images
        norm_vol = normalize(volume)
        for i in range(d):
            slice = norm_vol[:,:,i] * 255
            slice_path = os.path.join(png_path, file_name.replace('.nii', '_' + str(i) + '.png'))
            cv2.imwrite(slice_path, slice) 
