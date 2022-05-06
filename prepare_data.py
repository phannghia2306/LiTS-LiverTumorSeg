import os
from utils import utils


data_path = "D:/Study/Thesis/LiTS/datasets"
dst_path = "D:/Study/Thesis/LiTS/datasets-png"
sub_folder = os.listdir(data_path)

for folder in sub_folder:
    file_names = os.listdir(os.path.join(data_path, folder))
    for file_name in file_names:
        file_path = os.path.join(data_path, folder)
        if 'volume' in file_name:
            ct_path = os.path.join(dst_path, folder, 'ct/ct')
            utils.nifti_to_png(nifti_path=file_path, file_name=file_name, png_path=ct_path, ct=True)
        if 'segmentation' in file_name:
            liver_path = os.path.join(dst_path, folder, 'liver_seg/liver_seg')
            tumor_path = os.path.join(dst_path, folder, 'tumor_seg/tumor_seg')
            utils.nifti_to_png(nifti_path=file_path, file_name=file_name, png_path=liver_path, liver_seg=True)     
            utils.nifti_to_png(nifti_path=file_path, file_name=file_name, png_path=tumor_path, tumor_seg=True)  
