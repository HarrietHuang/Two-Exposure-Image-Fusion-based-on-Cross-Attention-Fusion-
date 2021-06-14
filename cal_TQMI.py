from helper import cal_TMQI
import os
import random
import shutil
import cv2
import numpy as np

accumulate_Q = 0
accumulate_S = 0
accumulate_N = 0

root = r"C:\Users\admin\Downloads\HDR-reconstruction-using-deep-CNNs-master\ori_HDR-reconstruction-using-deep-CNNs-master_ori\simple_model\40"
gt_root = r"C:\Users\admin\Downloads\random_select2"
for folder in os.listdir(gt_root):
    fuse_path = os.path.join(root, folder + '_fused.png')
    gt_path = os.path.join(gt_root, folder, 'GT.JPG')
    # if not os.path.exists(save_path):
    print(fuse_path)
    hdr = cv2.imread(gt_path)
    hdr = cv2.resize(hdr, (512, 512))
    ldr = cv2.imread(fuse_path)  # .resize((512,512))
    print(hdr.shape)
    print(ldr.shape)
    Q, S, N, s_local, s_maps = cal_TMQI(
        hdr.astype(np.float64), ldr.astype(np.float64))
    accumulate_Q += Q
    accumulate_S += S
    accumulate_N += N
print(accumulate_Q / len(root))
print(accumulate_S / len(root))
print(accumulate_N / len(root))
