from PIL import Image
import numpy as np
import cv2
import glob

dataset_path = "D:\\Research\\Viasat\\Implementation\\Experiment 4\\Dataset\\testset\\Set5"

image_paths = glob.glob(f"{dataset_path}\\LR\\X4\\imgs\\*")

sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = f"EDSR_x4.pb"
sr.readModel(path)
sr.setModel("edsr",4)

for img_path in image_paths:
    fname = img_path.split("\\")[-1]
    img = cv2.imread(img_path)
    hr_img = cv2.imread(f"{dataset_path}\\HR\\{fname}")
    result = sr.upsample(img)
    psnr = cv2.PSNR(hr_img,result)
    print(psnr)
    wr_path = fname.split('.')[0] + '_sr.png'
    cv2.imwrite(wr_path, result)
