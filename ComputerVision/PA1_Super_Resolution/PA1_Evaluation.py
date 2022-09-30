# Evaluation

import numpy as np
import cv2

dir_path = "./images/"


def read_image(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # cv2.imshow(gray_img)
    return gray_img


def get_mse(result, ground_truth):
    mse = np.sum(np.square(np.subtract(ground_truth, result)) / (height * width)).mean()
    return mse


def get_psnr(result_path):
    result = read_image(result_path)
    psnr = np.sum(10 * np.log10((255 * 255) / get_mse(result, groud_truth))).mean()
    return psnr


if __name__ == "__main__":
    img_gt_path = dir_path + "HR.png"
    groud_truth = read_image(img_gt_path)
    height, width = 256, 256

    upsampled_path = dir_path + "upsampled.png"

    result1_path = dir_path + "results/method1.png"
    result2_path = dir_path + "results/method2.png"

    print(get_psnr(upsampled_path))
    print(get_psnr(result1_path))
    print(get_psnr(result2_path))
