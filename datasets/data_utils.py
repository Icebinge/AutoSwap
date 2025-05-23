import numpy as np
import torch
import cv2

def mask_score(mask):
    '''根据连接性对掩码进行评分'''
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    conc_score = np.max(cnt_area) / np.sum(cnt_area)
    return conc_score

def sobel(img, mask, thresh = 50):
    '''计算高频图'''
    H, W = img.shape[0], img.shape[1]
    img = cv2.resize(img, (256, 256))
    mask = (cv2.resize(mask, (256, 256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)

    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = Ksize)
    sobel_X = cv2.convertScaleAbs(sobelx)
