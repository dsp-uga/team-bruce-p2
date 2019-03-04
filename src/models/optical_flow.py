'''
Optical Flow Model using test dataset

This model is based on the implementation from
"https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html"
"https://github.com/dsp-uga/Flanagan/tree/master/OpticalFlow"
'''

import cv2
import numpy as np
from PIL import Image
import os
from src.data_loader import DataLoader
import logging


logger = logging.getLogger(__name__)
dl = DataLoader()


def OpticalFlow(model):
    """
    Applies optical flow technique to save new masks
    
    Arguments
    ----------
    model : string
        User-defined model, to create cilia masks
    """
    count = 0
    for file in dl.test_hashes:
        count += 1
        # Previous image
        prvs = cv2.imread('../data/'+ file + '/frame0000.png', 0)
        dim_prvs = (prvs.shape[0], prvs.shape[1])
        dim_hsv = (prvs.shape[0], prvs.shape[1], 3)
        hsv = np.zeros(dim_hsv, np.uint8)
        hsv[...,1] = 255
        # Masks
        mask = np.zeros(dim_prvs, np.uint8)
        sum_mask = np.zeros(dim_prvs, np.uint8)

        for i in range(1,100):
            # Next image
            nxt = cv2.imread('../data/'+ file + '/frame00'+'%02d' % i+'.png', 0)
            flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = (ang * 180) / (np.pi * 2)
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            omg = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            omg2 = omg
            # Each image corresponding to one omg, use 0, 1, 2 to mark background, cell, celia
            # Hard coding the thresholds
            omg2[omg2 < 50] = 0
            omg2[(omg2 > 49) and (omg2 < 120)] = 1
            omg2[omg2 > 119] = 2
            # Add each frame to sum_mask
            for r in range(omg2.shape[0]):
                for c in range(omg2.shape[1]):
                    if omg2[r][c] == 0:
                        sum_mask[r][c] += 0
                    elif omg2[r][c] ==1:
                        sum_mask[r][c] += 1
                    else:
                        sum_mask[r][c] += 10
        # Scale sum_mask
        # Hard coding thresholds
        mask[(sum_mask > 10) and (sum_mask < 100)] = 1
        mask[sum_mask > 100] = 2
        omask = Image.fromarray(mask)
        omask.save(os.path.join('results', model, file + '.png'), 0)
        logger.info('Progress: ' + str(int(count / len(dl.test_hashes) * 100)) + '%')
