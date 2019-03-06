"""
Optical Flow Model using test dataset combineed with K-Mean(3 mean) to cluster cell, background and celia

Optical flow is based on the implementation from:
"https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html"
"https://github.com/dsp-uga/Flanagan/tree/master/OpticalFlow"

K-Mean is based on the implementation from:
"https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html"
-----------------
Author : Lie Xian
"""


import cv2
import numpy as np
from PIL import Image
import os
from ..data_loader import DataLoader
import logging


dl = DataLoader()
logger = logging.getLogger(__name__)


def OpticalFlow(model):
    """
    Implementation of the Optical Flow model, which generates predictions based
    on hard thresholds

    Arguments
    ---------
    model : string
        User-inputted argument
    """
    test_hash = dl.test_hashes
    count = 0
    for hash_name in test_hash:
        count += 1
        # Previous image
        prvs = cv2.imread(os.path.join(dl.dataset_folder, 'test', 'data', 
            hash_name, 'frame0000.png'), 0)
        dim_prvs = (prvs.shape[0], prvs.shape[1])
        dim_hsv = (prvs.shape[0], prvs.shape[1], 3)
        hsv = np.zeros(dim_hsv, np.uint8)
        hsv[...,1] = 255
        # Masks
        mask = np.zeros(dim_prvs, np.uint8)
        sum_mask = np.zeros(dim_prvs, np.uint8)

        for i in range(1,100):
            # Next image
            nxt = cv2.imread(os.path.join(dl.dataset_folder, 'test/data', 
                hash_name, '/frame00'+'%02d' % i+'.png'), 0)
            flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = (ang * 180) / (np.pi * 2)
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            omg = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            omg2 = omg
            # Each image corresponding to one omg, use 0, 1, 2 to mark background, cell, celia
            # Hard code the thresholds
            omg2[omg2 < 33] = 0
            omg2[(omg2 > 32) and (omg2 < 129)] = 1
            omg2[omg2 > 128] = 2
            # Add each frame to sum_mask
            for r in range(omg2.shape[0]):
                for c in range(omg2.shape[1]):
                    if omg2[r][c] == 0:
                        sum_mask[r][c] += 0
                    elif omg2[r][c] == 1:
                        sum_mask[r][c] += 1
                    else:
                        sum_mask[r][c] += 2
            prvs = nxt
        img = Image.fromarray(sum_mask)
        if not os.path.isdir(os.path.join('results', model)):
            os.makedirs(os.path.join('results', model))
        img.save(os.path.join('results', model + '_thesholding', hash_name + '.png'), 0)
        logger.info(str(np.ceil(count / len(dl.test_hashes))) + '% progress completed!' )

        
def OpticalFlow_KMeans(model):
    """
    Implementation of the Optical Flow model, which generates predictions based
    on KMeans Clustering algorithm.

    Arguments
    ---------
    model : string
        User-inputted argument
    """
    test_hash = dl.test_hashes
    for hash_name in test_hash:
        img = cv2.imread(os.path.join(dl.dataset_folder, 'test', 'data', 
            hash_name, 'frame0000.png')
        Z = img.reshape((-1,3))
        # Convert to np.float32
        Z = np.float32(Z)
        # Define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        # Scale pixels to 0 or 2
        res2[res2 == np.amax(res2)] = 2
        res2[res2 != 2] = 0
        res3 = Image.fromarray(res2)
        res3.save(os.path.join('results', model + '_KMeans', hash_name + '.png'), 0)
