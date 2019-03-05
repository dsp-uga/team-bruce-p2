"""
    Author: Yang Shi
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2 as cv
import pandas as pd
from ..data_loader import DataLoader
import logging

logger = logging.getLogger(__name__)
dl = DataLoader()

def read_file(file_name):
    f = open(file_name, "r")
    return f.read().split()

def get_image(hash_code):
    '''
        input: hash_code for the train sample
        output: images: 3-D np array of the train sample, the first dimension is the frame.
        '''
    image_folder = DATA_DIR + hash_code
    image_frames = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = []
    for frame in image_frames:
        images.append(cv.cvtColor(cv.imread(os.path.join(image_folder, frame)), cv.COLOR_BGR2GRAY))
    images = np.array(images)
    return images

def std_thresholding(std_image,threshold):
    '''
        input: std_image: the std of the images in a sample
        threshold: threshold of std, if std > threshold then assigned as celia (label as 2)
        output: predicted_mask
        '''
    predicted_mask = np.array([[2 if x > threshold else 0 for x in line] for line in std_image])
    return predicted_mask

def Variance(model):
    test_hash = dl.test_hashes

    # Reading images

    test_df = pd.DataFrame(test_hash,columns=['Hash_code'])
    test_df['images'] = [get_image(x) for x in test_df['Hash_code']]

    # Calculate the std of all trainings
    test_df['std'] = [np.std(x, axis=0) for x in test_df['images']]

    # Generating predicted masks
    threshold = 5
    test_df['pred_mask'] = [std_thresholding(x,threshold) for x in test_df['std']]

    # Output the result
    if not os.path.isdir(os.path.join('results',model):
        os.mkdir(os.path.join('results',model)

    for index, test_sample in test_df.iterrows():
        cv.imwrite(os.path.join('results', model, file + '.png'), test_sample['pred_mask'])
