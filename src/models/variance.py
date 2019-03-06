"""                     
This script implements the Variance technique on the dataset to create masks on the
test set - hard thrsholding is done to give labels to the final masks generated.
------------------
Author : Yang Shi
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2 as cv
import pandas as pd
from ..data_loader import DataLoader
import numpy as np
import logging


logger = logging.getLogger(__name__)
dl = DataLoader()


def get_image(hash_code):
    """
    Reads images from the cilia_dataset directory as numpy arrays

    Arguments
    ---------
    hash_code : string
        Hash codes for the train samples

    Returns:
    --------
    images : numpy array
        3D numpy of the train sample, in which, the first dimension is the frame index  
    """
    image_folder = os.path.join(dl.dataset_folder, 'test/data', hash_code)
    image_frames = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = []
    for frame in image_frames:
        images.append(cv.cvtColor(cv.imread(os.path.join(image_folder, frame)), 
            cv.COLOR_BGR2GRAY))
    images = np.array(images)
    return images


def std_thresholding(std_image,threshold):
    """
    Performs thresholding on the predictions to assign labels 0, 2.

    Arguments
    ---------
    std_image : string
        Hash codes for the train samples
    threshold : float
        Threshold of standard deviation (std), i.e, if std > threshold then assigned 
        as celia (label as 2)
        
    Returns:
    --------
    predicted_mask : numpy array
        The final predicted mask from the dataset.
    """
    predicted_mask = np.array([[2 if x > threshold else 0 for x in line] 
        for line in std_image])
    return predicted_mask

def Variance(model):
    """
    Runs the variance technique and writes the final predictions to results directory

    Arguments
    ---------
    model : string
        Model, as input by the user - 'unet'by default
    """

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
    if not os.path.isdir(os.path.join('results',model)):
        os.makedirs(os.path.join('results',model))
    for index, test_sample in test_df.iterrows():
        cv.imwrite(os.path.join('results', model, test_hash[index] + '.png'), test_sample['pred_mask'])
