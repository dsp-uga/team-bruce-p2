"""
Script to post-process the 'npy' files, so as to obtain the prediction masks. Includes
frame unpadding, histogram binning and thresholding. 
---------------------------
Author : Aashish Yadavally
"""


import os
import numpy as np
import matplotlib.image as mpimg
from ..data_loader import DataLoader
import cv2
import logging


logger = logging.getLogger(__name__)
dl = DataLoader()


def get_final_prediction(image, min_value, max_value):
    """
    All pixels with prediction probability outside the interval (min_value, max_value)
    are filled with 0's and the ones in the interval with '2'.
    Arguments
    ---------
    image : numpy array
        Numpy array of the unpadded prediction image
    min_value : float
        Lower bound of prediction probability
    max_value : float
        Upper bound of prediction probability
        
    Returns
    -------
    apply_label_2 : numpy array
        Numpy array of image with 0,2 labels
    """
    # Assigns 0 to all pixels with prediction probability less than the min value
    apply_min = np.where(image < min_value, 0, image) 
    # Assigns 0 to all pixels with prediction probability more than the max value
    apply_max = np.where(apply_min > max_value, 0, apply_min)
    # Assigns 2 to all pixels with prediction probability in interval (min value, max value)
    apply_label_2 = np.where(apply_max == 0, 0, 2) 
    return apply_label_2


def frame_unpad(image, required_shape):
    """
    Unpad the pads added before UNet model training
    Arguments
    ---------
    image : numpy array
        Numpy array of the prediction image
    required_shape : tuple
        Shape of corresponding test image
        
    Returns
    -------
    image : numpy array
        Removing the left-right and top-bottom pads which were added before U-Net
        model training
    """
    left_right_pad = int((image.shape[0] - required_shape[0])/2) # Gets column pads
    top_bottom_pad = int((image.shape[1] - required_shape[1])/2) # Gets row pads
    # Slicing the column pads
    side_slice = slice(left_right_pad, image.shape[0] - left_right_pad)
    # Slicing the row pads 
    vertical_slice = slice(top_bottom_pad, image.shape[1] - top_bottom_pad)
    return image[side_slice, vertical_slice]

      
def get_thresholds(reshaped_prediction):
    """
    Gets thresholds of the bin which has minimum count of prediction probabilities
    Arguments
    ---------
    reshaped_prediction : numpy array
        (640,640) shaped prediction image
    
    Returns
    -------
    [min_value, max_value] : list
        List of minimum and maximum value of the histogram bin with minimum number 
        of pixels count
    """
    hist = np.histogram(np.ndarray.flatten(reshaped_prediction), bins = 3)
    hist_list = list(hist)
    bin_index = np.argmin(np.array(hist[0])) # Calculating the minimum number of counts in bins
    min_value = hist_list[1][bin_index] # Minimum value of threshold of required bin
    max_value = hist_list[1][bin_index+ 1] # Maximum value of threshold of required bin
    return [min_value, max_value]


def histogram_binning(model):
    """
    Divides prediction probabilities of all test images into three bins, assigning 
    '2' to bin with minimum count of prediction probabilities in a bin, and the 
    remaining bins are assigned '0'

    Arguments:
    ---------
    model : string
        User defined model input
    """
    path = os.path.join('results', model, 'predictions')
    os.makedirs(path)
    for i in range(len(dl.test_hashes)):
        # Loading saved numpy arrays of predictions from U-Net
        prediction = np.load(os.path.join('results', model, 'pred_array', 'prediction' + str(i) + ".npy"))
        # Reshaping the prediction into (640, 640)
        reshaped_prediction = np.ndarray.flatten(prediction).reshape(640,640)
        # Get minimum and maximum thresholds
        min_value, max_value = get_thresholds(reshaped_prediction)
        # Writing 0's and 2's into final image
        categorical_image = get_final_prediction(reshaped_prediction, min_value, max_value)
        # Unpadding extra pixels in prediction image
        unpadded_image = frame_unpad(categorical_image, dl.test_dimensions[i])
        cv2.imwrite(os.path.join(path, dl.test_hashes[i] + '.png'),
            np.array(unpadded_image, dtype=np.uint8))
    logger.info('Predictions have successfully been saved as images!')
