import numpy as np
from PIL import Image
from src.data_loader import DataLoader


dl = DataLoader()


def get_score_image(pred_image, actual_image):
    """
    Compares the the given predicted image and actual image using
    intersection-over-union(IOU) metric.

    :pred_image: Predicted mask 
    :actual_image: Original mask
    :return: IOU score if there is an intersection of images, otherwise 0.
    """
    pred_img_arr = np.array(pred_image)
    actual_img_arr = np.array(actual_image)
    # Element wise multiplication of numpy arrays which gives 0,1 or 4
    # as the elements are 0,1 and 2
    pred_img_two = pred_img_arr == 2
    actual_img_two = actual_img_arr == 2
    # if actual_mask does not contain cilia label return 1
    if np.sum(actual_img_two) == 0:
        return 1
    else:
        common = np.sum(np.logical_and(pred_img_two, actual_img_two))
        total = np.sum(np.logical_or(pred_img_two, actual_img_two))
        return common/total


def get_mean_score(pred_masks_path, actual_masks_path, filenames):
    """
    Compares all the predicted masks and actual masks in a path using get_score_image
    function to get individual accuracies

    :pred_masks_path: Predicted mask path
    :actual_masks_path: Actual mask path 
    :filenames: List of filenames for which accuracy is to be calculated
    :return: Mean accuracy of all the predicted masks
    """
    acc_arr = np.array([])
    for file in filenames:
        pred_mask = Image.open(pred_masks_path + file + '.png')
        actual_mask = Image.open(actual_masks_path + file + '.png')
        accuracy = get_score_image(pred_mask, actual_mask)
        acc_arr = np.append(acc_arr, accuracy)
    return np.mean(acc_arr)

