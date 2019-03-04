import numpy as np
from PIL import Image

# Global variable declaration
DIR = "D:/dsp_project2_data/"


def train_file_name():
    """
    Reading the file train.txt to get filenames for train data
    :return: a list of filenames
    """
    path = DIR + "/train.txt"
    with open(path) as file:
        hash_name = file.read().split('\n')
    # returns all the filenames as a list except the empty line at end of file
    return hash_name[:-1]


def get_score_image(pred_image, actual_image):
    """
    Compares the the given predicted image and actual image using
    intersection-over-union(IOU).

    :pred_image:The mask image that is predicted
    :actual_image:The original mask image
    :return:accuracy/IOU score if there is intersection of images
            otherwise 0.
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
    Compares all the predicted masks and actual masks in a path using
    function to get individual accuracies

    :pred_masks_path:The mask image path where the predicted masks are saved
    :actual_masks_path:The mask image path where the actual masks are saved
    :filenames: The list filenames on which accuracies are to be calculated
    :return: mean accuracy of all the predicted masks
    """
    acc_arr = np.array([])
    for file in filenames:
        pred_mask = Image.open(pred_masks_path + file + '.png')
        actual_mask = Image.open(actual_masks_path + file + '.png')
        accuracy = get_score_image(pred_mask, actual_mask)
        acc_arr = np.append(acc_arr, accuracy)
    return np.mean(acc_arr)


def main():
    image_path = DIR + "masks/"
    filename = train_file_name()
    accuracy = get_mean_score(image_path, image_path, filename)
    print("Accuracy=", accuracy)


if __name__ == "__main__":
    main()
