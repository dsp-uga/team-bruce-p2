import numpy as np
from PIL import Image
from collections import Counter

# Global Variable Declaration
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


def test_file_name():
    """
    Reading the file train.txt to get filenames for train data
    :return: a list of filenames
    """
    path = DIR + "/test.txt"
    with open(path) as file:
        hash_name = file.read().split('\n')
        # returns all the filenames as a list except the empty line at EOF
    return hash_name[:-1]


def different_sizes(filenames):
    """
    Printing the different sizes in the data
    :filenames: the files to which sizes are to be measured
    """
    path = DIR + "/masks/"
    lis = []
    for i in range(len(filenames)):
        if len(filenames[i]) > 0:
            im = np.array(Image.open(path+filenames[i]+'.png').convert('L'))
            lis.append(im.shape)
    print(Counter(lis))


def main():
    """
    prints the different image sizes of train dataset

    """
    DIR = "D:/dsp_project2_data/"
    train_hash = train_file_name()
    test_hash = test_file_name()
    different_sizes(train_hash)


if __name__ == "__main__":
    main()
