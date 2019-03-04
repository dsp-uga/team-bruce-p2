import numpy as np
from PIL import Image
import os
from collections import Counter
from operator import itemgetter


dl = DataLoader()


def different_sizes(key):
    """
    Prints the counter of different image shape in the data
    
    Arguments:
    ---------
    key : string
        One of 'train', 'test' or 'masks'
    
    Returns:
    -------
    dimension_counter: Counter 
        Counter of each image shape
    """
    path = os.path.join(dl.dataset_folder, "masks")
    if key == 'train':
        dimension_counter = Counter(dl.train_dimensions)
    elif key == 'test':
        dimension_counter = Counter(dl.test_dimensions)
    elif key == 'masks':
        dimension_counter = Counter(dl.masks_dimensions)
    else:
        print('Invalid Argument: Please enter one of \'test\', \'train\', \'masks\'')
    return dimension_counter


def max_size(key):
    """
    Gives the maximum image shape across each dimension in the set

    Arguments:
    ---------
    key : string
        One of 'train', test', 'masks'

    Returns:
    -------
    (max_0, max_1) : Tuple
        Maximum shape of image across each dimension

    """
    unique_dimensions = list(set(dl.train_dimensions))
    max_0 = (unique_dimensions.sort(key=itemgetter(0)))[0][0]
    max_1 = (unique_dimensions.sort(key=itemgetter(1)))[0][1]
    return (max_0, max_1)
