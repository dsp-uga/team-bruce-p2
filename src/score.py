def get_score_image(pred_image, actual_image):
    """
    Compares the the given predicted image and actual image using
    intersection-over-union(IOU).
    The function raises an error if the images sizes are different.

    :pred_image:The mask image that is predicted
    :actual_image:The original mask image
    :return:accuracy/IOU score if the image sizes are equal
            otherwise raises an error.
    """
    pred_img_arr = np.array(pred_image)
    actual_img_arr = np.array(actual_image)
    # print(np.shape(pred_img_arr))
    if np.shape(pred_img_arr) != np.shape(actual_img_arr):
        raise Exception('The predicted image and actual image are of different dimensions')
    else:
        pred_img_two = (pred_img_arr == 2)
        actual_img_two = (actual_img_arr == 2)
        common_two = logical_and(pred_img_two, actual_img_two)
        overall_two = logical_or(pred_img_two, actual_img_two)
        common = np.sum(common_two)
        overall = np.sum(overall_two)
        if overall > 0:
            return common/overall
        else:
            return 0


def main():
    # image1 = Image.open("D:/dsp_project2_data/masks/0b599d0670fcbafcaa8ed5567c0f4b10b959e6e49eed157be700bc62cffd1876.png")
    # image2 = Image.open("D:/dsp_project2_data/masks/04c85d8f80b9130890fadec7a4da0a1b61f29dda1219c9b745b76d6af654ae62.png")
    get_score_image(image1, image1)
    accuracy = get_score_image(image1, image1)
    print("Accuracy=", accuracy)


if __name__ == "__main__":
    main()
