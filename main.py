# import the necessary packages
import cv2
import numpy as np
import imutils
import os
from imutils import contours
from imutils.perspective import four_point_transform

# import user defined functions
import functions as f


def load_image(path):
    """
    loads and resizes an image

    Args:
        path (str): path to image

    Returns:
        object: image
    """

    # read image
    image = f.read_image(path)

    # resize image
    image_resized = f.resize_image(image=image, image_height=500)

    return image_resized


def find_display_on_image(image):
    """
    Find display on image

    Args:
        path (str): path to image with a display

    Returns:
        four contour points of image
    """

    # step 2: make image gray
    image_grayed = f.gray_image(image=image)

    # step 3: blur image
    image_blurred = f.blur_image(
        image=image_grayed, kernel_size_width=7, kernel_size_height=7)

    # step 4: get image edges
    image_edged = f.get_image_edges(image=image_blurred, threshold_1=0,
                                    threshold_2=150, edges=255)

    # step 5: find display
    countours_of_display = f.identify_display_contours(
        image_resized=image, image_grayed=image_grayed, image_edged=image_edged, accuracy=0.05, width_display=[190, 200], height_display=[65, 75])

    return countours_of_display


def read_digits_from_display(image, countours_of_display):
    """
    Read digits from display. User can provide x, y, w, h of display region. If not provided, it will try to identify region automatically.

    Args:
        path (str): path to image
        estimated_display_region (dbl): optional; x, y, w, h of display

    Returns:
        number on display
    """

    # extract display
    display = f.extract_display_from_image(
        image_resized=image, countours_of_display=countours_of_display)

    # make image grey
    display_grayed = f.gray_image(image=display)

    # step 6: increase contrast of image
    display_contrast = f.increase_contrast(
        display_grayed, threshold=1, size=(10, 10))

    # step 7: make display binary (black or white)
    display_binary = f.make_pixels_black_or_white(
        display_contrast, threshold_1=291, threshold_2=23)

    # step 8: remove noise from display
    display_binary_sharp = f.remove_noise_from_image(
        display_binary, shape_opening=cv2.MORPH_ELLIPSE, shape_closing=cv2.MORPH_ELLIPSE, size_opening=(1, 1), size_closing=(31, 31))

    # step 9: find digit areas
    contoursOfDigits = f.find_digit_areas(
        binary_display_without_noise=display_binary_sharp, display=display, min_width_digit_area=60, max_width_digit_area=275, min_height_digit_area=320, max_height_digit_area=400)

    # step 10: read digits
    digits, display_annotated = f.read_digits(
        binary_display_without_noise=display_binary_sharp, display=display, contours_of_digits=contoursOfDigits, alpha=0.25, beta=0.15, gamma=0.05, min_fill_area=0.4, min_width_digit_1=60, max_width_digit_1=100, bottom_right_segment_offset_to_left=20)

    # step 11: convert to number
    result = f.convert_to_number(digits)

    # write image with annotations out
    cv2.imwrite("images_result/"+str(result)+".jpg", display_annotated)

    return result


# loop through all images in directory
directory = "images_set_1"
for path in os.listdir(directory):
    full_path = os.path.join(directory, path)
    try:

        # load image
        image = load_image(full_path)

        # find the display on the image
        contours_of_display = find_display_on_image(image)

        # if the display is not found, than take default location (requires fixed setup)
        if contours_of_display is None:
            contours_of_display = np.array(
                [[[133, 173]], [[139, 239]], [[328, 239]], [[324, 176]]])

        # read digits on display
        result = read_digits_from_display(image, contours_of_display)

        # print number on display
        print(result)

    except:
        print(full_path, " - Unable to read digits")
