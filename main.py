# import the necessary packages
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import os

# import user defined functions
import functions as f


def read_digits_from_display(path):
    """
    Read digits from display

    Args:
        path (str): path to image with a display

    Returns:
        number on display
    """

    # step 0: read the image
    image_original = f.read_image(path=path)

    # step 1: resize the image
    image_resized = f.resize_image(
        image=image_original, image_height=500)

    # step 2: make image gray
    image_grayed = f.gray_image(image=image_resized)

    # step 3: blur image
    image_blurred = f.blur_image(
        image=image_grayed, kernel_size_width=7, kernel_size_height=7)

    # step 4: get image edges
    image_edged = f.get_image_edges(image=image_blurred, threshold_1=0,
                                    threshold_2=150, edges=255)

    # step 5: find display
    display_grayed, display = f.extract_display(
        image_resized=image_resized, image_grayed=image_grayed, image_edged=image_edged, accuracy=0.05, width_display=[190, 200], height_display=[65, 75])

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
    cv2.imwrite("images_result/"+str(os.path.splitext(os.path.basename(
        path))[0])+"_"+str(result)+".jpg", display_annotated)

    return result


result = read_digits_from_display("images_set_1/IMG_5289.JPG")
# # loop through all images in dataset
# directory = "images_set_1"
# for path in os.listdir(directory):
#     full_path = os.path.join(directory, path)
#     if os.path.isfile(full_path):
#         try:
#             result = read_digits_from_display(full_path)
#             print(full_path, result)
#         except:
#             print(full_path, "Unable to read digits")
