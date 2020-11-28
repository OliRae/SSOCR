# import the necessary packages
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

# import user defined functions
import functions as f

# step 0: read the image
image_original = f.read_image("examples/thermostat.jpg")

# step 1: resize the image
image_resized = f.resize_image(image=image_original, image_height=500)

# step 2: make image gray
image_grayed = f.gray_image(image=image_resized)

# step 3: blur image
image_blurred = f.blur_image(
    image=image_grayed, kernel_size=(5, 5), sigma_x=0, sigma_y=0)

# step 4: get image edges
image_edged = f.get_image_edges(image=image_blurred, threshold_1=50,
                                threshold_2=200, edges=255)

# step 5: find display
display_grayed, display = f.extract_display(
    image_resized=image_resized, image_grayed=image_grayed, image_edged=image_edged)

# step 6: make display binary (black or white)
display_binary = f.make_pixels_black_or_white(
    display_grayed, threshold=50)

# step 7: remove noise from display
display_binary_sharp = f.remove_noise_from_image(
    display_binary, shape=cv2.MORPH_ELLIPSE, size=(8, 12))

# step 8: find digit areas
contoursOfDigits = f.find_digit_areas(
    binary_display_without_noise=display_binary_sharp, display=display, min_width_digit_area=100, max_width_digit_area=170, min_height_digit_area=180, max_height_digit_area=250)

# step 9: read digits
result = f.read_digits(
    binary_display_without_noise=display_binary_sharp, display=display, contours_of_digits=contoursOfDigits, alpha=0.25, beta=0.15, gamma=0.05, min_fill_area=0.5)

print(result)
