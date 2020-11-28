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
image = f.resize_image(image=image_original, image_height=500)

# step 2: make image gray
gray = f.gray_image(image=image)

# step 3: blur image
blurred = f.blur_image(
    image=gray, kernel_size=(5, 5), sigma_x=0, sigma_y=0)

# step 4: get image edges
edged = f.get_image_edges(image=blurred, threshold_1=50,
                          threshold_2=200, edges=255)

# step 5: find display
warped, output = f.extract_display(
    image_resized=image, image_grayed=gray, image_edged=edged)

# step 6: make display binary (black or white)
thresh = f.make_pixels_black_or_white(warped, threshold=0, maximum_value=255)

# step 7: remove noise from display
thresh = f.remove_noise_from_image(
    thresh, shape=cv2.MORPH_ELLIPSE, size=(1, 5))

# step 8: find digit areas
contoursOfDigits = f.find_digit_areas(
    binary_display_without_noise=thresh, display=output, min_width_digit_area=15, max_width_digit_area=30, min_height_digit_area=30, max_height_digit_area=40)

# step 9: read digits
result = f.read_digits(
    binary_display_without_noise=thresh, display=output, contours_of_digits=contoursOfDigits, alpha=0.25, beta=0.15, gamma=0.05, min_fill_area=0.5)

print(result)
