# import the necessary packages
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils


def read_image(path):
    # step 0: read original image

    # load the example image
    image = cv2.imread(path)

    # write out image
    cv2.imwrite("steps/step_0_original.jpg", image)

    return image


def resize_image(image, image_height=500):
    # step 1: resize the image

    # resize image
    image_resized = imutils.resize(image=image, height=image_height)

    # write out image
    cv2.imwrite("steps/step_1_resize.jpg", image_resized)

    return image_resized


def gray_image(image):
    # step 2: convert to greyscale

    # make image gray
    image_grayed = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    # write out image
    cv2.imwrite("steps/step_2_grayed.jpg", image_grayed)

    return image_grayed


def blur_image(image, kernel_size=(5, 5), sigma_x=0, sigma_y=0):
    # step 3: blur image to remove noise from image

    # make image blurry
    image_grayed = cv2.GaussianBlur(
        src=image, ksize=kernel_size, sigmaX=0, sigmaY=0)

    # write out image
    cv2.imwrite("steps/step_3_blurred.jpg", image_grayed)

    return image_grayed


def get_image_edges(image, threshold_1=50, threshold_2=200, edges=255):
    # step 4: get edges in the image

    # get edges
    image_edges = cv2.Canny(
        image=image, threshold1=threshold_1, threshold2=threshold_2, edges=edges)

    # write out image
    cv2.imwrite("steps/step_4_edges.jpg", image_edges)

    return image_edges


def extract_display(image_resized, image_grayed, image_edged, epsilon_factor=0.02):
    # step 5: find contours in the edge map, then sort them by their size in descending order

    contours = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    countoursOfDisplay = None

    # loop over the contours
    for c in contours:

        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(
            curve=c, epsilon=epsilon_factor * peri, closed=True)

        # if the contour has four vertices, then we have found the display
        if len(approx) == 4:
            countoursOfDisplay = approx
            break

    # extract the thermostat display, apply a perspective transform to it
    image_display_grayed = four_point_transform(
        image_grayed, countoursOfDisplay.reshape(4, 2))
    image_display = four_point_transform(
        image_resized, countoursOfDisplay.reshape(4, 2))

    # resize images
    image_display_grayed = resize_image(
        image_display_grayed, image_height=500)
    image_display = resize_image(image_display, image_height=500)

    # write out image
    cv2.imwrite("steps/step_5_display.jpg", image_display)

    return image_display_grayed, image_display


def make_pixels_black_or_white(image, threshold=0):
    # step 6: make image negative

    # For every pixel, the same threshold value is applied. If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value.
    binary_image = cv2.threshold(src=image, thresh=threshold, maxval=255,
                                 type=cv2.THRESH_BINARY_INV)[1]

    # write out image
    cv2.imwrite("steps/step_6_black_or_white_pixels.jpg",
                binary_image)

    return binary_image


def remove_noise_from_image(binary_image, shape_opening=cv2.MORPH_ELLIPSE, shape_closing=cv2.MORPH_ELLIPSE, size_opening=(1, 5), size_closing=(1, 5)):
    # step 7: remove noise from image

    # define structuring element for opening
    kernel_opening = cv2.getStructuringElement(
        shape=shape_opening, ksize=size_opening)

    # remove noise
    image_opening = cv2.morphologyEx(
        src=binary_image, op=cv2.MORPH_OPEN, kernel=kernel_opening)

    # write out image
    cv2.imwrite("steps/step_7_opening.jpg", image_opening)

    # define structuring element for opening
    kernel_closing = cv2.getStructuringElement(
        shape=shape_closing, ksize=size_closing)

    # close small holes inside the foreground objects, or small black points on the object
    image_closing = cv2.morphologyEx(
        src=image_opening, op=cv2.MORPH_CLOSE, kernel=kernel_closing)

    # write out image
    cv2.imwrite("steps/step_7_closing.jpg", image_closing)

    return image_closing


def find_digit_areas(binary_display_without_noise, display, min_width_digit_area=15, max_width_digit_area=30, min_height_digit_area=30, max_height_digit_area=40):
    # step 8: find digit areas

    # find contours in the thresholded image, then initialize the digit contours lists
    contours = cv2.findContours(image=binary_display_without_noise.copy(
    ), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contoursOfDigits = []

    # loop over the digit area candidates
    for c in contours:

        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Draw a red box around all digit area candidates
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # if the contour is sufficiently large, it must be a digit
        if (w >= min_width_digit_area and w <= max_width_digit_area) and (h >= min_height_digit_area and h <= max_height_digit_area):
            contoursOfDigits.append(c)

            # Draw a green box around the contours where a digit is
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imwrite("steps/step_8_digit_areas.jpg", display)

    return contoursOfDigits


def read_digits(binary_display_without_noise, display, contours_of_digits, alpha=0.25, beta=0.15, gamma=0.05, min_fill_area=0.5):

    # define the dictionary of digit segments so we can identify each digit on the thermostat
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 1, 0): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }

    # sort the contours from left-to-right
    contours_of_digits = contours.sort_contours(
        contours_of_digits, method="left-to-right")[0]

    # initialize the actual digits themselves
    digits = []

    # Initialize counter
    digit_count = 0

    # loop over each of the digits
    for c in contours_of_digits[0]:

        # extract the digit ROI (region of interest)
        (x, y, w, h) = cv2.boundingRect(c)
        roi = binary_display_without_noise[y:y + h, x:x + w]

        # Increase counter
        digit_count = digit_count + 1

        # Write out image of digit ROI
        cv2.imwrite("steps/step_9_RIO_" + str(digit_count) + ".jpg", roi)

        # compute the width and height of each of the 7 segments we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * alpha), int(roiH * beta))
        dHC = int(roiH * gamma)

        # To do: if ROI is greater than a certain width, than try to identiy 7 segments. If ROI is less than certain width, then idently 2 segments (can be 1)

        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w, h // 2)),  # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w, h))  # bottom
        ]
        on = [0] * len(segments)

        # sort the contours from left-to-right, then initialize the actual digits themselves loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of thresholded pixels in the segment, and then compute the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            cv2.imshow(mat=segROI, winname='gray')
            cv2.waitKey()
            print(xB, xA, yB, yA)
            # if the total number of non-zero pixels is greater than 50% of the area, mark the segment as "on"
            if total / float(area) > min_fill_area:
                on[i] = 1

        # lookup the digit and draw it on the image
        digit = DIGITS_LOOKUP[tuple(on)]
        digits.append(digit)
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.putText(display, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Make image bigger
    display = resize_image(display, image_height=300)

    # Write out image
    cv2.imwrite("steps/step_9_result.jpg", display)

    cv2.imshow(winname="Result is " + str(digits), mat=display)
    cv2.waitKey()

    return digits
