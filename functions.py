# import the necessary packages
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils


def read_image(path):
    """
    loads an image

    Args:
        path (str): path to image

    Returns:
        object: image
    """

    # load the example image
    image = cv2.imread(path)

    # write out image
    cv2.imwrite("steps/step_0_original.jpg", image)

    return image


def resize_image(image, image_height=500):
    """
    Resize image so image can be made smaller. Larger images are often easier to work with because they hold more pixels.

    Args:
        image (object): image to resize
        image_heigt (int): desired height of image

    Returns:
        object: image with given height
    """

    # cut to estimated region
    # if estimated_display_region is not None:
    #     image = image[estimated_display_region[0]:estimated_display_region[2],
    #                   estimated_display_region[1]:estimated_display_region[3]].copy()

    # resize image
    image_resized = imutils.resize(image=image, height=image_height)

    # write out image
    cv2.imwrite("steps/step_1_resize.jpg", image_resized)

    return image_resized


def gray_image(image):
    """
    Converts color image to greyscale

    Args:
        image (object): image to convert to greyscale
        image_heigt (int): desired height of image

    Returns:
        object: grey image
    """

    # make image gray
    image_grayed = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    # write out image
    cv2.imwrite("steps/step_2_grayed.jpg", image_grayed)

    return image_grayed


def blur_image(image, kernel_size_width=5, kernel_size_height=5):
    """
    Blurs image. Increase kernel sizes to make image more blurry. Kernel sizes define a rectange of pixels of which the average color is taken

    Args:
        image (object): image to blur
        kernel_size_width (int): width of rectange in pixels
        kernel_size_height (int): height of rectange in pixels

    Returns:
        object: blurred image
    """

    # make image blurry
    image_grayed = cv2.GaussianBlur(
        src=image, ksize=(kernel_size_width, kernel_size_height), sigmaX=0, sigmaY=0)

    # write out image
    cv2.imwrite("steps/step_3_blurred.jpg", image_grayed)

    return image_grayed


def get_image_edges(image, threshold_1=50, threshold_2=200, edges=255):
    """
    Detect edges in image.

    Args:
        image (object): image to detect edges in
        threshold_1 (int):
        threshold_2 (int):
        edges (int):

    Returns:
        object: image of edges
    """

    # get edges
    image_edges = cv2.Canny(
        image=image, threshold1=threshold_1, threshold2=threshold_2, edges=edges, L2gradient=True)

    # write out image
    cv2.imwrite("steps/step_4_edges.jpg", image_edges)

    return image_edges


def extract_display(image_resized, image_grayed, image_edged, accuracy=0.02, width_display=[650, 850], height_display=[250, 350]):
    """
    Extract the display from the image

    Args:
        image_resized (object): image with specific height
        image_grayed (object): image in greyscale
        image_edged (object): edges of image
        accuracy (dbl): parameter specifying the approximation accuracy
        width_display (int[min, max]): minimum and maximum width of display
        height_display (int[min, max]): minimum and maximum height of display

    Returns:
        object: image with rectange around display, image of display
    """

    # get contours in edge map
    contours = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    countoursOfDisplay = None

    # make copy of image to draw annotations on
    image_annotated = image_resized.copy()

    # loop over the contours
    for c in contours:

        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(
            curve=c, epsilon=accuracy*peri, closed=True)

        # draw a boudnig rectange around the vertices
        (x, y, w, h) = cv2.boundingRect(approx)

        # if the contour has four vertices, then we have found the display
        # display must be within min and max sizes
        if len(approx) == 4 and w >= width_display[0] and w <= width_display[1] and h >= height_display[0] and h <= height_display[1]:

            countoursOfDisplay = approx

            # draw rectangle around the display
            cv2.rectangle(image_annotated, (x, y),
                          (x + w, y + h), (0, 255, 0), 5)

            # draw contour points
            cv2.drawContours(image=image_annotated, contours=approx,
                             contourIdx=-1, color=(0, 0, 255), thickness=5)

            # annotate rectangle with dimensions
            cv2.putText(img=image_annotated, text=('vertices: '+str(len(approx))+' | dim: '+str(w)+' x '+str(h)), org=(x, y-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 0))

            break
        else:
            # if conditions are not met, just annotate the candidate

            # draw rectangle around the contour
            cv2.rectangle(image_annotated, (x, y),
                          (x + w, y + h), (255, 0, 0), 3)

            # draw contour points
            cv2.drawContours(image=image_annotated, contours=approx,
                             contourIdx=-1, color=(2505, 0, 255), thickness=5)

            # annotate rectangle with dimensions
            cv2.putText(img=image_annotated, text=('vertices: '+str(len(approx))+' | dim: '+str(w)+' x '+str(h)), org=(x, y-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 0))

    # draw rectangle around the display
    cv2.imwrite("steps/step_5_display_full.jpg", image_annotated)

    # extract the thermostat display, apply a perspective transform to it
    image_display_grayed = four_point_transform(
        image_grayed, countoursOfDisplay.reshape(4, 2))
    image_display = four_point_transform(
        image_resized, countoursOfDisplay.reshape(4, 2))

    # resize images
    image_display_grayed = imutils.resize(
        image=image_display_grayed, height=500)
    image_display = imutils.resize(
        image=image_display, height=500)

    # write out image
    cv2.imwrite("steps/step_5_display.jpg", image_display_grayed)

    return image_display_grayed, image_display


def increase_contrast(image, threshold=2, size=(6, 6)):
    """
    Increase contrast of image

    Args:
        threshold (int): bigger number is results in higher contrast
        size (int, int): bigger number is results in higher contrast

    Returns:
        object: image with increased contrast
    """
    clahe = cv2.createCLAHE(clipLimit=threshold, tileGridSize=size)
    image_contrast = clahe.apply(image)

    # write out image
    cv2.imwrite("steps/step_6_contrast.jpg",
                image_contrast)

    return image_contrast


def make_pixels_black_or_white(image, threshold_1=0, threshold_2=0):
    """
    Make image binary (black or white)

    Args:
        threshold_1 (int): higher number increases black/white
        threshold_2 (int): higher number increases black/white

    Returns:
        object: binary image
    """

    # apply thresholding
    binary_image = cv2.adaptiveThreshold(src=image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         thresholdType=cv2.THRESH_BINARY_INV, blockSize=threshold_1, C=threshold_2)

    # write out image
    cv2.imwrite("steps/step_6_black_or_white_pixels.jpg",
                binary_image)

    return binary_image


def remove_noise_from_image(binary_image, shape_opening=cv2.MORPH_ELLIPSE, shape_closing=cv2.MORPH_ELLIPSE, size_opening=(1, 5), size_closing=(1, 5)):
    """
    Make image binary (black or white)

    Args:
        image (object): a greyscale image
        shape_opening:
        shape_closing:
        size_opening:
        size_closing:

    Returns:
        object: sharp image
    """
    image = binary_image.copy()

    # define structuring element for opening
    kernel_opening = cv2.getStructuringElement(
        shape=shape_opening, ksize=size_opening)

    # remove noise
    image = cv2.morphologyEx(
        src=image, op=cv2.MORPH_OPEN, kernel=kernel_opening)

    # write out image
    cv2.imwrite("steps/step_7_1_opening.jpg", image)

    # define structuring element for closing
    kernel_closing = cv2.getStructuringElement(
        shape=shape_closing, ksize=size_closing)

    # close small holes inside the foreground objects, or small black points on the object
    image = cv2.morphologyEx(
        src=image, op=cv2.MORPH_CLOSE, kernel=kernel_closing)

    # write out image
    cv2.imwrite("steps/step_7_2_closing.jpg", image)

    return image


def find_digit_areas(binary_display_without_noise, display, min_width_digit_area=15, max_width_digit_area=30, min_height_digit_area=30, max_height_digit_area=40):
    """
    Find areas of digits

    Args:
        binary_display_without_noise (object): image of display without noise
        display (object): image of display
        min_width_digit_area:
        max_width_digit_area:
        min_height_digit_area:
        max_height_digit_area:

    Returns:
        list: contours of digit areas
    """

    # make a copy of the display where annotations can be added to
    image_annotated = display.copy()

    # find contours in the thresholded image, then initialize the digit contours lists
    contours = cv2.findContours(image=binary_display_without_noise.copy(
    ), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contoursOfDigits = []

    # loop over the digit area candidates
    for c in contours:

        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # if the contour is sufficiently large, it must be a digit
        if (w >= min_width_digit_area and w <= max_width_digit_area) and (h >= min_height_digit_area and h <= max_height_digit_area):

            # the contour is a digit
            contoursOfDigits.append(c)

            # draw contour points
            cv2.drawContours(image=image_annotated, contours=c,
                             contourIdx=-1, color=(0, 0, 255), thickness=5)

            # draw a thick blue box around the contours where a digit is
            cv2.rectangle(image_annotated, (x, y), (x + w, y + h),
                          (255, 0, 0), 3)

            # annotate rectangle with dimensions
            cv2.putText(img=image_annotated, text=('dim: '+str(w)+' x '+str(h)), org=(x, y-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255))

        else:

            # draw a blue box around digit area candidate
            cv2.rectangle(image_annotated, (x, y),
                          (x + w, y + h), (255, 0, 0), 1)

            # draw contour points
            cv2.drawContours(image=image_annotated, contours=c,
                             contourIdx=-1, color=(0, 0, 255), thickness=5)

            # annotate rectangle with dimensions
            cv2.putText(img=image_annotated, text=('dim: '+str(w)+' x '+str(h)), org=(x, y-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255))

    cv2.imwrite("steps/step_8_digit_areas.jpg", image_annotated)

    return contoursOfDigits


def read_digits(binary_display_without_noise, display, contours_of_digits, alpha=0.25, beta=0.15, gamma=0.05, min_fill_area=0.5, min_width_digit_1=20, max_width_digit_1=60, bottom_right_segment_offset_to_left=5):
    """
    Read digits on display

    Args:
        binary_display_without_noise (object): image of display without noise
        display (object): image of display
        contours_of_digits:
        alpha: width of vertical segment / width of ROI
        beta: height of horizontal segment / height of ROI
        gamma: half the height of the horizontal center segment / height of ROI
        min_fill_area: minium % of segment that must be filled in order to be considered as on
        min_width_digit_1:
        max_width_digit_1:
        bottom_right_segment_offset_to_left:

    Returns:
        array of integers
    """
    # w = width of ROI
    # dw = width of vertical segment
    # h = height of ROI
    # dh = height of horizontal segment
    # dHC = half the height of the horizontal center segment
    # alpha = dw / w
    # beta = dh / h
    # gamma = dHC / h

    # make a copy of the image to annotate
    display_annotated = display.copy()

    # sort the contours from left-to-right
    contours_of_digits = contours.sort_contours(
        contours_of_digits, method="left-to-right")[0]

    # initialize the actual digits themselves
    digits = []

    # Initialize counter
    digit_count = 0

    # loop over each of the digits
    for c in contours_of_digits:

        # extract the digit ROI (region of interest)
        (x, y, w, h) = cv2.boundingRect(c)
        roi = binary_display_without_noise[y:y + h, x:x + w]
        roi_color = display[y:y + h, x:x + w]

        # Increase counter
        digit_count = digit_count + 1

        # compute the width and height of each of the 7 segments we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * alpha), int(roiH * beta))
        dHC = int(roiH * gamma)

        # ROI is greater than a certain width, than define 7 segments. If ROI is less than certain width, then define 2 segments (can be 1)
        if (w >= min_width_digit_1 and w <= max_width_digit_1):

            # 2 segments
            DIGITS_LOOKUP = {
                (1, 1): 1
            }

            segments = [
                ((0, 0), (w, h // 2 - dHC)),  # top
                # bottom
                ((0, h // 2 + dHC),
                 (w, h))
            ]

            on = [0] * len(segments)

        else:

            # 7 segments
            DIGITS_LOOKUP = {
                (1, 1, 1, 0, 1, 1, 1): 0,
                (0, 0, 1, 0, 0, 1, 0): 1,
                (1, 0, 1, 1, 1, 0, 1): 2,
                (1, 0, 1, 1, 0, 1, 1): 3,
                (0, 1, 1, 1, 0, 1, 0): 4,
                (1, 1, 0, 1, 0, 1, 1): 5,
                (1, 1, 0, 1, 1, 1, 1): 6,
                (1, 0, 1, 0, 0, 1, 0): 7,
                (1, 1, 1, 1, 1, 1, 1): 8,
                (1, 1, 1, 1, 0, 1, 1): 9
            }

            segments = [
                ((0, 0), (w, dH)),  # top
                ((0, 0), (dW, h // 2)),  # top-left
                ((w - dW, 0), (w, h // 2)),  # top-right
                ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
                ((0, h // 2), (dW, h)),  # bottom-left
                # bottom-right
                ((w - dW - bottom_right_segment_offset_to_left, h // 2),
                 (w - bottom_right_segment_offset_to_left, h)),
                ((0, h - dH), (w, h))  # bottom
            ]

            on = [0] * len(segments)

        # sort the contours from left-to-right, then initialize the actual digits themselves loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of thresholded pixels in the segment, and then compute the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)

            # if the total number of non-zero pixels is greater than 50% of the area, mark the segment as "on"
            if total / float(area) > min_fill_area:
                on[i] = 1
                # Draw green rectangle around segment if filled
                cv2.rectangle(roi_color, (xA, yA),
                              (xB, yB), (0, 255, 0), 3)
            else:
                on[i] = 0
                # draw a red rectangle around segment if not filled
                cv2.rectangle(roi_color, (xA, yA),
                              (xB, yB), (0, 0, 255), 1)

        # Write out image of digit ROI
        cv2.imwrite("steps/step_9_RIO_" +
                    str(digit_count) + ".jpg", roi_color)

        # lookup the digit and draw it on the image
        digit = DIGITS_LOOKUP[tuple(on)]
        digits.append(digit)
        cv2.putText(img=display_annotated, text=str(digit), org=(x+(w//4), y+(h)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=7, color=(0, 0, 255), thickness=20)

    # Make image bigger
    # display = resize_image(display, image_height=300)
    display_annotated = imutils.resize(
        image=display_annotated, height=300)

    # Write out image
    cv2.imwrite("steps/result.jpg", display_annotated)

    return digits


def convert_to_number(array):
    """
    converts arrays of integer to a single number.
    e.g. [1, 2, 3] becomes 123
    """
    # convert digits array to single number
    result = 0

    for d in list(reversed(array)):

        multiplicator = 10 ** (list(reversed(array)).index(d))
        result += d * multiplicator

    return(result)
