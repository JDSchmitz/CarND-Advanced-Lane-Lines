import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import plot_images

THRESH_SOB_X = (20, 100)
THRESH_SOB_Y = (20, 100)
THRESH_GRAD = (50, 255)
THRESH_DIR  = (0.7, 1.2)
THRESH_S_CHAN = (90, 255)   # S channel in HLS

YELLOW_MIN = [15, 100, 120]
YELLOW_MAX = [80, 255, 255]

WHITE_MIN = [0, 0, 200]
WHITE_MAX = [255, 30, 255]

def gaussian_blur(image, ksize):
    ''' Apply Gaussian blur of kernel size ksize to given image, and return image
    ''' 
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def canny(image, low, high):
    ''' Applies Canny edge detection to given image with low/high threshold
    '''
    return cv2.Canny(image, low, high)

def median_blur(image, ksize):
    ''' Apply median blur of kernel size ksize
    '''
    return cv2.medianBlur(image, ksize)

def absolute_sobel(gray_img, orient='x', sobel_kernel=3):
    '''  Returns absolute of the given gray image
        :param gray_img: - the input GRAY image
        :param orient: 'x' or 'y'
        :param sobel_kernel: kernel size (must be odd)
        :return absolute sobel value
    '''

    if orient == 'x':
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise ValueError('orientation must be "x" or "y" not "%s"' % orient)

    return np.absolute(sobel)

def gradient_magnitude(sobel_x, sobel_y):
    ''' Return magnitude of gradient
    '''
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y **2)

    scalefact = np.max(magnitude)/255
    magnitude = (magnitude/scalefact).astype(np.uint8)

    # TODO: rescale???
    return magnitude # .astype(np.uint16)   # magnitude.astype(np.uint16) ??

def gradient_direction(sobel_x, sobel_y):
    '''  Return direction of gradient
    ''' 
    with np.errstate(divide='ignore', invalid='ignore'):
        direction = np.absolute(np.arctan2(sobel_y, sobel_x))
        ## TODO: check for NaN??
        direction[np.isnan(direction)] = np.pi / 2   # clip isNan values to PI/2

    return direction.astype(np.float32)

def extract_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))


def process_image(image, y_cutoff=0):
    '''
    '''

    # 1 - create window
    img_window = image[y_cutoff:, :, :]  # ignore top y_cutoff pixels

    # 2 - convert to gray
    gray = cv2.cvtColor(img_window, cv2.COLOR_RGB2GRAY)

    # 3 - convert to HLS
    hls = cv2.cvtColor(img_window, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # 4 - get Sobel X, Sobel Y abs values (no thresh)
    sob_x = absolute_sobel(gray, orient='x', sobel_kernel=5)
    sob_y = absolute_sobel(gray, orient='y', sobel_kernel=5)

    # 5 - get gradient - direction and magnitude
    grad_magnitude = gradient_magnitude(sob_x, sob_y)  # (0, 255)
    grad_direction = gradient_direction(sob_x, sob_y)  # float
    
    # TODO: scale factors??
    sob_x = np.uint8(255 * sob_x / np.max(sob_x))  # (0, 255)
    sob_y = np.uint8(255 * sob_y / np.max(sob_y))  # (0, 255)

    # 6 - other channels? TODO: 
    yellow = extract_yellow(img_window)

    # 7 - create blank mask
    combined_mask = np.zeros(image.shape[:-1], dtype=np.uint8)
    
    # 8 - apply thresholds 
    # TODO: define thresholds for Magnitude, X, Y, Direction
    combined_mask[y_cutoff:, :][((sob_x >= THRESH_SOB_X[0]) & (sob_x <= THRESH_SOB_X[1]) & 
                                (sob_y >= THRESH_SOB_Y[0])  & (sob_y <= THRESH_SOB_Y[1])) |
                                ((grad_magnitude >= THRESH_GRAD[0]) & (grad_magnitude <= THRESH_GRAD[1]) & 
                                 (grad_direction >= THRESH_DIR[0]) & (grad_direction <= THRESH_DIR[1])) |
                                 # TODO: add any other filters?? Yellow, White??
                                ((s_channel > THRESH_S_CHAN[0]) & (s_channel < THRESH_S_CHAN[1])) |
                                (yellow == 255)
                                ] = 1

    # 9 - binary blur TODO:
    combined_mask = median_blur(combined_mask, 7)

    return combined_mask

def direction_threshold(sobel_x, sobel_y):
    ''' Return binary image with Direction threshold applied
    '''
    sobel_x = np.abs(sobel_x)
    sobel_y = np.abs(sobel_y)
    direction = np.arctan2(sobel_y, sobel_x)
    result = np.zeros_like(direction)
    result[(direction >= THRESH_DIR[0]) & 
           (direction <= THRESH_DIR[1])] = 1
    return result

def magnitude_threshold(sobel_x, sobel_y):
    ''' Return binary image with Magnitude threshold applied
    ''' 
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    scale = np.max(magnitude) / 255. 
    magnitude = (magnitude / scale).astype(np.uint8)

    result = np.zeros_like(magnitude)
    result[(magnitude >= THRESH_GRAD[0]) & 
           (magnitude <= THRESH_GRAD[1])] = 1
    return result

def color_threshold(color_img):
    ''' Apply color thresholds and return binary image
    ''' 
    img = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)

    # 1 - extract white
    white_img = cv2.inRange(img,
                            np.array(WHITE_MIN, np.uint8),
                            np.array(WHITE_MAX, np.uint8))

    # 2 - extract yellow
    yellow_img = cv2.inRange(img, 
                            np.array(YELLOW_MIN, np.uint8),
                            np.array(YELLOW_MAX, np.uint8))
    # 3 - combine
    result = np.zeros_like(img[:, :, 0])
    result[((white_img !=0) | (yellow_img !=0))] = 1

    orig_img = img
    orig_img[((white_img == 0) & (yellow_img == 0))] = 0
    return result 


def apply_thresholds(img):
    '''  Apply a series of thresholds to given image
        :return image with thresholds applied
    '''

    # 1 - convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2 - get sobel X and y 
    #sob_x = absolute_sobel(gray, orient='x', sobel_kernel=15)
    #sob_y = absolute_sobel(gray, orient='y', sobel_kernel=15)

    sob_x = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 1, 0, ksize=9)
    sob_y = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 0, 1, ksize=9)

    # 3 - direction threshold
    direction_binary = direction_threshold(sob_x, sob_y)
    # 4 - magnitude threshold
    magnitude_binary = magnitude_threshold(sob_x, sob_y)
    # 5 - color threshold
    color_binary = color_threshold(img)

    # 6 - combine all binaries
    combined = np.zeros_like(direction_binary)
    combined[((color_binary == 1) & 
             ((magnitude_binary == 1) | (direction_binary == 1)))] = 1
    
    return combined



if __name__ == '__main__':

    from scipy.misc import imread

    original_images, right_images = [], []

    DIR_TEST = 'test_images/'

    test_images = [
        # DIR_TEST + 'straight_lines2.jpg',
        DIR_TEST + 'test5.jpg',
        # DIR_TEST + 'test6.jpg',
        DIR_TEST + 'test4.jpg',
    ]

    for imgfile in test_images:
        print('testing file:', imgfile)
        orig_img    = imread(imgfile)

        # masked_img = process_image(orig_img, 400)
        img_threshold = apply_thresholds(orig_img)
        original_images.append(orig_img)
        right_images.append(img_threshold)

    plot_images(original_images, right_images, 
                        save_filename='output_images/thresh6b_k7.png', 
                        left_title='Original', 
                        right_title='Thresholded')

