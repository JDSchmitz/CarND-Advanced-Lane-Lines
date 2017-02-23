import cv2
import numpy as np

OFFSET = 250

SRC_POINTS = np.float32([
    (130, 700),   # Bottom left
    (540, 465),   # Top L  
    (740, 465),   # Top R 
    (1200, 700)   # Bottom R
])

DEST_POINTS = np.float32([
    (SRC_POINTS[0][0] + OFFSET, 720),
    (SRC_POINTS[0][0] + OFFSET, 0),
    (SRC_POINTS[-1][0] - OFFSET, 0),
    (SRC_POINTS[-1][0] - OFFSET, 720)
])

class PerspectiveTransformer:
    ''' Transforms perspective of image - from roadview to birds-eye view and vice versa
        Saves matrices required for warping perspective.
    '''

    def __init__(self):
        '''
        ''' 
        # forward transform matrix
        self.M      = cv2.getPerspectiveTransform(SRC_POINTS, DEST_POINTS)
        # inverse transform matrix
        self.M_inv  = cv2.getPerspectiveTransform(DEST_POINTS, SRC_POINTS)
        print('(init: PerspectiveTransformer)')

    
    def forward_transform(self, img):
        '''  Apply transform on img to produce a bird's eye view
            :param img -- input image
            :return: bird's eye view (transformed image)
        '''
        return cv2.warpPerspective(img, self.M, 
                        (img.shape[1], img.shape[0]),  # (X,Y)
                        flags=cv2.INTER_LINEAR)
    
    def inverse_transform(self, img):
        '''  Inverse warping on given input image
            :param img -- the bird's eye view image
            :return: the regular (perspective'd) image (transformed via M_inv)
        '''
        return cv2.warpPerspective(img, self.M_inv, 
                        (img.shape[1], img.shape[0]),  # (X,Y)
                        flags=cv2.INTER_LINEAR)


if __name__ == '__main__':

    from scipy.misc import imresize, imread
    from utils import *

    import os
    import glob 

    transformer = PerspectiveTransformer()

    left_images, right_images = [], []

    DIR_CHESSBOARD     = 'camera_cal/'
    DIR_TEST           = 'test_images/'
    test_images = [
        DIR_TEST + 'test4.jpg',
        DIR_TEST + 'test5.jpg',
        DIR_TEST + 'test6.jpg'
    ] 

    # test_images = glob.glob(DIR_TEST + '/*.jpg')

    p1 = tuple(SRC_POINTS[0])
    p2 = tuple(SRC_POINTS[1])
    p3 = tuple(SRC_POINTS[2])
    p4 = tuple(SRC_POINTS[3])

    counter = 0
    for img_file in test_images:
        print('reading:', img_file)
        img = imread(img_file)
        counter += 1

        color=(0,255,255)
        thickness=2

        cv2.line(img, p1, p2, color, thickness)
        cv2.line(img, p2, p3, color, thickness)
        cv2.line(img, p3, p4, color, thickness)
        cv2.line(img, p4, p1, color, thickness)
       
        birdseye = transformer.forward_transform(img)

        left_images.append(img)
        right_images.append(birdseye)

        plot_image(birdseye, title=None, save_file='out{}.png'.format(counter))

    plot_images(left_images, right_images, save_filename='output_images/tx-persp1.png', left_title='Original', right_title='Birds Eye')

