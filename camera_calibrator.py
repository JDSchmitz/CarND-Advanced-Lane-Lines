import numpy as np
import cv2
import os, glob, sys
import pickle
from tqdm import tqdm

from scipy.misc import imread, imresize

from utils import plot_images

NX, NY = 9, 6  # number of inside corners for (x,y) on chessboard

DIR_CHESSBOARD     = 'camera_cal/'
IMAGES_CALIBRATION = 'calibration*.jpg'
CALIBRATION_FILE   = DIR_CHESSBOARD + 'camera_calibration_params.p'
IMG_SIZE = (720, 1280, 3)


class CameraCalibrator:
    ''' Camera Calibrator -- calibrates camera using chessboard pattern images 

        Uses code from Udacity lessons
    '''

    def __init__(self):

        if not os.path.isfile(CALIBRATION_FILE): # calibrate camera as needed
            self.do_calibrate_camera()

        # load calibration params
        self._load_calibration()
        print('(init: CameraCalibrator)')

    def do_calibrate_camera(self):
        ''' Calibrate camera using given chessboard patterns
            Save calibration params

            Uses code from Udacity lessons
        '''
        print('Doing camera calibration...')
        objp        = np.zeros((NX * NY, 3), np.float32)
        objp[:,:2]  = np.mgrid[0:NX, 0:NY].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        # list of calibration images
        images = glob.glob(DIR_CHESSBOARD + IMAGES_CALIBRATION)
        print('Found calibration images:', len(images))

        processed = 0
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(tqdm(images, desc='calibrating images...')):
            img  = imread(fname)   # RGB format via scipy
            if img.shape[0] != IMG_SIZE[0] or img.shape[1] != IMG_SIZE[1]:
                img = imresize(img, IMG_SIZE) 
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                processed += 1

        print("Calibrated {}/{} images".format(processed, len(images)))

        # save calibration info
        self._save_calibration()
    
    def undistort(self, image):
        ''' Use loaded calibration params to undistort image
            :param image - given image to undistor
            :return - undistorted image
        '''
        return cv2.undistort(image, self.camMatrix, self.distCoeffs, None, self.camMatrix)

    def _save_calibration(self):
        ''' Save calibration parameters
        ''' 
        # Do camera calibration given object points and image points
        retval, camMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, 
                                            self.imgpoints, 
                                            IMG_SIZE[:-1], None, None)

        calibration_params = {
            'objpoints':    self.objpoints,
            'imgpoints':    self.imgpoints,
            'camMatrix':    camMatrix,
            'distCoeffs':   distCoeffs,
            'rvecs':        rvecs,
            'tvecs':        tvecs
        }

        with open(CALIBRATION_FILE, 'wb') as FF:
            pickle.dump(calibration_params, file=FF)
        print('camera calibration saved:', CALIBRATION_FILE)

    def _load_calibration(self):
        ''' Load existing camera calibration
        '''

        print('Loading calibration file:', CALIBRATION_FILE)
        with open(CALIBRATION_FILE, 'rb') as FIN:
            calibration_params = pickle.load(FIN)

        self.objpoints = calibration_params['objpoints']
        self.imgpoints = calibration_params['imgpoints']
        self.camMatrix = calibration_params['camMatrix']    # camera matrix
        self.distCoeffs = calibration_params['distCoeffs']  # distortion coeff

        print('loaded')



if __name__ == '__main__':
    
    calibrator = CameraCalibrator()

    original_images, undistorted_images = [], []

    DIR_TEST = 'test_images/'

    test_images = [
        # DIR_CHESSBOARD + 'calibration2.jpg',
        DIR_TEST + 'test4.jpg',
        DIR_TEST + 'test5.jpg'
    ]

    for imgfile in test_images:
        print('testing file:', imgfile)
        orig_img    = imread(imgfile)
        undistorted = calibrator.undistort(orig_img)
        original_images.append(orig_img)
        undistorted_images.append(undistorted)

    plot_images(original_images, undistorted_images, 
                        save_filename='output_images/cam-road-calib1.png', 
                        left_title='Original', 
                        right_title='Undistorted')
