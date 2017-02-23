import cv2
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc

# local
from camera_calibrator import CameraCalibrator
from threshold_detector import *
from perspective import PerspectiveTransformer
from lane_detector import LaneDetector
from utils import *

# create instances
camera_calibrator = CameraCalibrator()
Transformer = PerspectiveTransformer()
detector = LaneDetector()


def process_frame(frame):
    ''' Main function that processes each frame of the video
    ''' 
    
    fig = plt.figure(figsize=(10,8))
    i = 1
    # i = show_image(fig, frame, i, 'Original', 'gray' )

    # 1 - Undistort the base input frame
    undistorted = camera_calibrator.undistort(frame)
    # i = show_image(fig, undistorted, i, 'Undistorted', 'gray')

    # 2 - Apply thresholds to get lane pixels
    img = apply_thresholds(undistorted)
    # i = show_image(fig, img, i, 'Threshold', 'gray')

    # 3 - Warp Perspective to make a birds' eye view image
    birdseye_img = Transformer.forward_transform(img)
    # i = show_image(fig, birdseye_img, i, 'BirdsEye', 'gray')
    #plt.show()

    # 4a - Find polynomial fit for lanes
    left_fit, right_fit = detector.window_fit(birdseye_img)

    # 4b - Draw polygon on the image, after warping perspective 
    img_poly = detector.draw_polygon(undistorted, left_fit, right_fit, Transformer.M_inv)
    # i = show_image(fig, img_poly, i, 'Polygon')
    # plt.show()

    # 5 - Find lane curvature
    lane_curve = detector.find_lane_curvature(img_poly)

    # 6 - Display lane info dashboard
    img_poly = detector.display_dashboard(img_poly, lane_curve)

    return img_poly


def main():
    # video = 'project_video'
    # video = 'challenge_video'
    video = 'harder_challenge_video'
    output = '{}_out-full.mp4'.format(video)
    clip   = VideoFileClip('{}.mp4'.format(video)).subclip(0, 15)
    print('Processing video..')
    w_clip = clip.fl_image(process_frame)  # expects color images
    w_clip.write_videofile(output, audio=False)

if __name__ == '__main__':
    main()

