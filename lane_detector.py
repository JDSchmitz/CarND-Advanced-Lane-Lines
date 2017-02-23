import cv2
import numpy as np

class LaneDetector():
    ''' Lane Detector - performs three key functions:
        1a) detects lanes in given image using sliding window algorithm
        1b) detects lanes around previously found lanes
        2) calculates lane curvature
        3) displays lane information

        Uses code from Udacity lessons
    '''
    def __init__(self):
        self.left_fit  = None
        self.right_fit = None
        self.leftx     = None 
        self.rightx    = None
        self.car_position = None
        print('(init: LaneDetector)')

    def window_fit(self, img):
        '''  Apply polynomial fit to the given image, returning fit for left/right lanes
            Called when one frame of image has previously found left_fit/right_fit. 
            This method attempts to find lane fits in the vicinity of previous fits
            :param img -- input image with lane lines
            :return left_fit, right_fit
        '''

        if self.left_fit is None or self.right_fit is None:
            return self.sliding_window_fit(img)
        
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero  = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds  = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & 
                        (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & 
                        (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        self.leftx  = nonzerox[left_lane_inds]
        lefty       = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        righty      = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit  = np.polyfit(lefty,  self.leftx, 2)
        self.right_fit = np.polyfit(righty, self.rightx, 2)

        return self.left_fit, self.right_fit

    def sliding_window_fit(self, img):
        ''' Apply sliding windows search to the given image to find polynomial to find lane lines
            Code based largely on Udacity lessons
        :param img - given image
        :return left_fit, right_fit - polynomials fitting the left/right lane lines
        '''
        y_half = int(img.shape[0]/2)
        # take histogram of bottom half of img
        histogram = np.sum(img[y_half:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint    = np.int(histogram.shape[0]/2)
        leftx_base  = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero     = img.nonzero()
        nonzeroy    = np.array(nonzero[0])
        nonzerox    = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current   = leftx_base
        rightx_current  = rightx_base

        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds  = []
        right_lane_inds = []

        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low       = img.shape[0] - (window+1) * window_height
            win_y_high      = img.shape[0] - window * window_height
            win_xleft_low   = leftx_current - margin
            win_xleft_high  = leftx_current + margin
            win_xright_low  = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds  = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int( np.mean(nonzerox[good_left_inds]) )
            if len(good_right_inds) > minpix:        
                rightx_current = np.int( np.mean(nonzerox[good_right_inds]) )

        
        # Concatenate the arrays of indices
        left_lane_inds  = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        self.leftx  = nonzerox[left_lane_inds]
        lefty       = nonzeroy[left_lane_inds] 
        
        righty      = nonzeroy[right_lane_inds]
        self.rightx = nonzerox[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit  = np.polyfit(lefty, self.leftx, 2)
        self.right_fit = np.polyfit(righty,self.rightx, 2)

        return self.left_fit, self.right_fit

    def find_lane_curvature(self, img):
        ''' Find lane curvature for the given img
            :param img - the input image
            :return lane curvature
        '''
        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, 719, num=720)    # to cover same y-range as image
        quadratic_coeff = 3e-4                  # arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])
        rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                        for y in ploty])

        leftx  = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


        # Fit a second order polynomial to pixel positions in each fake lane line
        # left_fit  = np.polyfit(ploty, leftx, 2)
        # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        # right_fit = np.polyfit(ploty, rightx, 2)
        # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        # left_curverad  = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        # right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        # print(left_curverad, right_curverad)
        # Example values: 1926.74 1908.48

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr  = np.polyfit(ploty * ym_per_pix, leftx  * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval*ym_per_pix + \
                        left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + \
                        right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m

        lx = self.left_fit[0] * (img.shape[0] - 1)**2 + \
            self.left_fit[1] * (img.shape[0] - 1) + \
            self.left_fit[2]
        rx = self.right_fit[0] * (img.shape[0] - 1)**2 + \
            self.right_fit[1] * (img.shape[0] - 1) + \
            self.right_fit[2]
        # calc car's position in the lane w.r.to center
        position = ((img.shape[1] / 2) - ((lx + rx)/2)) * xm_per_pix

        # calc mean curvature
        mean_curverad = (left_curverad + right_curverad) / 2
        # save the car's position
        self.car_position = position.round(2)
        return mean_curverad

    def draw_polygon(self, img, left_fit, right_fit, M_inverse):
        ''' Draw shaded polygon on the lane between left_fit and right_fit
            :param img - undistorted image, on which to draw the lane polygon
            :param left_fit - left lane values (x)
            :param right_fit - right lane values (x)
            :param M_inverse - matrix for inverse transform warping
            :return - img - the modified image with polygon
        ''' 
        fity       = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx  = left_fit[0] * fity ** 2  + left_fit[1] * fity  + left_fit[2]
        right_fitx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

        color_warp = np.zeros_like(img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left  = np.array( [np.transpose(np.vstack([left_fitx, fity]))] )
        pts_right = np.array( [np.flipud(np.transpose(np.vstack([right_fitx, fity])))] )
        pts = np.hstack((pts_left, pts_right))
        pts = np.array(pts, dtype=np.int32)

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, M_inverse, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        return result

    def display_dashboard(self, img, lane_curve):
        ''' Display a dashboard on the image, with info on
            Lane curve (avg)
            :param img - image with lane lines
            :param lane_curve - the avg lane curvature
            :param position
            :return modified img
        '''
        COLOR_LIGHTBLUE = (172,227,239) 
        COLOR_GOLD = (255, 215, 0)

        if self.car_position > 0:
            msg = '{}m right of center'.format(self.car_position)
        else:
            msg = '{}m left of center'.format(np.abs(self.car_position))

        cv2.putText(img, 'Lane curve radius: {}m'.format(lane_curve.round()), 
                    (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    color=COLOR_GOLD, thickness=2)
        cv2.putText(img, 'Car is {}'.format(msg), 
                    (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=COLOR_GOLD, thickness=2)
        cv2.rectangle(img, (5, 10), (480, 100), color=COLOR_GOLD, thickness=2)
        return img



