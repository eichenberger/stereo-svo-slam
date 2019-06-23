import math
import numpy as np

from corner_detector import CornerDetector

class DepthCalculator:
    def __init__(self, baseline, fx, fy, cx, cy, window_size, search_x, search_y, margin):
        self.baseline = baseline
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.window_size = window_size
        self.search_x = search_x
        self.search_y = search_y
        self.detector = CornerDetector(margin)

    def match(self, roi, templ):
        x = 0
        y = 0
        min_err = math.inf
        x_match = 0
        y_match = 0
        for x in range(0, roi.shape[1] - templ.shape[1]):
            for y in range(0, roi.shape[0] - templ.shape[0]):
                # 0.0 to make sure we can have negative numbers. Image is uint
                diff = 0.0 + templ - roi[y:templ.shape[0]+y, x:templ.shape[1]+x]
                err = np.sum(np.abs(diff))
                if err < min_err:
                    min_err = err
                    x_match = x
                    y_match = y

        return x_match, y_match, min_err

    def calculate_depth(self, left, right, split_count):
        half_window_size = int(self.window_size/2)
        search_x = self.search_x
        search_y = self.search_y

        keypoints = self.detector.detect_keypoints(left, split_count)
        keypoints2d = np.zeros((2, len(keypoints)))
        disparity = np.zeros((1, len(keypoints)))
        err = [0]*len(keypoints)

        for i, keypoint in enumerate(keypoints):
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            x1 = x-half_window_size
            x2 = x+half_window_size
            y1 = y-half_window_size
            y2 = y+half_window_size

            if x1 < 0 or x2 >= left.shape[1] or y1 < search_y or y2 >= (left.shape[0]-search_y):
                continue

            templ = left[y1:y2,x1:x2]
            roi = right[y1-search_y:y2+search_y,x1:x2+search_x]

            x_match, y_match, err[i] = self.match(roi, templ)

            # The location of the minimum is the left point of the window
            # therefore the middle point is at left point + half_window_size
            disparity[0, i] = (x_match + half_window_size)
            keypoints2d[0, i] = x + x_match + half_window_size
            keypoints2d[1, i] = y + y_match + half_window_size

        keypoints2d = np.asarray(keypoints2d)
        keypoints3d = np.empty((3, keypoints2d.shape[1]))
        keypoints3d [2,:] = self.baseline/disparity
        keypoints3d [0,:] = ((keypoints2d[0,:] - self.cx)/self.fx)*keypoints3d [2,:]
        keypoints3d [1,:] = ((keypoints2d[1,:] - self.cy)/self.fy)*keypoints3d [2,:]

        return keypoints2d, keypoints3d, err



