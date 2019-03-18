import math
import cv2
import numpy as np

from corner_detector import CornerDetector
from keyframe import KeyFrame

class DepthCalculator:
    def __init__(self, baseline, fx, fy, cx, cy):
        self.baseline = baseline
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.detector = CornerDetector()

    def calculate_depth(self, left, right):
        window_size = 7
        search_x = 40
        search_y = 3
        disparity = []
        #dy_list = []
        diff_list = []
        keypoints2d = []

        keypoints = self.detector.detect_keypoints(left)

        key = 0
        for index, keypoint in enumerate(keypoints):
            diff_max = math.inf
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            x1 = x-window_size
            x2 = x+window_size
            y1 = y-window_size
            y2 = y+window_size

            if x1 < 0 or x2 >= left.shape[1] or y1 < search_y or y2 >= (left.shape[0]-search_y):
                continue

            templ = left[y1:y2,x1:x2]
            roi = right[y1-search_y:y2+search_y,x1:x2+search_x]

            matches = cv2.matchTemplate(roi, templ, cv2.TM_SQDIFF)
            matches_norm = matches.copy()
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matches)
            #if minVal > 10000:
            #    continue
            matches_norm = cv2.normalize(matches, matches_norm)
            minVal2, maxVal, minLoc2, maxLoc = cv2.minMaxLoc(matches_norm)
            #if minVal2 > 0.5:
            #    continue
            disparity.append(minLoc[0] + window_size)
            #dy_list.append(minLoc[1] + window_size)
            diff_list.append(minVal)
            keypoints2d.append([keypoint.pt[0], keypoint.pt[1]])

        keypoints2d = np.asarray(keypoints2d)
        keypoints3d = np.empty((keypoints2d.shape[0], 3))
        keypoints3d [:,2] = self.baseline/np.array(disparity)
        keypoints3d [:,0] = ((keypoints2d[:,0] - self.cx)/self.fx)*keypoints3d [:,2]
        keypoints3d [:,1] = ((keypoints2d[:,1] - self.cy)/self.fy)*keypoints3d [:,2]

        return keypoints2d, keypoints3d



