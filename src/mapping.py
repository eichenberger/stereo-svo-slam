from collections import deque
import cv2
import numpy as np

from keyframe import KeyFrame
from depth_calculator import DepthCalculator
from slam_accelerator import rotation_matrix

class Mapping:
    def __init__(self, baseline, fx, fy, cx, cy):
        self.left = deque()
        self.right = deque()
        self.pose = deque()
        self.matches = deque()
        self.cost = deque()
        self.keyframes = deque()
        self.quit = False
        self.split_count = 15
        self.max_matches = self.split_count**2

        self.depth_calculator = DepthCalculator(baseline, fx, fy, cx, cy,
                                                self.split_count)

    def new_image(self, left, right, pose, matches, cost):
        self.left.append(left)
        self.right.append(right)
        self.pose.append(pose)
        self.matches.append(matches)
        self.cost.append(cost)

    def calculate_depth(self, left, right, pose):
        keypoints3d = np.ones((4,self.max_matches))
        keypoints2d, keypoints3d[0:3,:], err = \
            self.depth_calculator.calculate_depth(left, right)

        transformation = np.empty((3,4))
        # Why is pinv not the same as rotation matrix from minus angle!
        transformation[0:3,0:3] = np.linalg.pinv(rotation_matrix(pose[3:6]))
        transformation[0:3,3] = -pose[0:3]

        keypoints3d = np.matmul(transformation, keypoints3d)

        colors = np.random.randint(0, 255, (keypoints2d.shape[1], 3), dtype=np.uint8)

        self.keyframes.append(KeyFrame(left, right,
                                       keypoints2d, keypoints3d,
                                       pose, colors))

    def process_image(self):
        left = self.left.popleft()
        right = self.right.popleft()
        pose = self.pose.popleft()
        matches = self.matches.popleft()
        cost = self.cost.popleft()

        # If cost is high, insert a new keyframe. Maybe we need to
        # change that
        if matches < (self.max_matches*0.8) or \
                cost > 2000000:
            print("Insert new keyframe")
            self.calculate_depth(left, right, pose)

    def get_last_keyframe(self):
        return self.keyframes[-1]

    def number_of_keyframes(self):
        return len(self.keyframes)

    def quit(self):
        self.quit = True
