from collections import deque
import numpy as np

from keyframe import KeyFrame
#from depth_calculator import DepthCalculator
from slam_accelerator import rotation_matrix
from slam_accelerator import DepthCalculator

class Mapping:
    def __init__(self, baseline, fx, fy, cx, cy):
        self.left = deque()
        self.right = deque()
        self.pose = deque()
        self.matches = deque()
        self.cost = deque()
        self.keyframes = deque()
        self.quit = False
        self.split_count = 16
        self.max_matches = self.split_count**2

        self.depth_calculator = DepthCalculator(baseline, fx, fy, cx, cy,
                                                18, 40, 1, 40)

    def new_image(self, left, right, pose, matches, cost):
        self.left.append(left)
        self.right.append(right)
        self.pose.append(pose)
        self.matches.append(matches)
        self.cost.append(cost)

    def calculate_depth(self, left, right, pose):
        keypoints3d = [None]*len(left)
        keypoints2d = [None]*len(left)
        for i in range(0, len(left)):
            split_count = max(4, int(self.split_count*(left[i].shape[0]/left[0].shape[0])))
            n_keypoints = split_count**2
            keypoints3d[i] = np.ones((4, n_keypoints))
            # Bigger image -> more blocks
            keypoints2d[i], keypoints3d[i][0:3,:], err = \
                self.depth_calculator.calculate_depth(left[i], right[i],
                                                      split_count)

            transformation = np.empty((3,4))

            # The transpose is the inverse of the matrix
            transformation[0:3,0:3] = np.transpose(rotation_matrix(pose[3:6]))
            # This is the inverse transformation
            transformation[0:3,3] = np.matmul(-transformation[0:3,0:3],pose[0:3])

            keypoints3d[i] = np.matmul(transformation, keypoints3d[i])

        colors = np.random.randint(0, 255, (keypoints2d[0].shape[1], 3), dtype=np.uint8)

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
