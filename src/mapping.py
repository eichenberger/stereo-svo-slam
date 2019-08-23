from collections import deque
import numpy as np

# from keyframe import KeyFrame
#from depth_calculator import DepthCalculator
from slam_accelerator import transform_keypoints_inverse
from slam_accelerator import DepthCalculator, KeyFrame, CameraSettings

class Mapping:
    def __init__(self, camera_settings):
        self.stereo_image = deque()
        self.pose = deque()
        self.matches = deque()
        self.cost = deque()
        self.keyframes = deque()
        self.quit = False
        self.camera_settings = camera_settings

        self.depth_calculator = DepthCalculator()

    def new_image(self, stereo_image, pose, matches, cost):
        self.stereo_image.append(stereo_image)
        self.pose.append(pose)
        self.matches.append(matches)
        self.cost.append(cost)
        left = stereo_image[0].left
        grid_height = self.camera_settings.grid_height
        grid_width = self.camera_settings.grid_width
        self.max_matches = int((left.shape[0])/grid_height) \
                *int((left.shape[1])/grid_width)

    def calculate_depth(self, stereo_image, pose):
        keypoints = [None]*len(stereo_image)
        camera_settings = CameraSettings(self.camera_settings)
        camera_settings.window_size = 4

        for i in range(0, len(stereo_image)):
            # Bigger image -> more blocks
            _keypoints = self.depth_calculator.calculate_depth(stereo_image[i],
                                                      camera_settings)

            _keypoints.kps3d = transform_keypoints_inverse(pose, _keypoints.kps3d)
            keypoints[i] = _keypoints

            # Is everything /2 right?
            camera_settings.baseline = camera_settings.baseline/2
            camera_settings.fx = camera_settings.fx/2
            camera_settings.fy = camera_settings.fy/2
            camera_settings.cx = camera_settings.cx/2
            camera_settings.cy = camera_settings.cy/2
            camera_settings.search_x = camera_settings.search_x/2
            camera_settings.search_y = camera_settings.search_y/2
            camera_settings.grid_width = camera_settings.grid_width/2
            camera_settings.grid_height = camera_settings.grid_height/2

        colors = []
        for i in range(0, len(stereo_image)):
            _colors = np.random.randint(0, 255, (len(keypoints[0].kps2d), 3),
                                    dtype=np.uint8)
            _colors = list(map(lambda x: {'r': x[0], 'g': x[1], 'b': x[2]}, _colors))
            colors.append(_colors)

        kf = KeyFrame()

        kf.stereo_images = stereo_image
        kf.pose = pose
        kf.kps = keypoints
        kf.colors = colors
        self.keyframes.append(kf)

    def process_image(self):
        stereo_image = self.stereo_image.popleft()
        pose = self.pose.popleft()
        matches = self.matches.popleft()
        cost = self.cost.popleft()

        # If cost is high, insert a new keyframe. Maybe we need to
        # change that
        if matches < (self.max_matches*0.8) or \
                cost > 2000000:
            print("Insert new keyframe")
            self.calculate_depth(stereo_image, pose)

    def get_last_keyframe(self):
        return self.keyframes[-1]

    def number_of_keyframes(self):
        return len(self.keyframes)

    def quit(self):
        self.quit = True
