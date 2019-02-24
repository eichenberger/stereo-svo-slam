import cv2
import argparse
import numpy as np
import math
from glumpy import app
import matplotlib.pyplot as plt

from pointcloudviewer import PointCloudViewer, rot_mat

class CornerDetector:
    def __init__(self):
        self.detector = cv2.FastFeatureDetector_create(threshold=8)
        self.keypoints = None
        self.descriptors = None

    def detect_keypoints(self, image):
        self.keypoints = self.detector.detect(image)

class StereoSLAM:
    def __init__(self, baseline, fx, fy, cx, cy):
        self.detector = CornerDetector()
        self.keypoints = None
        self.left = None
        self.right = None
        self.baseline = baseline
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def new_image(self, left, right):
        self.left = left
        self.right = right
        self.detector.detect_keypoints(left)
        self.keypoints = self.detector.keypoints

        self._calculate_depth()

    def _calculate_depth(self):
        window_size = 7
        search_x = 40
        search_y = 3
        dx_list = []
        dy_list = []
        diff_list = []
        keypoints_valid = []

        key = 0
        for index, keypoint in enumerate(self.keypoints):
            diff_max = math.inf
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            x1 = x-window_size
            x2 = x+window_size
            y1 = y-window_size
            y2 = y+window_size

            if x1 < 0 or x2 >= self.left.shape[1] or y1 < search_y or y2 >= (self.left.shape[0]-search_y):
                continue

            templ = self.left[y1:y2,x1:x2]
            roi = self.right[y1-search_y:y2+search_y,x1:x2+search_x]

            matches = cv2.matchTemplate(roi, templ, cv2.TM_SQDIFF)
            matches_norm = matches.copy()
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matches)
            if minVal > 3000:
                continue
            matches_norm = cv2.normalize(matches, matches_norm)
            minVal2, maxVal, minLoc2, maxLoc = cv2.minMaxLoc(matches_norm)
            if minVal2 > 0.005:
                continue
            dx_list.append(minLoc[0] + window_size)
            dy_list.append(minLoc[1] + window_size)
            diff_list.append(minVal)
            keypoints_valid.append(keypoint)


        keypoints_right = [0]*len(dx_list)
        matches = [0]*len(dx_list)
        x = [0]*len(dx_list)
        y = [0]*len(dx_list)
        disp = [0]*len(dx_list)
        for index, keypoint in enumerate(keypoints_valid):
            x[index] = keypoint.pt[0]
            y[index] = keypoint.pt[1]
            disp[index] = dx_list[index]
            keypoints_right[index] = cv2.KeyPoint(x[index]+dx_list[index], y[index]+dy_list[index], diff_list[index])
            matches[index] = cv2.DMatch(index ,index, diff_list[index])

        result = self.left.copy()
        result = cv2.drawMatches(self.left, keypoints_valid, self.right, keypoints_right, matches, result)

        x = np.array(x)
        y = np.array(y)
        points = np.empty((len(keypoints_valid), 3))
        points[:,2] = self.baseline/np.array(disp)
        points[:,0] = ((x - self.cx)/self.fx)*points[:,2]
        points[:,1] = ((y - self.cy)/self.fy)*points[:,2]

        colors = np.empty((len(keypoints_valid), 3))
        intensity = self.left[y.astype(np.uint32),x.astype(np.uint32)]
        intensity = intensity/255.0
        colors[:,0] = intensity
        colors[:,1] = intensity
        colors[:,2] = intensity

        self.points = points
        self.colors = colors
        print("points found: " + str(len(points)))
        cv2.imshow("result", result)


def main():
    parser = argparse.ArgumentParser(description='Edge slam test')
    parser.add_argument('camera', help='camera to use', type=str)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)

    slam = StereoSLAM(45.1932, 680.0, 680.0, 357.0, 225.0)

    pcv = PointCloudViewer()

    pose = np.append(rot_mat([0, 0, 0]), [[0],[0],[0]], axis=1)
    pcv.set_camera_pose(pose)

    def read_frame(dt):
        ret, image = cap.read()

        gray_r = cv2.extractChannel(image, 1);
        gray_l = cv2.extractChannel(image, 2);

        tm = cv2.TickMeter()
        tm.start()
        slam.new_image(gray_l, gray_r)
        tm.stop()
        print(f"processing took: {tm.getTimeSec()}")
        pcv.add_points(slam.points, slam.colors)
        if cv2.waitKey(1) == ord('q'):
            app.quit()


    app.clock.schedule_interval(read_frame, 0.1)
    app.run()

    cap.release()

if __name__ == "__main__":
    main()
