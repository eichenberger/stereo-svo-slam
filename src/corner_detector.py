import cv2
import matplotlib.pyplot as plt
from mat_to_points import mat_to_points

class CornerDetector:
    def __init__(self, split_count):
        self.detector = cv2.FastFeatureDetector_create(threshold=3)
        self.keypoints = None
        self.descriptors = None
        self.split_count = split_count
        self.kps_per_block = 1
        self.margin = 12

    def detect_keypoints(self, image):
        self._keypoints = self.detector.detect(image)
        edge = cv2.Sobel(image, cv2.CV_8U, 1, 0)

        return self._distribute_keypoints(image, edge)

    def _distribute_keypoints(self, image, edge):
        sub_width = (image.shape[1]-2*self.margin)/self.split_count
        sub_height = (image.shape[0]-2*self.margin)/self.split_count

        keypoints = []
        for i in range(0, self.split_count):
            for j in range(0, self.split_count):
                left = self.margin + int(i * sub_width)
                right = self.margin + int((i+1) * sub_width - 1)
                top = self.margin + int(j * sub_height)
                bottom = self.margin + int((j+1) * sub_height - 1)

                selection = None
                for keypoint in self._keypoints:
                    if keypoint.pt[0] < left or keypoint.pt[0] > right \
                            or keypoint.pt[1] < top or keypoint.pt[1] > bottom:
                        continue

                    if selection == None:
                        selection = keypoint
                        continue

                    elif selection.response < keypoint.response:
                        selection = keypoint
                        break

                if selection == None:
                    for k in range(left, right):
                        for l in range(top, bottom):
                            if selection == None:
                                selection = cv2.KeyPoint(k, l, 1, -1, edge[l, k])
                                break
                            elif selection.response < edge[l,k]:
                                selection = cv2.KeyPoint(k, l, 1, -1, edge[l, k])
                                break

                keypoints.append(selection)

        return keypoints
