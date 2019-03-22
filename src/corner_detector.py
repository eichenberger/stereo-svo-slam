import cv2
import matplotlib.pyplot as plt
from mat_to_points import mat_to_points

class CornerDetector:
    def __init__(self):
        self.detector = cv2.FastFeatureDetector_create(threshold=7)
        self.keypoints = None
        self.descriptors = None
        self.split_count = 16
        self.kps_per_block = 1
        self.margin = 15

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

                selection_count = 0
                selection = [None]*self.kps_per_block
                for keypoint in self._keypoints:
                    if keypoint.pt[0] < left or keypoint.pt[0] > right \
                            or keypoint.pt[1] < top or keypoint.pt[1] > bottom:
                        continue

                    for k in range(0, self.kps_per_block):
                        if selection[k] == None:
                            selection[k] = keypoint
                            break
                        elif selection[k].response < keypoint.response:
                            # Make sure we have sort selection descending
                            for l in range(k+1, self.kps_per_block):
                                selection[l] = selection[l-1]
                            selection[k] = keypoint
                            break

                selection_count_start = selection_count
                if selection_count < self.kps_per_block:
                    for k in range(left, right):
                        for l in range(top, bottom):
                            for m in range(selection_count_start, self.kps_per_block):
                                if selection[m] == None:
                                    selection[m] = cv2.KeyPoint(k, l, 1, -1, edge[l, k])
                                    break
                                elif selection[m].response < edge[l,k]:
                                    # Make sure we have sort selection descending
                                    for n in range(m+1, self.kps_per_block):
                                        selection[n] = selection[m-1]
                                    selection[m] = cv2.KeyPoint(k, l, 1, -1, edge[l, k])
                                    break

                keypoints.extend(selection)

        #result = cv2.drawKeypoints(edge, keypoints, edge)
        #plt.imshow(result)
        #plt.show()

        return keypoints
