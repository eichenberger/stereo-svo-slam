import cv2
import matplotlib.pyplot as plt

from slam_accelerator import transform_keypoints

def draw_kps(pose, new_image, old_image, keypoints2d, keypoints3d, fx, fy, cx, cy):
            estimated_keypoints2d = transform_keypoints(pose,
                                                        keypoints3d,
                                                        fx, fy,
                                                        cx, cy)
            keypoints_new_ocv = [None]*estimated_keypoints2d.shape[1]
            keypoints_old_ocv = [None]*estimated_keypoints2d.shape[1]
            matches = [None]*estimated_keypoints2d.shape[1]

            for i in range(estimated_keypoints2d.shape[1]):
                keypoints_new_ocv [i] = cv2.KeyPoint(estimated_keypoints2d[0, i], estimated_keypoints2d[1, i], 1)
                keypoints_old_ocv [i] = cv2.KeyPoint(keypoints2d[0, i], keypoints2d[1, i], 1)
                matches[i] = cv2.DMatch(i ,i, 1)

            result = old_image.copy()
            result = cv2.drawMatches(old_image,
                                     keypoints_old_ocv,
                                     new_image,
                                     keypoints_new_ocv,
                                     matches, result)

            cv2.imshow("result", result)

