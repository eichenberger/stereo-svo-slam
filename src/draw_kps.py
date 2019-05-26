import cv2
import matplotlib.pyplot as plt

from slam_accelerator import transform_keypoints

def draw_kps(pose, new_image, old_image, keypoints2d, keypoints3d, colors, fx, fy, cx, cy):
    estimated_keypoints2d = transform_keypoints(pose,
                                                    keypoints3d,
                                                    fx, fy,
                                                    cx, cy)
    #keypoints_new_ocv = [None]*estimated_keypoints2d.shape[1]
    #keypoints_old_ocv = [None]*estimated_keypoints2d.shape[1]
    #matches = [None]*estimated_keypoints2d.shape[1]

    result = old_image.copy()
    for i in range(estimated_keypoints2d.shape[1]):
        keypoints_new_ocv = [0]*1
        keypoints_old_ocv = [0]*1
        matches = [None]*1
        keypoints_new_ocv[0] = cv2.KeyPoint(estimated_keypoints2d[0, i], estimated_keypoints2d[1, i], 1)
        keypoints_old_ocv[0] = cv2.KeyPoint(keypoints2d[0, i], keypoints2d[1, i], 1)
        matches[0] = cv2.DMatch(0 ,0, 1)
        color = colors[i,:]
        # This function accepts only one color, therefore draw each point
        # separate
        draw_flags = 0 if i == 0 else 1
        result = cv2.drawMatches(old_image, keypoints_old_ocv,
                                 new_image, keypoints_new_ocv,
                                 matches, result,
                                 (int(color[0]), int(color[1]), int(color[2])),
                                 flags=draw_flags)


    cv2.imshow("result", result)

