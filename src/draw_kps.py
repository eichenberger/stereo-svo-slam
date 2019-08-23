import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_kps(stereo_images, kps, kf):
    for j in range(len(stereo_images)):
        _left_kf = kf.stereo_images[j].left.copy()
        _left = stereo_images[j].left.copy()

        _colors = kf.colors[j]
        _kf_kps = kf.kps[j].kps2d
        for i in range(len(_kf_kps)):
            kp = cv2.KeyPoint(_kf_kps[i]['x'], _kf_kps[i]['y'], 2)
            color = (_colors[i]['r'], _colors[i]['g'], _colors[i]['b'])
            _left_kf = cv2.drawKeypoints(_left_kf, [kp], _left_kf, color=color)

        _kps = kps[j].kps2d
        for i in range(len(_kps)):
            kp = cv2.KeyPoint(_kps[i]['x'], _kps[i]['y'], 2)
            color = (_colors[i]['r'], _colors[i]['g'], _colors[i]['b'])
            _left = cv2.drawKeypoints(_left, [kp], _left, color=color)

        result = np.zeros((_left.shape[0], _left.shape[1]*2, 3), dtype=np.uint8)
        result[:, 0:_left.shape[1], :] = _left_kf
        result[:, _left.shape[1]:2*_left.shape[1], :] = _left

        plt.imshow(result)
        plt.show()

