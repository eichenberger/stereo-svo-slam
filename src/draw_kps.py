import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_frame(keyframe, frame):
    _left_kf = cv2.cvtColor(keyframe.stereo_image.left[0], cv2.COLOR_GRAY2RGB)
    _left = cv2.cvtColor(frame.stereo_image.left[0], cv2.COLOR_GRAY2RGB)

    _kf_kps = keyframe.kps.kps2d
    for i in range(len(_kf_kps)):
        kp = cv2.KeyPoint(_kf_kps[i].x, _kf_kps[i].y, 2)
        color = (255, 0, 0)
        _left_kf = cv2.drawKeypoints(_left_kf, [kp], _left_kf, color=color)
        kp3d = keyframe.kps.kps3d[i]
        #text = f'{kp3d["x"]:.1f}:{kp3d["y"]:.1f}:{kp3d["z"]:.1f}'
#        text = f'{kp3d.z:.1f}'
#        _left_kf = cv2.putText(_left_kf,
#                               text,
#                               (int(_kf_kps[i].x, int(_kf_kps[i].y))),
#                               cv2.FONT_HERSHEY_PLAIN,
#                               1.0,
#                               (255,0,0))

    _kps = frame.kps.kps2d
    for i in range(len(_kps)):
        kp = cv2.KeyPoint(_kps[i].x, _kps[i].y, 2)
        color = (0, 255, 0)
        _left = cv2.drawKeypoints(_left, [kp], _left, color=color)

    result = np.zeros((_left.shape[0], _left.shape[1]*2, 3), dtype=np.uint8)
    result[:, 0:_left.shape[1], :] = _left_kf
    result[:, _left.shape[1]:2*_left.shape[1], :] = _left

    plt.imshow(result)
    #plt.show()
    plt.pause(0.01)
