import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_kps(stereo_images, kps2d, kf):
    _left_kf = cv2.cvtColor(kf.stereo_images[0].left, cv2.COLOR_GRAY2RGB)
    _left = cv2.cvtColor(stereo_images[0].left, cv2.COLOR_GRAY2RGB)

    _colors = kf.colors
    _kf_kps = kf.kps.kps2d
    for i in range(len(_kf_kps)):
        kp = cv2.KeyPoint(_kf_kps[i]['x'], _kf_kps[i]['y'], 2)
        color = (_colors[i]['r'], _colors[i]['g'], _colors[i]['b'])
        _left_kf = cv2.drawKeypoints(_left_kf, [kp], _left_kf, color=color)
        kp3d = kf.kps.kps3d[i]
        #text = f'{kp3d["x"]:.1f}:{kp3d["y"]:.1f}:{kp3d["z"]:.1f}'
        text = f'{kp3d["z"]:.1f}'
        _left_kf = cv2.putText(_left_kf,
                               text,
                               (int(_kf_kps[i]['x']), int(_kf_kps[i]['y'])),
                               cv2.FONT_HERSHEY_PLAIN,
                               1.0,
                               (255,0,0))

    _kps = kps2d
    for i in range(len(_kps)):
        kp = cv2.KeyPoint(_kps[i]['x'], _kps[i]['y'], 2)
        color = (_colors[i]['r'], _colors[i]['g'], _colors[i]['b'])
        _left = cv2.drawKeypoints(_left, [kp], _left, color=color)

    result = np.zeros((_left.shape[0], _left.shape[1]*2, 3), dtype=np.uint8)
    result[:, 0:_left.shape[1], :] = _left_kf
    result[:, _left.shape[1]:2*_left.shape[1], :] = _left

    plt.imshow(result)
    #plt.show()
    plt.pause(0.05)
