import cv2
import numpy as np
from slam_accelerator import KeyPointType
#import matplotlib.pyplot as plt

def draw_frame(keyframe, frame):
    SCALE = 2
    shape = keyframe.stereo_image.left[0].shape
    shape = (SCALE*shape[1], SCALE*shape[0])
    left_kf = cv2.resize(keyframe.stereo_image.left[0], shape)
    left = cv2.resize(frame.stereo_image.left[0], shape)
    _left_kf = cv2.cvtColor(left_kf, cv2.COLOR_GRAY2RGB)
    _left = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)

    _kf_kps = keyframe.kps.kps2d
    info = keyframe.kps.info
    for i in range(len(_kf_kps)):
        kp = (SCALE*int(_kf_kps[i].x), SCALE*int(_kf_kps[i].y))
        color = (info[i].color['r'], info[i].color['g'], info[i].color['b'])

        if keyframe.kps.info[i].type == KeyPointType.KP_FAST:
            marker = cv2.MARKER_CROSS
        else:
            marker = cv2.MARKER_SQUARE

        _left_kf = cv2.drawMarker(_left_kf, kp, color, markerType=marker, markerSize=10)
        #print(f"kp score: {keyframe.kps.info[i].score}, tpye: {keyframe.kps.info[i].type} \
        #      confidence: {keyframe.kps.info[i].confidence}")
        kp3d = keyframe.kps.kps3d[i]
#        text = f'{kp3d.x.1f}:{kp3d.y:.1f}:{kp3d.z:.1f}'
        text = f'{info[i].keyframe_id}:{kp3d.x:.1f},{kp3d.y:.1f},{kp3d.z:.1f}'
        _left_kf = cv2.putText(_left_kf,
                               text,
                               kp,
                               cv2.FONT_HERSHEY_PLAIN,
                               0.8,
                               (0,0,0))

    text = f"ID: {keyframe.id}"
    _left_kf = cv2.putText(_left_kf,
                           text,
                           (20,20),
                           cv2.FONT_HERSHEY_PLAIN,
                           1,
                           (255,0,0))



    _kps = frame.kps.kps2d
    info = keyframe.kps.info
    for i in range(len(_kps)):
        kp = (SCALE*int(_kps[i].x), SCALE*int(_kps[i].y))
        color = (info[i].color['r'], info[i].color['g'], info[i].color['b'])

        if frame.kps.info[i].type == KeyPointType.KP_FAST:
            marker = cv2.MARKER_CROSS
        else:
            marker = cv2.MARKER_SQUARE

        marker_size = 20;
        _left = cv2.drawMarker(_left, kp, color, markerType=marker, markerSize=marker_size)
        kp3d = frame.kps.kps3d[i]
        text = f'{kp3d.x:.1f},{kp3d.y:.1f},{kp3d.z:.1f}'
        _left = cv2.putText(_left,
                               text,
                               kp,
                               cv2.FONT_HERSHEY_PLAIN,
                               0.8,
                               (0,0,0))

    text = f"ID: {frame.id}"
    _left = cv2.putText(_left,
                        text,
                        (20,20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255,0,0))


    result = np.zeros((_left.shape[0], _left.shape[1]*2, 3), dtype=np.uint8)
    result[:, 0:_left.shape[1], :] = _left_kf
    result[:, _left.shape[1]:2*_left.shape[1], :] = _left

    cv2.namedWindow("result", cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow("result", result)
    key = cv2.waitKey(1)
    if key == ord('s'):
        key = cv2.waitKey(0)

    if key == ord('q'):
        raise("end")

    #plt.imshow(result)
    ##plt.show()
    #plt.pause(0.01)
