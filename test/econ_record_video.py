import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np
import argparse
import io

def set_manual_exposure(hidraw, value):
    MAX_EXPOSURE=300000
    if value >= MAX_EXPOSURE:
        print(f'Exposure must be less than {MAX_EXPOSURE} (is {value})')
        return
    f = io.open(hidraw, 'wb', buffering=0)
    data = bytes([0x78, 0x02, (value >> 24)&0xFF, (value >> 16)&0xFF, (value>>8)&0xFF, value&0xFF])
    f.write(data)
    f.close()

def set_auto_exposure(hidraw):
    set_manual_exposure(hidraw, 1)

def main():
    WIDTH = 752
    HEIGHT = 480
    parser = argparse.ArgumentParser(description='OpenCV test')
    parser.add_argument('camera', help='camera to use', type=str)
    parser.add_argument('hidraw', help='hdiraw control device', type=str)
    parser.add_argument('exposure', help='exposure value', type=int)
    parser.add_argument('out', help='output name', type=str)

    np.set_printoptions(linewidth=200, suppress=True)

    args = parser.parse_args()

    cap1 = cv2.VideoCapture(args.camera)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    set_manual_exposure(args.hidraw, args.exposure)

    key = 0
    i = 0

    out_image = np.zeros((HEIGHT, 2*WIDTH, 3), np.uint8)
    video_writer = cv2.VideoWriter(args.out + ".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (2*WIDTH, HEIGHT))
    while key != ord('q'):
        ret, image1 = cap1.read()
        gray_l = cv2.cvtColor(cv2.extractChannel(image1, 1), cv2.COLOR_GRAY2RGB);
        gray_r = cv2.cvtColor(cv2.extractChannel(image1, 2), cv2.COLOR_GRAY2RGB);

        out_image[:,0:WIDTH] = gray_l
        out_image[:,WIDTH:2*WIDTH] = gray_r

        video_writer.write(out_image)

        cv2.imshow("output", out_image)
        key = cv2.waitKey(1)

    cap1.release()
    video_writer.release()

if __name__ == "__main__":
    main()
