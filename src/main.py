import cv2
import argparse
import numpy as np
import math
import io

import yaml

from stereo_slam import StereoSLAM

from econ_utils import set_auto_exposure, set_manual_exposure

class EconInput():
    def __init__(self, camera, hidraw, settings):
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 752)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        set_manual_exposure(hidraw, 15000)

        self.settings = cv2.FileStorage(settings, cv2.FILE_STORAGE_READ)

        l_d = self.settings.getNode('LEFT.D').mat()
        l_k = self.settings.getNode('LEFT.K').mat()
        l_r = self.settings.getNode('LEFT.R').mat()
        l_p = self.settings.getNode('LEFT.P').mat()
        l_width = int(self.settings.getNode('LEFT.width').real())
        l_height = int(self.settings.getNode('LEFT.height').real())

        r_d = self.settings.getNode('RIGHT.D').mat()
        r_k = self.settings.getNode('RIGHT.K').mat()
        r_r = self.settings.getNode('RIGHT.R').mat()
        r_p = self.settings.getNode('RIGHT.P').mat()
        r_width = int(self.settings.getNode('RIGHT.width').real())
        r_height = int(self.settings.getNode('RIGHT.height').real())

        self.m1l, self.m2l = cv2.initUndistortRectifyMap(l_k, l_d, l_r, l_p, (l_width, l_height), cv2.CV_32F)
        self.m1r, self.m2r = cv2.initUndistortRectifyMap(r_k, r_d, r_r, r_p, (r_width, r_height), cv2.CV_32F)


    def read(self):
        ret, image = self.cap.read()
        gray_r = cv2.extractChannel(image, 1);
        gray_l = cv2.extractChannel(image, 2);

        gray_l = cv2.remap(gray_l, self.m1l, self.m2l, cv2.INTER_LINEAR)
        gray_r = cv2.remap(gray_r, self.m1r, self.m2r, cv2.INTER_LINEAR)

        return gray_l, gray_r


class VideoInput():
    def __init__(self, left, right, settings):
        self.capl = cv2.VideoCapture(left)
        self.capr = cv2.VideoCapture(right)
        #skip first 10 images
        #self.capl.set(cv2.CAP_PROP_POS_FRAMES, 10)
        #self.capr.set(cv2.CAP_PROP_POS_FRAMES, 10)
        self.settings = cv2.FileStorage(settings, cv2.FILE_STORAGE_READ)

        l_d = self.settings.getNode('LEFT.D').mat()
        l_k = self.settings.getNode('LEFT.K').mat()
        l_r = self.settings.getNode('LEFT.R').mat()
        l_p = self.settings.getNode('LEFT.P').mat()
        l_width = int(self.settings.getNode('LEFT.width').real())
        l_height = int(self.settings.getNode('LEFT.height').real())

        r_d = self.settings.getNode('RIGHT.D').mat()
        r_k = self.settings.getNode('RIGHT.K').mat()
        r_r = self.settings.getNode('RIGHT.R').mat()
        r_p = self.settings.getNode('RIGHT.P').mat()
        r_width = int(self.settings.getNode('RIGHT.width').real())
        r_height = int(self.settings.getNode('RIGHT.height').real())

        self.m1l, self.m2l = cv2.initUndistortRectifyMap(l_k, l_d, l_r, l_p, (l_width, l_height), cv2.CV_32F)
        self.m1r, self.m2r = cv2.initUndistortRectifyMap(r_k, r_d, r_r, r_p, (r_width, r_height), cv2.CV_32F)


    def read(self):
        ret, iml = self.capl.read()
        ret, imr = self.capr.read()

        # This is super slow (why?)
        gray_l = cv2.cvtColor(iml, cv2.COLOR_RGB2GRAY)
        gray_r = cv2.cvtColor(imr, cv2.COLOR_RGB2GRAY)

        tm = cv2.TickMeter()
        tm.start()
        gray_l = cv2.remap(gray_l, self.m1l, self.m2l, cv2.INTER_LINEAR)
        gray_r = cv2.remap(gray_r, self.m1r, self.m2r, cv2.INTER_LINEAR)
        tm.stop()

        print(f'rectification took: {tm.getTimeMilli()} ms')

        return gray_l, gray_r


def main():
    parser = argparse.ArgumentParser(description='Edge slam test')

    subparsers = parser.add_subparsers()
    econ_parser = subparsers.add_parser(name='econ', description='use econ input')
    econ_parser.add_argument('camera', help='camera to use', type=str)
    econ_parser.add_argument('hidraw', help='camera hidraw device', type=str)
    econ_parser.add_argument('settings', help='settings file', type=str)
    econ_parser.set_defaults(func=lambda args: EconInput(args.camera, args.hidraw, args.settings))

    video_parser = subparsers.add_parser(name='video', description='use video input')
    video_parser.add_argument('left', help='left video input', type=str)
    video_parser.add_argument('right', help='right video input', type=str)
    video_parser.add_argument('settings', help='settings file', type=str)
    video_parser.set_defaults(func=lambda args: VideoInput(args.left, args.right, args.settings))

    args = parser.parse_args()
    camera = args.func(args)

    slam = StereoSLAM(45.1932, 680.0, 680.0, 357.0, 225.0)

    gray_l, gray_r = camera.read()
    def read_frame():

        gray_l, gray_r = camera.read()
        tm = cv2.TickMeter()
        tm.start()
        slam.new_image(gray_l, gray_r)
        tm.stop()
        print(f"processing took: {tm.getTimeMilli()} ms")

    key = 0
    while key != ord('q'):
        read_frame()
        key = cv2.waitKey(1)

if __name__ == "__main__":
    main()
