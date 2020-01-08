# Stereo SVO SLAM Library

This repository contains a proof of concept for a SVO based stereo camera SLAM library.

The repository is organized as follows:
 Direcotry | Description                                                                  
-----------|------------------------------------------------------------------------------
 doc       | Documentation of the whole algorithm, implementation, etc.                   
 src       | Source code of library, test application, demo, qt viewer and python wrapper 
 test      | Test scripts, test videos, etc.                                              

The source code includes some doxygen comments. Check out the documentation folder for indepth information.

## Library

The library allows to process stereo images and calculates the camera position based on this images

## Test application

![test applicaiton](https://github.com/eichenberger/stereo_svo_slam/raw/master/doc/img/test_app.png "Test applicaiton")

The test application allows to process input images from different sources like Econ Tara Camera, EuRoC dataset or video input. It requires a YAML file with camera parameters. See src/app/Blender.yaml for more details.

## Demo application

![demo applicaiton](https://github.com/eichenberger/stereo_svo_slam/raw/master/doc/img/demo_app2.png "Demo applicaiton")

The demo application is a simple ar-application which shows what a SLAM library can do. It only supports Econ Tara an requires a YAML file with camera settings (src/app/Econ.yaml).

## Qt 3D Viewer

![demo applicaiton](https://github.com/eichenberger/stereo_svo_slam/raw/master/doc/img/qt_viewer2.png "Qt 3D Viewer")

The Qt 3D viewer can connect to the test application to show keyframes, current pose and trajectory.
