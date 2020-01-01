cimport numpy as np
import numpy as np
import cv2
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t

from cython.parallel import parallel, prange
from cython import boundscheck, wraparound
from cpython.ref cimport PyObject
from cython cimport view
from libc.math cimport fabs
from libc.math cimport sin, cos
from libc.string cimport memcpy

ctypedef double (*opt_fun)(const double*)

cimport stereo_slam

from stereo_slam_types cimport KeyPointType as _KeyPointType
from stereo_slam_types cimport CameraSettings as _CameraSettings
from stereo_slam_types cimport Color as _Color
from stereo_slam_types cimport Frame as _Frame
from stereo_slam_types cimport KeyFrame as _KeyFrame
from stereo_slam_types cimport KeyPoint2d as _KeyPoint2d
from stereo_slam_types cimport KeyPoint3d as _KeyPoint3d
from stereo_slam_types cimport KeyPointInformation as _KeyPointInformation
from stereo_slam_types cimport KeyPoints as _KeyPoints
from stereo_slam_types cimport Mat as _Mat
from stereo_slam_types cimport Pose as _Pose
from stereo_slam_types cimport StereoImage as _StereoImage
from stereo_slam_types cimport KalmanFilter as _KalmanFilter


cdef _c2np(_Mat image):
    cdef _Mat _copy
    image.copyTo(_copy)
    rows = image.rows
    cols = image.cols
    _array = np.asarray(<np.uint8_t[:rows,:cols]> _copy.data)
    return _array.copy()

cdef _np2c(pyimage, _Mat &cimage):
    r = pyimage.shape[0]
    c = pyimage.shape[1]
    cimage.create(r, c, cv2.CV_8UC1)
    cdef unsigned char[:,:] image_buffer = pyimage
    memcpy(cimage.data, &image_buffer[0,0], r*c)


cdef class StereoSlam:
    cdef stereo_slam.StereoSlam *_stereo_slam

    def __cinit__(self, CameraSettings camera_settings):
        self._stereo_slam = new stereo_slam.StereoSlam(camera_settings.inst);

    def __dealloc__(self):
        del self._stereo_slam

    def new_image(self, left, right, float dt):
        cdef _Mat _left
        cdef _Mat _right

        _np2c(left, _left)
        _np2c(right, _right)

        self._stereo_slam.new_image(_left, _right, dt)

    def get_keyframe(self):
        kf = KeyFrame()
        self._stereo_slam.get_keyframe(kf.inst)

        return kf


    def get_frame(self):
        frame = Frame()
        self._stereo_slam.get_frame(frame.inst)

        return frame

    def get_keyframes(self):
        cdef vector[_KeyFrame] _keyframes
        self._stereo_slam.get_keyframes(_keyframes)

        keyframes = []
        for i in range(_keyframes.size()):
            kf = KeyFrame()
            kf.inst = _keyframes[i]
            keyframes.append(kf)

        return keyframes


cdef class StereoImage:
    """
    Cython implementation of _StereoImage
    """

    cdef _StereoImage inst

    property left:
        def __set__(self, list left):
            cdef vector[_Mat] v0
            cdef _Mat _item0
            for item0 in left:
                _np2c(item0, _item0)
                v0.push_back(_item0)
            self.inst.left = v0

        def __get__(self):
            _r = self.inst.left
            py_result = []
            for i in range(_r.size()):
               py_result.append(_c2np(_r[i]))
            return py_result

    property right:
        def __set__(self, list right):
            cdef vector[_Mat] v0
            cdef _Mat _item0
            for item0 in right:
                _np2c(item0, _item0)
                v0.push_back(_item0)
            self.inst.right = v0

        def __get__(self):
            _r = self.inst.right
            py_result = []
            for i in range(_r.size()):
               py_result.append(_c2np(_r[i]))
            return py_result

###### Most stuff was autogenerated with autowrap below here ######

cdef class CameraSettings:
    """
    Cython implementation of _CameraSettings
    """

    cdef _CameraSettings inst

    property baseline:
        def __set__(self, float baseline):

            self.inst.baseline = (<float>baseline)


        def __get__(self):
            cdef float _r = self.inst.baseline
            py_result = <float>_r
            return py_result

    property fx:
        def __set__(self, float fx):

            self.inst.fx = (<float>fx)


        def __get__(self):
            cdef float _r = self.inst.fx
            py_result = <float>_r
            return py_result

    property fy:
        def __set__(self, float fy):

            self.inst.fy = (<float>fy)


        def __get__(self):
            cdef float _r = self.inst.fy
            py_result = <float>_r
            return py_result

    property cx:
        def __set__(self, float cx):

            self.inst.cx = (<float>cx)


        def __get__(self):
            cdef float _r = self.inst.cx
            py_result = <float>_r
            return py_result

    property cy:
        def __set__(self, float cy):

            self.inst.cy = (<float>cy)


        def __get__(self):
            cdef float _r = self.inst.cy
            py_result = <float>_r
            return py_result

    property k1:
        def __set__(self, float k1):

            self.inst.k1 = (<float>k1)


        def __get__(self):
            cdef float _r = self.inst.k1
            py_result = <float>_r
            return py_result

    property k2:
        def __set__(self, float k2):

            self.inst.k2 = (<float>k2)


        def __get__(self):
            cdef float _r = self.inst.k2
            py_result = <float>_r
            return py_result

    property k3:
        def __set__(self, float k3):

            self.inst.k3 = (<float>k3)


        def __get__(self):
            cdef float _r = self.inst.k3
            py_result = <float>_r
            return py_result

    property p1:
        def __set__(self, float p1):

            self.inst.p1 = (<float>p1)


        def __get__(self):
            cdef float _r = self.inst.p1
            py_result = <float>_r
            return py_result

    property p2:
        def __set__(self, float p2):

            self.inst.p2 = (<float>p2)


        def __get__(self):
            cdef float _r = self.inst.p2
            py_result = <float>_r
            return py_result

    property grid_height:
        def __set__(self,  grid_height):

            self.inst.grid_height = (<int>grid_height)


        def __get__(self):
            cdef int _r = self.inst.grid_height
            py_result = <int>_r
            return py_result

    property grid_width:
        def __set__(self,  grid_width):

            self.inst.grid_width = (<int>grid_width)


        def __get__(self):
            cdef int _r = self.inst.grid_width
            py_result = <int>_r
            return py_result

    property search_x:
        def __set__(self,  search_x):

            self.inst.search_x = (<int>search_x)


        def __get__(self):
            cdef int _r = self.inst.search_x
            py_result = <int>_r
            return py_result

    property search_y:
        def __set__(self,  search_y):

            self.inst.search_y = (<int>search_y)


        def __get__(self):
            cdef int _r = self.inst.search_y
            py_result = <int>_r
            return py_result

    property window_size:
        def __set__(self,  window_size):

            self.inst.window_size = (<int>window_size)


        def __get__(self):
            cdef int _r = self.inst.window_size
            py_result = <int>_r
            return py_result

    property window_size_opt_flow:
        def __set__(self,  window_size_opt_flow):

            self.inst.window_size_opt_flow = (<int>window_size_opt_flow)


        def __get__(self):
            cdef int _r = self.inst.window_size_opt_flow
            py_result = <int>_r
            return py_result

    property max_pyramid_levels:
        def __set__(self,  max_pyramid_levels):

            self.inst.max_pyramid_levels = (<int>max_pyramid_levels)


        def __get__(self):
            cdef int _r = self.inst.max_pyramid_levels
            py_result = <int>_r
            return py_result

    property window_size_depth_calculator:
        def __set__(self, window_size_depth_calculator):
            self.inst.window_size_depth_calculator = (<int>window_size_depth_calculator)


        def __get__(self):
            cdef int _r = self.inst.window_size_depth_calculator
            py_result = <int>_r
            return py_result


    property min_pyramid_level_pose_estimation:
        def __set__(self, min_pyramid_level_pose_estimation):
            self.inst.min_pyramid_level_pose_estimation= (<int>min_pyramid_level_pose_estimation)


        def __get__(self):
            cdef int _r = self.inst.min_pyramid_level_pose_estimation
            py_result = <int>_r
            return py_result


cdef class Frame:
    """
    Cython implementation of _Frame
    """

    cdef _Frame inst

    property id:
        def __set__(self, uint64_t id):
            self.inst.id = id


        def __get__(self):
            return self.inst.id


    property pose:
        def __set__(self, Pose pose):

            self.inst.pose.set_pose(pose.inst)


        def __get__(self):
            cdef Pose py_result = Pose()
            py_result.inst = self.inst.pose.get_pose()
            return py_result

    property stereo_image:
        def __set__(self, StereoImage stereo_image):

            self.inst.stereo_image = (stereo_image.inst)


        def __get__(self):
            cdef StereoImage py_result = StereoImage()
            py_result.inst = self.inst.stereo_image
            return py_result

    property kps:
        def __set__(self, KeyPoints kps):

            self.inst.kps = (kps.inst)


        def __get__(self):
            cdef KeyPoints py_result = KeyPoints()
            py_result.inst = self.inst.kps
            return py_result

cdef class KeyFrame:
    """
    Cython implementation of _KeyFrame
    """

    cdef _KeyFrame inst

    property id:
        def __set__(self, uint64_t id):

            self.inst.id = id


        def __get__(self):
            return self.inst.id

    property pose:
        def __set__(self, Pose pose):

            self.inst.pose.set_pose(pose.inst)


        def __get__(self):
            cdef Pose py_result = Pose()
            py_result.inst = self.inst.pose.get_pose()
            return py_result

    property stereo_image:
        def __set__(self, StereoImage stereo_image):

            self.inst.stereo_image = (stereo_image.inst)


        def __get__(self):
            cdef StereoImage py_result = StereoImage()
            py_result.inst = self.inst.stereo_image
            return py_result

    property kps:
        def __set__(self, KeyPoints kps):

            self.inst.kps = (kps.inst)


        def __get__(self):
            cdef KeyPoints py_result = KeyPoints()
            py_result.inst = self.inst.kps
            return py_result

cdef class KeyPoint2d:
    """
    Cython implementation of _KeyPoint2d
    """

    cdef _KeyPoint2d inst

    property x:
        def __set__(self, float x):

            self.inst.x = (<float>x)


        def __get__(self):
            cdef float _r = self.inst.x
            py_result = <float>_r
            return py_result

    property y:
        def __set__(self, float y):

            self.inst.y = (<float>y)


        def __get__(self):
            cdef float _r = self.inst.y
            py_result = <float>_r
            return py_result

cdef class KeyPoint3d:
    """
    Cython implementation of _KeyPoint3d
    """

    cdef _KeyPoint3d inst

    property x:
        def __set__(self, float x):

            self.inst.x = (<float>x)


        def __get__(self):
            cdef float _r = self.inst.x
            py_result = <float>_r
            return py_result

    property y:
        def __set__(self, float y):

            self.inst.y = (<float>y)


        def __get__(self):
            cdef float _r = self.inst.y
            py_result = <float>_r
            return py_result

    property z:
        def __set__(self, float z):

            self.inst.z = (<float>z)


        def __get__(self):
            cdef float _r = self.inst.z
            py_result = <float>_r
            return py_result

cdef class KeyPointInformation:
    """
    Cython implementation of _KeyPointInformation
    """

    cdef _KeyPointInformation inst

    def __cinit__(self):
        self.inst.kf = _KalmanFilter()

    property score:
        def __set__(self, float score):

            self.inst.score = (<float>score)


        def __get__(self):
            cdef float _r = self.inst.score
            py_result = <float>_r
            return py_result

    property level:
        def __set__(self,  level):

            self.inst.level = (<int>level)


        def __get__(self):
            cdef int _r = self.inst.level
            py_result = <int>_r
            return py_result

    property type:
        def __set__(self, int type):

            self.inst.type = (<_KeyPointType>type)


        def __get__(self):
            cdef _KeyPointType _r = self.inst.type
            py_result = <int>_r
            return py_result

    property confidence:
        def __set__(self, float confidence):

            self.inst.confidence = (<float>confidence)


        def __get__(self):
            cdef float _r = self.inst.confidence
            py_result = <float>_r
            return py_result


    property keyframe_id:
        def __set__(self, uint64_t id):
            self.inst.keyframe_id = id

        def __get__(self):
            return self.inst.keyframe_id

    property color:
        def __set__(self, color):
            self.inst.color = color

        def __get__(self):
            return self.inst.color


cdef class KeyPoints:
    """
    Cython implementation of _KeyPoints
    """

    cdef _KeyPoints inst

    property kps2d:
        def __set__(self, list kps2d):
            cdef vector[_KeyPoint2d] v0
            cdef KeyPoint2d item0
            for item0 in kps2d:
                v0.push_back(item0.inst)
            self.inst.kps2d = v0

        def __get__(self):
            _r = self.inst.kps2d
            py_result = []
            cdef KeyPoint2d item_py_result
            for i in range(0, _r.size()):
                kp2d = KeyPoint2d()
                kp2d.inst = _r[i]
                py_result.append(kp2d)
            return py_result

    property kps3d:
        def __set__(self, list kps3d):
            cdef vector[_KeyPoint3d] v0
            cdef KeyPoint3d item0
            for item0 in kps3d:
                v0.push_back(item0.inst)
            self.inst.kps3d = v0

        def __get__(self):
            _r = self.inst.kps3d
            py_result = []
            cdef KeyPoint3d item_py_result
            for i in range(0, _r.size()):
                kp3d = KeyPoint3d()
                kp3d.inst = _r[i]
                py_result.append(kp3d)
            return py_result

    property info:
        def __set__(self, list info):
            cdef vector[_KeyPointInformation] v0
            cdef KeyPointInformation item0
            for item0 in info:
                v0.push_back(item0.inst)
            self.inst.info = v0

        def __get__(self):
            _r = self.inst.info
            py_result = []
            cdef KeyPointInformation item_py_result
            for i in range(0, _r.size()):
                kpi = KeyPointInformation()
                kpi.inst = _r[i]
                py_result.append(kpi)
            return py_result

cdef class Pose:
    """
    Cython implementation of _Pose
    """

    cdef _Pose inst

    property x:
        def __set__(self, float x):

            self.inst.x = (<float>x)


        def __get__(self):
            cdef float _r = self.inst.x
            py_result = <float>_r
            return py_result

    property y:
        def __set__(self, float y):

            self.inst.y = (<float>y)


        def __get__(self):
            cdef float _r = self.inst.y
            py_result = <float>_r
            return py_result

    property z:
        def __set__(self, float z):

            self.inst.z = (<float>z)


        def __get__(self):
            cdef float _r = self.inst.z
            py_result = <float>_r
            return py_result

    property pitch:
        def __set__(self, float pitch):

            self.inst.pitch = (<float>pitch)


        def __get__(self):
            cdef float _r = self.inst.pitch
            py_result = <float>_r
            return py_result

    property yaw:
        def __set__(self, float yaw):

            self.inst.yaw = (<float>yaw)


        def __get__(self):
            cdef float _r = self.inst.yaw
            py_result = <float>_r
            return py_result

    property roll:
        def __set__(self, float roll):

            self.inst.roll = (<float>roll)


        def __get__(self):
            cdef float _r = self.inst.roll
            py_result = <float>_r
            return py_result


cdef class KeyPointType:
    KP_FAST = 0
    KP_EDGELET = 1
