from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("slam_accelerator",
              sources=["lib/slam_accelerator.pyx",
                       "lib/slam_accelerator_helper.cpp",
                       "lib/depth_calculator.cpp",
                       "lib/corner_detector.cpp"],
              libraries=["m", "opencv_core", 'omp5', 'opencv_features2d', 'opencv_imgproc'],  # Unix-like specific
              library_dirs=['/usr/local/lib'],
              include_dirs=['/usr/local/include/opencv4'],
              extra_compile_args=['-std=c++17','-fopenmp', '-O3'],
              language='c++'
              )
]

setup(name="SLAM",
      ext_modules=cythonize(ext_modules, compiler_directives={'language_level': 3}))
