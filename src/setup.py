from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("image_operators",
              sources=["lib/slam_accelerator.pyx",
                       "lib/depth_adjustment_helper.cpp"],
              libraries=["m", "opencv_core", 'omp'],  # Unix-like specific
              extra_compile_args=['-std=c++11','-fopenmp'],
              language='c++'
              )
]

setup(name="Demos",
      ext_modules=cythonize(ext_modules, compiler_directives={'language_level': 3}))
