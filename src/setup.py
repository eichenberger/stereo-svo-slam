from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("slam_accelerator",
              sources=["lib/slam_accelerator.pyx",
                       "lib/slam_accelerator_helper.cpp"],
              libraries=["m", "opencv_core", 'omp5'],  # Unix-like specific
              extra_compile_args=['-std=c++11','-fopenmp'],
              language='c++'
              )
]

setup(name="SLAM",
      ext_modules=cythonize(ext_modules, compiler_directives={'language_level': 3}))
