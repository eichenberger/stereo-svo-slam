from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("image_operators",
              sources=["lib/image_operators.pyx"],
              libraries=["m"]  # Unix-like specific
              )
]

setup(name="Demos",
      ext_modules=cythonize(ext_modules))
