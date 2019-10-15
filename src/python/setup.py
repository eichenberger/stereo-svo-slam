from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

NB_COMPILE_JOBS = 8

# monkey-patch for parallel compilation
def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # parallel code
    N=NB_COMPILE_JOBS # number of parallel compilations
    import multiprocessing.pool
    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N).imap(_single_compile,objects))
    return objects

import distutils.ccompiler
distutils.ccompiler.CCompiler.compile=parallelCCompile


sources = ["wrapper/slam_accelerator.pyx"]
libraries = ["stereosvo"]
#libraries = ["m", "opencv_core", 'omp5', 'opencv_features2d',
#             'opencv_imgproc', 'opencv_calib3d','opencv_video', 'opencv_highgui']
library_dirs = ['../lib']
include_dirs = ['/usr/local/include/opencv4', '../include']

ext_modules_release = [
    Extension("slam_accelerator",
              sources=sources,
              libraries=libraries,  # Unix-like specific
              library_dirs=library_dirs,
              include_dirs=include_dirs,
              extra_compile_args=['-std=c++17','-fopenmp', '-O3','-g0','-s',
                                  '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
              extra_link_args=['-Wl,-rpath,../lib'],
              language='c++'
              )
]

ext_modules_debug = [
    Extension("slam_accelerator",
              sources=sources,
              libraries=libraries,  # Unix-like specific
              library_dirs=library_dirs,
              include_dirs=include_dirs,
              extra_compile_args=['-std=c++17','-fopenmp', '-ggdb', '-O0',
                                  '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
              extra_link_args=['-Wl,-rpath,../lib,-rpath,.'],
              language='c++'
              )
]



_ext_modules = ext_modules_debug
#_ext_modules = ext_modules_release
setup(name="SLAM", ext_modules=cythonize(_ext_modules,
                                         nthreads=NB_COMPILE_JOBS,
                                         compiler_directives={'language_level': 3}))

