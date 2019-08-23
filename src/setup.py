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


sources = ["lib/slam_accelerator.pyx",
            "lib/slam_accelerator_helper.cpp",
            "lib/transform_keypoints.cpp",
            "lib/depth_calculator.cpp",
            "lib/image_comparison.cpp",
            "lib/corner_detector.cpp"]
libraries = ["m", "opencv_core", 'omp5', 'opencv_features2d',
             'opencv_imgproc', 'opencv_calib3d']
library_dirs = ['/usr/local/lib']
include_dirs = ['/usr/local/include/opencv4']

ext_modules_release = [
    Extension("slam_accelerator",
              sources=sources,
              libraries=libraries,  # Unix-like specific
              library_dirs=library_dirs,
              include_dirs=include_dirs,
              extra_compile_args=['-std=c++17','-fopenmp', '-O3', '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
              language='c++'
              )
]

ext_modules_debug = [
    Extension("slam_accelerator",
              sources=sources,
              libraries=libraries,  # Unix-like specific
              library_dirs=library_dirs,
              include_dirs=include_dirs,
              extra_compile_args=['-std=c++17','-fopenmp', '-ggdb', '-O0', '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
              language='c++'
              )
]



ext_modules = ext_modules_debug
setup(name="SLAM", ext_modules=cythonize(ext_modules,
                                         nthreads=NB_COMPILE_JOBS,
                                         compiler_directives={'language_level': 3}))

