from __future__ import print_function

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

import numpy as np


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [
    # Extension(
    #     name='extensions.layers.iou.src.cython_iou',
    #     sources=['extensions/layers/iou/src/cython_iou.pyx'],
    #     extra_compile_args=['-Wno-cpp'],
    #     include_dirs=[numpy_include]
    # ),
    Extension(
        name='extensions.nms.src.cython_nms',
        sources=['extensions/nms/src/cython_nms.pyx'],
        extra_compile_args=['-Wno-cpp'],
        include_dirs=[numpy_include]
    )
]

setup(
    name='extensions',
    packages=['extensions'],
    ext_modules=cythonize(ext_modules)
)