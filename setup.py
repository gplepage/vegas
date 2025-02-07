from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_args = dict(
    include_dirs=[numpy.get_include()],  # not needed anymore
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[]
    )

ext_modules = [
    Extension('vegas._vegas', ['src/vegas/_vegas.pyx'], **ext_args),
    ]

setup(ext_modules=cythonize(ext_modules))
