""" to build modules in module_list:   python setup.py build_ext --inplace

Created by G. Peter Lepage (Cornell University) in 12/2013.
Copyright (c) 2013-18 G. Peter Lepage.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version (see <http://www.gnu.org/licenses/>).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

VEGAS_VERSION = '3.3.6'

from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build_py import build_py as _build_py

# compile from existing .c files if USE_CYTHON is False
USE_CYTHON = False # True

class build_ext(_build_ext):
    # delays using numpy and cython until they are installed;
    # cython is optional (set USE_CYTHON)
    # this code adapted from https://github.com/pandas-dev/pandas setup.py
    def build_extensions(self):
        import numpy
        if USE_CYTHON:
            from Cython.Build import cythonize
            self.extensions = cythonize(self.extensions)
        numpy_include = numpy.get_include()
        for ext in self.extensions:
            ext.include_dirs.append(numpy_include)
        _build_ext.build_extensions(self)

class build_py(_build_py):
    # adds version info
    def run(self):
        """ Append version number to vegas/__init__.py """
        with open('src/vegas/__init__.py', 'a') as vfile:
            vfile.write("\n__version__ = '%s'\n" % VEGAS_VERSION)
        _build_py.run(self)

ext_args = dict(
    libraries=[],
    include_dirs=[],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[],
    )
data_files = [('vegas_include', ['vegas.pxd'])]

ext = '.pyx' if USE_CYTHON else '.c'

ext_modules = [
    Extension('vegas._vegas', ['src/vegas/_vegas' + ext], **ext_args),
    ]

# for distutils:
requires = (
    ["cython (>=0.17)","numpy (>=1.7)", "gvar (>=8.0)"]
    if USE_CYTHON else
    ["numpy (>=1.7)", "gvar (>=8.0)"]
    )
# for pip:
install_requires = (
    ['cython>=0.17', 'numpy>=1.7', 'gvar>=8.0']
    if USE_CYTHON else
    ['numpy>=1.7', 'gvar>=8.0']
    )

# pypi
with open('README.rst', 'r') as file:
    long_description = file.read()

setup(
	name='vegas',
	version=VEGAS_VERSION,
	description='Tools for adaptive multidimensional Monte Carlo integration.',
	author='G. Peter Lepage',
	author_email='g.p.lepage@cornell.edu',
    cmdclass={'build_ext':build_ext, 'build_py':build_py},
	packages=['vegas'],
    package_dir=dict(vegas='src/vegas'),
    package_data=dict(vegas=['../vegas.pxd','_vegas.pxd']),
    ext_modules=ext_modules,
    install_requires=install_requires, # for pip
    requires=requires, # for disutils
    url="https://github.com/gplepage/vegas.git",
    license='GPLv3+',
    platforms='Any',
    long_description=long_description,
    classifiers = [                     #
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering'
        ]

	)
