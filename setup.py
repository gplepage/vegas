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

VEGAS_VERSION = '3.3.4'

from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext as _build_ext

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



# create vegas/version.py so vegas knows its version number
with open("src/vegas/_version.py","w") as version_file:
    version_file.write(
        "# File created by vegas setup.py\nversion = '%s'\n"
        % VEGAS_VERSION
        )

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

setup(
	name='vegas',
	version=VEGAS_VERSION,
	description='Tools for adaptive multidimensional Monte Carlo integration.',
	author='G. Peter Lepage',
	author_email='g.p.lepage@cornell.edu',
    cmdclass={'build_ext':build_ext},
	packages=['vegas'],
    package_dir=dict(vegas='src/vegas'),
    package_data=dict(vegas=['../vegas.pxd','_vegas.pxd']),
    ext_modules=ext_modules,
    install_requires=install_requires, # for pip
    requires=requires, # for disutils
    url="https://github.com/gplepage/vegas.git",
    license='GPLv3+',
    platforms='Any',
    long_description="""
    This package provides tools evaluating multidimensional
    integrals numerically using an enhanced version of
	the adaptive Monte Carlo vegas algorithm (G. P. Lepage,
	J. Comput. Phys. 27(1978) 192).
    """,
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
